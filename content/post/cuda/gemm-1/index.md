---
title: "GEMM 1"
slug: "gemm-1"
date: "2025-10-26"
categories:
  - cuda
tags:
  - cuda
  - nvidia
---

This post is my study notes based on the excellent [worklog](https://siboehm.com/articles/22/CUDA-MMM). I rewrote the kernels and diagrams myself while following along.

CUDA provides highly optimized GEMM APIs in cuBLAS, but with enough care we can write custom kernels that approach cuBLAS performance. Below, we will gradually apply core CUDA optimization ideas to a simple GEMM kernel.

- A: (M, K), row-major
- B: (K, N), row-major
- C: (M, N), row-major
- DRAM: Global memory
- SRAM: Shared memory

We implement a series of kernels and compare their performance. The results are:

1. Naive implementation, DRAM coalescing
2. SRAM caching
3. SRAM 1d tiling

```
[BENCHMARK]                    CUBLAS GEMM │ 0.045334 ms (w:10 r:20)
[BENCHMARK]               GPU GEMM 0 NAIVE │ 3.943722 ms (w:10 r:20) [PASSED]
[BENCHMARK]     GPU GEMM 0 DRAM COALESCING │ 0.517949 ms (w:10 r:20) [PASSED]
[BENCHMARK]        GPU GEMM 1 SRAM CACHING │ 0.248670 ms (w:10 r:20) [PASSED]
[BENCHMARK]      GPU GEMM 2 SRAM 1D TILING │ 0.249046 ms (w:10 r:20) [PASSED]
```

## 0. Naive implementation

![](images/image.png)

The most basic implementation assigns one element of $C$ to each thread. Because the matrices are stored in row-major order, this layout forces us to read non-contiguous memory for $B$. The following figure shows how this looks at the warp level.

![](images/image-1.png)

In this loop structure, when loading $A$ each thread in a warp accesses a different column, so the accesses are not contiguous and cannot be coalesced. Without coalescing, the hardware effectively has to service 32 separate loads per warp, which significantly degrades performance.

When loading $B$, all threads in the warp access the same element, so the hardware can use a broadcast. Even so, given the overall access pattern, there is little benefit from grouping these threads into a single warp in this naïve mapping.

### DRAM coalescing

![](images/image-2.png)

To leverage the warp properly, we need to access contiguous memory as shown above. When loading $A$, we can exploit broadcast to share values within the warp. When loading $B$, each thread in the warp accesses adjacent elements, so the hardware can coalesce the loads into 128‑byte transactions.

```cpp
__global__ void gemm_gpu_0_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = tid % N;
  int col = tid / N;
 ...
}

__global__ void gemm_gpu_0_dram_coalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = tid / N;
  int col = tid % N;
  ...
}
```

The only difference between the two kernels is how `row` and `col` are computed, yet the performance difference is large. Profiling shows that this gap comes from DRAM operations.

```bash
# RTX 5090
$ sudo /usr/local/cuda/bin/ncu --metrics dram__bytes.sum.per_second gemm
  gemm_gpu_0_naive(int, int, int, float, float *, float *, float, float *) (4096, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    dram__bytes.sum.per_second     Gbyte/s         2.40
    -------------------------- ----------- ------------

  gemm_gpu_0_dram_coalescing(int, int, int, float, float *, float *, float, float *) (4096, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    dram__bytes.sum.per_second     Gbyte/s        18.07
    -------------------------- ----------- ------------
```

## 1. SRAM caching
In the naïve kernel, the same data is fetched from DRAM many times, which is very expensive. According to this [paper](https://arxiv.org/abs/1804.06826), on a V100 the DRAM bandwidth is about 900 GB/s, while the shared-memory (SRAM) bandwidth is around 13,800 GB/s (the exact SRAM number is not officially documented). We therefore want to cache data in shared memory and reuse it as much as possible.

![](images/image-3.png)

Each block is responsible for a 32×32 tile of $C$. The shaded regions in the figure indicate the corresponding tiles of $A$ and $B$ that must be loaded from DRAM. The `bkIdx` loop walks along $K$ in chunks of `BLOCKSIZE`, loading tiles into shared memory; the `tIter` loop then performs the GEMM on those tiles. Since each thread produces a single output element, it accumulates the partial result into a scalar `sum`, which is finally written back to the appropriate location in $C$.

```cpp
template <int BLOCKSIZE>
__global__ void gemm_gpu_1_sram_caching(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int bkRow = blockIdx.y;
    int bkCol = blockIdx.x;

    A += K * BLOCKSIZE * bkRow;
    B += BLOCKSIZE * bkCol;
    C += N * BLOCKSIZE * bkRow + BLOCKSIZE * bkCol;

    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    int tRow = threadIdx.x / BLOCKSIZE;
    int tCol = threadIdx.x % BLOCKSIZE;

    float sum = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        sA[threadIdx.x] = A[tRow * K + tCol];
        sB[threadIdx.x] = B[tRow * N + tCol];
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        
        for (int tIter = 0; tIter < BLOCKSIZE; tIter++)
        {
            sum += sA[tRow * BLOCKSIZE + tIter] * sB[tIter * BLOCKSIZE + tCol];
        }
        __syncthreads();
    }

    C[tRow * N + tCol] = alpha * sum + beta * C[tRow * N + tCol];
}
```

## 2. SRAM 1D tiling
SRAM caching significantly improves performance, but it is still not on par with cuBLAS. In the current design, each thread produces one output element. Its memory access pattern looks like this:
for each element of $C$, we need approximately `K/16` DRAM loads and `K*2` shared-memory loads.

- DRAM: K/32 iterations of outer loop * 2 loads
- SRAM: K/32 iterations of outer loop * BLOCKSIZE (=32) * 2 loads
- Memory accesses per result: K / 16 DRAM, K * 2 SRAM

Profiling shows that warps often stall on memory input/output (MIO), confirming that memory traffic is the main bottleneck.

![](images/image-4.png)

We can alleviate this by reusing each loaded value more aggressively. If each thread computes 8 output elements instead of just 1 (a 1D tiling in the $M$ dimension), the access pattern becomes:

![](images/image-5.png)

Within a warp, each thread now computes 8 elements of $C$ along the column direction. Recomputing the memory accesses per result under this scheme gives us:

- DRAM: K/8 iters (dotIdx) loop * 2 loads
- SRAM: K/8 iters (dotIdx) loop * BK(=8) * (1 + TM(=8))
- Memory accesses per result: K/32 DRAM, K * 9/8 SRAM

This reduces the number of memory accesses per output element from `K/16` to `K/32` for DRAM and from `K*2` to `K*9/8` for shared memory.

```cpp
for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
{
    sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerRowB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++)
    {
        float _b = sB[dotIdx * BN + tCol];
        for (int resIdx = 0; resIdx < TM; resIdx++)
        {
            sum[resIdx] += sA[(tRow * TM + resIdx) * BK + dotIdx] * _b;
        }
    }
    __syncthreads();
}
```

In this kernel, `BM` and `TM` must be chosen so that `BM == TM * (number of results per thread in M)`. The number of threads per block is `(BM * BN / TM)`and we rely on that to match the sizes of `sA` and `sB` so that the DRAM→SRAM loads can be implemented with simple strided accesses.

```cpp 
sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerRowB];
```

## References
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
