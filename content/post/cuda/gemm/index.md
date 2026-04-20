---
title: "CUDA GEMM Kernel Optimization: From Naive to Tiled SGEMM"
slug: "gemm"
date: "2026-02-23"
draft: false
categories:
  - cuda
tags:
  - cuda
  - nvidia
showToc: true
---

This post is my study notes based on the excellent [worklog](https://siboehm.com/articles/22/CUDA-MMM). I rewrote the kernels and diagrams myself while following along.
 
CUDA provides highly optimized GEMM APIs in cuBLAS, but with enough care we can write custom kernels that approach cuBLAS performance. Below, we will gradually apply core CUDA optimization ideas to a simple GEMM kernel.

- A: (M, K), row-major
- B: (K, N), row-major
- C: (M, N), row-major
- DRAM: Global memory
- SRAM: Shared memory

We implement a series of kernels and compare their performance. The results are:

0. Naive implementation, DRAM coalescing
1. SRAM caching
2. SRAM 1d tiling
3. SRAM 2d tiling
4. Vectorized SRAM 2d tiling
5. Warp tiling

```
[BENCHMARK]                    CUBLAS GEMM │ 0.045334 ms (w:10 r:20)
[BENCHMARK]               GPU GEMM 0 NAIVE │ 3.943722 ms (w:10 r:20) [PASSED]
[BENCHMARK]     GPU GEMM 0 DRAM COALESCING │ 0.517949 ms (w:10 r:20) [PASSED]
[BENCHMARK]        GPU GEMM 1 SRAM CACHING │ 0.248670 ms (w:10 r:20) [PASSED]
[BENCHMARK]      GPU GEMM 2 SRAM 1D TILING │ 0.249046 ms (w:10 r:20) [PASSED]
```

## 0. Naive implementation

![](images/1-image.png)

The most basic implementation assigns one element of $C$ to each thread. Because the matrices are stored in row-major order, this layout forces us to read non-contiguous memory for $B$. The following figure shows how this looks at the warp level.

![](images/1-image-1.png)

In this loop structure, when loading $A$ each thread in a warp accesses a different column, so the accesses are not contiguous and cannot be coalesced. Without coalescing, the hardware effectively has to service 32 separate loads per warp, which significantly degrades performance.

When loading $B$, all threads in the warp access the same element, so the hardware can use a broadcast. Even so, given the overall access pattern, there is little benefit from grouping these threads into a single warp in this naïve mapping.

### DRAM coalescing

![](images/1-image-2.png)

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

![](images/1-image-3.png)

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

![](images/1-image-4.png)

We can alleviate this by reusing each loaded value more aggressively. If each thread computes 8 output elements instead of just 1 (a 1D tiling in the $M$ dimension), the access pattern becomes:

![](images/1-image-5.png)

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

## Arithmetic Intensity (AI)
Arithmetic intensity (AI) is defined as the ratio of operations to memory traffic, typically measured in ops/byte. A higher AI means you can perform more computation per byte of data moved. In the previous chapter we used CUDA shared memory (SRAM) and 1D tiling to improve performance, letting each thread compute multiple output elements as shown below. Let's revisit that setup and think about how AI changes as we extend the algorithm.

![](images/2-image.png)

In the original kernel where each thread produced just one result, we needed 17 loads per output. With 1D tiling, that dropped to 11 loads. Moving to 2D tiling reduces it further to 9 loads. This reflects a fundamental property of GEMM: we can dramatically improve efficiency by reusing data in multiple output elements.

## 3. SRAM 2d tiling
Since 2D tiling is more effective, let's implement it. We introduce a new parameter `TN` and extend the loops accordingly.

![](images/2-image-1.png)

```cpp
  int totalResultsBlocktile = BM * BN;  // 128*128=16384
  int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);  // 16384/(8*8)=256
  int strideA = numThreadsBlocktile / BK;  // 256/8=32

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int offset = 0; offset < BM; offset += strideA) {
      A_shared[(innerRowA + offset) * BK + innerColA] =
          A[(innerRowA + offset) * K + innerColA];
    }
    for (int offset = 0; offset < BK; offset += strideB) {
      B_shared[(innerRowB + offset) * BN + innerColB] =
          B[(innerRowB + offset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
      for (int i = 0; i < TM; i++) {
        regM[i] = A_shared[(threadRow * TM + i) * BK + dotIdx];
      }
      for (int i = 0; i < TN; i++) {
        regN[i] = B_shared[dotIdx * BN + threadCol * TN + i];
      }
      for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }
```

We launch the kernel with `BM = BN = 128` and `BK = TM = TN = 8`, which yields 256 threads per block:
```cpp
template <int BM, int BN, int BK, int TM, int TN>
void launch_gpu_kernel_4(float *A, float *B, float *C, int M, int N, int K) {
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_gpu_4_sram_2d_tiling<BM, BN, BK, TM, TN>
      <<<grid, block>>>(A, B, C, M, N, K);
}
```

![](images/2-image-2.png)

If we conceptually unroll the `dotIdx` loop, the access pattern looks like the figure above. In total, we only need 16 shared‑memory loads along this path.

- DRAM: K/8 iters * 2 (=A+B) * 4 (=sizeSRAM/numThreads) loads
- SRAM: K/8 iters * 8 (=dotIdx) * 2 (=A+B) * 8 (=TM,=TN) loads
- Memory accesses per result: K/64 DRAM, K/4 SRAM

## 4. Vectorized SRAM 2d tiling
On NVIDIA GPUs, the shared‑memory load instruction `LDS` can handle up to 128 bits at a time. This means we can read more data per instruction if we transpose $A$ in the 2D‑tiling kernel so that we can use vectorized loads. By transposing $A$ during the global‑to‑shared copy, we can use `LDS.128` in the same way we already do for $B$.

![](images/2-image-3.png)

By using the `float4` vector type, the compiler generates 128‑bit load instructions, which improves performance.

```cpp
float4 tmp =
    reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
// transpose A during the GMEM to SMEM transfer
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
__syncthreads();
```

## 5. Warp Tiling

In the previous section, we went up to `Vectorized SRAM 2D Tiling`. By using a 128-bit load instruction and transposing A during the DRAM->SRAM path, we improved data access efficiency within a warp. However, thread-level tiling logic eventually hits limits, which is why many kernels adopt warp-level tiling.

![](images/3-image.png)

The thread-level 2D tiling kernel can be summarized as follows:
- shared memory caching (`sA`, `sB`)
- `float4` vectorized load/store
- transpose A when moving global->shared to improve compute-side access patterns

Block parameters are fixed as:
- `BM=64`, `BN=128`, `BK=16`
- `TM=16`, `TN=4`
- `blockDim.x = (BM*BN)/(TM*TN) = 128`

One block computes a `64x128` C tile, and one thread is responsible for a `16x4` fragment. (Since there are four warp schedulers inside an SM, it is better to use at least 128 threads.)

The thread index is directly converted into `tRow`, `tCol` to partition the full block tile.
- `int tRow = threadIdx.x / (BN / TN);`
- `int tCol = threadIdx.x % (BN / TN);`

This structure is simple, but it does not explicitly define which subtile each warp owns. As a result, it is harder to control warp-level data/compute reuse by design.

Most GEMM implementations are based on warp-level tiling, and CUTLASS provides a good reference diagram. The figure below adds a few annotations to make it easier to read.

![](images/3-image-1.png)

By partitioning the C tile at warp level and controlling each thread in a warp to compute a fixed region, we can improve compute efficiency.

If you run the actual code, you can see the following performance gap.

```bash
$ ./src/cuda/gemm/gemm 
Matrix dimensions: M=1024, N=1024, K=1024
[BENCHMARK]                              CUBLAS GEMM │ 0.045798 ms (w:10 r:20)
[BENCHMARK]         GEMM 4 VECTORIZED SRAM 2D TILING │ 0.099450 ms (w:10 r:20) [PASSED]
[BENCHMARK]                       GEMM 5 WARP TILING │ 0.082891 ms (w:10 r:20) [PASSED]
```

## Arithmetic Intensity (AI) Comparison

Let us compare thread-level and warp-level approaches.

Assumptions:
- shared block tile: `BM=64`, `BN=128`, `BK=16`
- `thread-level tiling`: `TM=16`, `TN=4`, `blockDim=128`
- `warp-level tiling`: `TM=4`, `TN=4`, `numWarps=4`, `numThreads=128`
- derived `warp-level` values: `WM=32`, `WN=64`, `WNITER=2`, `WMITER=2`
- base definitions: one FMA = 2 FLOPs, one `float` = 4 bytes

### 1) DRAM-side AI

- FLOPs: `2 * BM * BN * BK`
- DRAM bytes: `4 * (BM*BK + BK*BN)`
- formula: `AI_dram_tile = (2*BM*BN*BK) / (4*(BM*BK + BK*BN))`

Substituting values:
- FLOPs = `2*64*128*16 = 262,144`
- Bytes = `4*(64*16 + 16*128) = 12,288`
- AI = `262,144 / 12,288 = 21.333... FLOP/Byte`

| Kernel | DRAM AI (tile) |
|---|---|
| thread-level | `21.33` |
| warp-level  | `21.33` |

The values are identical because both use the same block tile and the same A/B load volume. The same result holds for full-block accounting including all C elements.

### 2) Main-loop AI (SMEM/register perspective)

The key difference comes from shared-memory usage.
We focus only on the FMA accumulation segment after loading `As/Bs -> reg` inside the `dotIdx` loop.

| Item (one `dotIdx` iteration, per thread) | thread-level  | warp-level  |
|---|---|---|
| SMEM read floats | `regM 16 + regN 4 = 20` | `regA 8 + regB 8 = 16` |
| SMEM read bytes | `80` Bytes | `64` Bytes |
| FLOPs | `64 FMA = 128 FLOPs` | `64 FMA = 128 FLOPs` |
| AI | `128 / 80 = 1.60` | `128 / 64 = 2.00` |

The same ratio is preserved when computed across the full BK span.
- `thread-level`: `2048 / 1280 = 1.60`
- `warp-level`: `2048 / 1024 = 2.00`

### 3) Conclusion

- DRAM AI is effectively the same for both kernels.
- The performance gap mostly comes from data reuse inside the main loop.
- Warp-level tiling reduces per-thread SMEM reads from `80B -> 64B`, raising AI from `1.60 -> 2.00`.

In short, warp-level tiling improves performance through finer-grained control.

## What next?
- Starting from Ampere, DRAM -> SMEM memcpy and computation can proceed asynchronously. Double-buffering with this feature can push performance further.
- This warp-tiling kernel is not a fully generalized optimized kernel; it is better viewed as tuned for a specific MNK size. In practice, cuBLAS selects from many pre-defined kernels depending on size. This kind of auto-tuning is also important.
- FP32 does not use tensor core MMA directly, but TF32 is supported. Mantissa precision drops from 23 bits to 10 bits, but throughput can be much higher.
- Another direction is to use lower-precision types (BF16/FP16/NVFP4/...) instead of FP32 to leverage tensor cores.

Next, I plan to write a post on GEMM with BF16 data type using tensor cores.

## References
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
