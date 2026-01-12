---
title: "GEMM 2"
slug: "gemm-2"
date: "2025-11-08"
draft: false
categories:
  - cuda
showToc: true
---

## Arithmetic Intensity (AI)
Arithmetic intensity (AI) is defined as the ratio of operations to memory traffic, typically measured in ops/byte. A higher AI means you can perform more computation per byte of data moved. In the previous chapter we used CUDA shared memory (SRAM) and 1D tiling to improve performance, letting each thread compute multiple output elements as shown below. Let’s revisit that setup and think about how AI changes as we extend the algorithm.

![](images/image.png)


In the original kernel where each thread produced just one result, we needed 17 loads per output. With 1D tiling, that dropped to 11 loads. Moving to 2D tiling reduces it further to 9 loads. This reflects a fundamental property of GEMM: we can dramatically improve efficiency by reusing data in multiple output elements.

## 4. SRAM 2d tiling
Since 2D tiling is more effective, let’s implement it. We introduce a new parameter `TN` and extend the loops accordingly.

![](images/image-1.png)

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

![](images/image-2.png)

If we conceptually unroll the `dotIdx` loop, the access pattern looks like the figure above. In total, we only need 16 shared‑memory loads along this path.

- DRAM: K/8 iters * 2 (=A+B) * 4 (=sizeSRAM/numThreads) loads
- SRAM: K/8 iters * 8 (=dotIdx) * 2 (=A+B) * 8 (=TM,=TN) loads
- Memory accesses per result: K/64 DRAM, K/4 SRAM

## 5. Vectorized SRAM 2d tiling
On NVIDIA GPUs, the shared‑memory load instruction `LDS` can handle up to 128 bits at a time. This means we can read more data per instruction if we transpose $A$ in the 2D‑tiling kernel so that we can use vectorized loads. By transposing $A$ during the global‑to‑shared copy, we can use `LDS.128` in the same way we already do for $B$.

![](images/image-3.png)

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
