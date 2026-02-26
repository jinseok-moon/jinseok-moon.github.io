---
title: "GEMM 3"
slug: "gemm-3"
date: "2026-02-23"
draft: false
categories:
  - cuda
showToc: true
---

In the previous post, we went up to `Vectorized SRAM 2D Tiling`. By using a 128-bit load instruction and transposing A during the DRAM->SRAM path, we improved data access efficiency within a warp. However, thread-level tiling logic eventually hits limits, which is why many kernels adopt warp-level tiling. The main extension in this post is moving from thread-level to warp-level tiling.

![](images/image.png)

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

## 5. Warp Tiling

Most GEMM implementations are based on warp-level tiling, and CUTLASS provides a good reference diagram. The figure below adds a few annotations to make it easier to read.

![](images/image-1.png)

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
