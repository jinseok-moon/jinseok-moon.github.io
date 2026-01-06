---
title: "GPU architecture w/ LLM"
slug: "archs"
date: "2025-11-24"
draft: false
categories:
  - cuda
tags:
  - cuda
  - nvidia
showToc: true
---

GPU architectures have evolved over many generations and their features have changed accordingly. If you work with LLMs, it is especially useful to understand the hardware characteristics starting from Ampere.
In this post we will look at representative GPUs for each architecture—A100, H100 and Blackwell—without focusing on raw numbers like TOPS from Tensor Cores.

## A100: Ampere (SM80, 2020)
Released in 2020, A100 introduced an L1‑bypass path that optimizes copies from DRAM to shared memory (SRAM) by skipping several intermediate stages.

![](image.png)

These copies are implemented by the `cp.async` PTX instructions and are asynchronous, so we can hide their latency with a software pipeline like the one below.

![](image-1.png)

Software pipelining is a technique that removes dependency chains between successive instructions to better utilize hardware. Memory instructions are handled by the LSU (Load/Store Unit), while matrix multiplications are handled by the compute units (Tensor Cores), so there is no structural hazard between them. However, the following *data* dependency can still be a problem:

```cpp
for (i=0; i<N-1; i++) {
  load_to_register(i);
  compute(i);
}
```

We cannot execute `compute(i)` until `load_to_register(i)` has finished, because `compute(i)` needs the loaded data. To break this dependency chain, we restructure the loop into a pipeline:

```cpp
load_to_register(0);
for (i=0; i<N-2; i++) {
  load_to_register(i+1);
  compute(i);
}
compute(N-1);
```

Now the dependency between the load and compute for the *same* iteration is removed, allowing the hardware to overlap them. `FlashAttention-2` is a good example of a kernel that fully exploits this pattern; according to the authors, it approaches the theoretical peak performance on Ampere. Another heavily optimized kernel in this spirit is the mixed‑precision GEMM kernel `Marlin` (which I cover in a separate post).

## H100: Hopper (SM90, 2022)
Hopper was announced at GTC 2022 and delivers substantially higher performance than Ampere. The key architectural features we care about are TMA, WGMMA and warp specialization. Kernels such as `FlashAttention-3` and `Machete` are good examples of code that fully exploits these capabilities.

### Tensor Memory Accelerator (TMA)
On Ampere, L1‑bypass improved the performance of memory copies, but programmers still had to compute addresses and strides manually and manage synchronization barriers. To further reduce that burden, NVIDIA introduced the Tensor Memory Accelerator (TMA). With TMA you describe multi‑dimensional tensor layouts and the hardware performs bulk copies accordingly. TMA instructions are launched from a single thread, making much more efficient use of resources.

![](image-2.png)

### Warp Group Matrix Multiply-Accumulate (WGMMA)
Up through Ampere, MMA instructions were warp‑local. Hopper goes a step further: WGMMA groups 4 warps together to execute a single matrix multiply‑accumulate instruction, driving the Tensor Cores harder and improving throughput.

### Warp Specialization: Consumer-Producer
Warp specialization is a technique that makes direct use of Hopper’s ability to allocate different numbers of registers to different warps. Conceptually, our pipeline has two main stages: (1) memory and (2) computation. Memory operations need relatively few registers, while compute-heavy WGMMA operations benefit from many.

- **Producer warp group**: issues TMA memory instructions, requires few registers, focuses on pulling data in.
- **Consumer warp group**: runs WGMMA on Tensor Cores, uses many registers, focuses on computation.

## Blackwell (SM100, 2024)
Blackwell, announced at GTC 2024, is the next‑generation GPU architecture. Once again, a lot changed. Most notably, WGMMA is gone—it was effectively a Hopper‑only instruction. Instead, Blackwell introduces UMMA, with the following operand constraints:

- Operand A: TMEM or SMEM
- Operand B: SMEM
- Accumulator: TMEM

![](image-3.png)

Tensor Memory (TMEM) is a new storage structure introduced in Blackwell. Having the accumulator live in TMEM means that UMMA does not need regular registers for its accumulation. In other words, UMMA can run as a single‑thread instruction *without* consuming the usual register budget. Combined with TMA, most of the heavy lifting happens in specialized hardware; the CTA (Cooperative Thread Array, i.e., CUDA block) mainly handles setup and post‑processing.

> Historically, all of these changes follow the same trend: offloading more work to specialized hardware so that general‑purpose resources are freed for other tasks.
> - Volta: separated matrix math onto Tensor Cores, offloading it from the regular pipelines
> - Ampere: enabled true pipelining by overlapping async copies with compute
> - Hopper: used TMA and WGMMA to overlap data movement and MMA at low overhead
> - Blackwell: uses TMEM and UMMA so that even MMA can run as a single‑thread, asynchronous operation without consuming many registers

### Grace-Blackwell GB200
Grace–Blackwell combines NVIDIA’s Grace CPU with a Blackwell GPU. Everything we discussed above happens inside the GPU, but we still need to move data from host (CPU) to device (GPU). Traditionally this went over PCIe, which is relatively slow. Grace–Blackwell instead connects CPU and GPU with NVLink, which is much faster and better suited for large‑scale LLM workloads.

### Blackwell GeForce RTX 50 series (SM120, 2025)
These are the consumer GeForce GPUs based on Blackwell. Prices are high (though they are slowly coming down). Architecturally they are Blackwell, but they *do not* expose TMEM, even though they keep TMA. For LLM workloads you can still build Ampere‑style pipelines enhanced with TMA, but TMEM‑centric designs from server GPUs do not transfer directly. On the other hand, Tensor Core performance is dramatically better than previous GeForce generations.

## References
- [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining](https:/research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel)
- [CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA® Blackwell GPUs](https:/research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
