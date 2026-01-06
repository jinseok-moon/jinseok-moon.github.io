---
title: "Shared memory: bank conflicts"
slug: "bank-conflict"
date: "2024-12-02"
draft: false
categories:
  - cuda
showToc: true
---

## Bank Conflicts
When you study CUDA, you naturally end up studying shared memoryand along the way you will encounter `bank conflicts`. A bank conflict occurs when multiple memory requests from a warp are mapped to the same memory bank at the same time.

![](image.png)

![](image-1.png)

![](image-2.png)

Shared memory is composed of 32 memory modules (banks), each 32 bits wide. If we place 64 fp32 values into SRAM, they are distributed as shown below: each value is assigned sequentially across the 32 banks.

![](image-3.png)

When each thread accesses a different bank (cases 1 and 2 in the diagram), there is no bank conflict. In case 3, bank conflicts *may* occurand we need to distinguish between two sub-cases:
- Threads access the same address in the same bank: for example, threads 0–7 all access element 0 in SRAM. This is treated as a broadcast access, so no bank conflict occurs. A single memory request satisfies all 32 threads. (You can think of a memory request as operating on 128-byte units.)
- Threads access different addresses in the same bank: for example, threads 0–3 access element 0 in SRAM and threads 4–7 access element 32. In this case, the first memory request cannot also fetch the data at address 32, because bank 0 is already busy returning address 0. An additional memory request is required, which serializes the accesses and hurts performance. This situation is what we call a bank conflict.

## Experiment
To see how this behaves in practice, we can run the following four kernels. They operate on a 32×32 matrix; for simplicity the figures illustrate the 8×8 case. The data is stored in row-major order.

![](image-4.png)

### Kernel 0: ideal case
Kernel 0 is the ideal case. The 32 threads in a warp sequentially load elements 0–31 from the input in DRAM and store them into addresses 0–31 in SRAM. Then they read those values back from SRAM and write them to addresses 0–31 in the output in DRAM.


![](image-5.png)



### Kernel 1: bank conflicts case
Kernel 1 is written to intentionally create bank conflicts. It loads data from DRAM and forces all accesses to go through bank 0 in SRAM. As a result, there are 31 bank conflicts, which means the warp is effectively split into 32 serialized wavefronts.


![](image-6.png)

### Kernel 2: good case
In kernel 2, each thread accesses a different bank, but the addresses in DRAM are not contiguous. This is still fine: each memory controller is responsible for its own region of memory and can service those requests independently, so there is no performance penalty from bank conflicts. However, the indexing pattern is somewhat artificial and not always practical in real kernels — hence it is a “good case” mainly from the bank-conflict perspective.


![](image-7.png)


### Kernel 3: DRAM load overhead case
This case illustrates how important DRAM memory coalescing is. It is similar to kernel 2, but when loading from DRAM each thread fetches data from a different sector. In the coalesced case, 32 threads accessing contiguous fp32 values need only 4 sectors (4 bytes × 32 = 128 bytes). In this kernel, only 4 bytes per sector are actually useful. The hardware can still service four sectors per memory transaction, so we end up with 8 wavefronts, but overall we read 1024 bytes of DRAM to use only 128 bytes of data, which clearly degrades performance.


![](image-8.png)


## Conclusion
As these four kernels show, an essential part of CUDA programming is arranging data so that it can be processed efficiently in bulk. By carefully choosing the data layout and access patterns, you can avoid bank conflicts and prevent unnecessary performance loss.

## How about fp64?
As an aside, how does this work for fp64 data, given that the SRAM bank size is 4 bytes? Assuming aligned and coalesced memory accesses, the memory controller is designed to split a 256-byte fp64 transaction into two 128-byte requests. In this setup, bank conflicts do not occur, even though each value is 64 bits.
## References
1.  [https://forums.developer.nvidia.com/t/requesting-clarification-for-shared-memory-2024-12-02-bank-conflicts-and-shared-memory-access/268574/3](https://forums.developer.nvidia.com/t/requesting-clarification-for-shared-memory-2024-12-02-bank-conflicts-and-shared-memory-access/268574/3)
