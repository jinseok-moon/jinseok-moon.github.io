---
title: "Proper thread indexing and memory coalescing"
slug: "thread-indexing"
date: "2025-07-25"
draft: false
categories:
  - cuda
showToc: true
---

CUDA exposes a logical hierarchy of grid → block → thread, as shown in the figure below. Threads inside a block can be arranged in three dimensions. The key takeaway is: unless you have a very specific memory layout, **compute the linear thread index from (x, y, z) as $(x + y \cdot D_x + z \cdot D_x \cdot D_y)$**. Let’s see why.

![](images/image.png)

Does it matter if we arbitrarily swap the x, yand z dimensions when mapping to memory? The answer is **no, you cannot ignore it**. According to the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy), the linear thread ID for a 3D block is computed as follows:

![](images/image-1.png)(image-1.png)

Let’s see what happens if we ignore this in a real kernel. Suppose we have the following two kernels. Each adds two vectors of length 1024 and stores the result in `output` — this is a simple "2 loads + 1 store" pattern.

1) `kernel_0`: index computed in x–y order
2) `kernel_1`: index computed in y–x order
```cpp
__global__ void kernel_0(int* d_vec_a, int* d_vec_b, int* output, int size) {
	int index = threadIdx.x * blockDim.y + threadIdx.y;
	output[index] = d_vec_a[index] + d_vec_b[index];
}

__global__ void kernel_0(int* d_vec_a, int* d_vec_b, int* output, int size) {
	int index = threadIdx.y * blockDim.x + threadIdx.x;
	output[index] = d_vec_a[index] + d_vec_b[index];
}
```
Because each logical thread is mapped to the correct memory location, both kernels produce the same numerical result. However, their performance differs: `kernel_0` is noticeably slower.
```
kernel_0 mean time: 0.0052768 ms
kernel_1 mean time: 0.0028352 ms
```
The reason is that CUDA executes instructions at the warp level. When a warp issues a memory load, the hardware tries to fetch a contiguous 128‑byte segment from DRAM. In `kernel_0`, each thread in the warp ends up accessing data far apart in memory, so most of that 128‑byte segment is unused — this is similar to reading a row‑major matrix using column‑major indices. This uncoalesced access pattern severely hurts performance.

![](images/image-2.png)
