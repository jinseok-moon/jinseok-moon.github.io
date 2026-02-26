---
title: "Pageable vs. pinned data transfer"
slug: "pinned-memory"
date: "2025-06-19"
categories:
  - cuda
tags:
  - cuda
  - nvidia
draft: false

---

In CUDA, one common way to copy memory from the host to the device is via the `cudaMemcpy` API. By default, memory you allocate on the host without any special handling is *pageable* memory. To copy data from pageable host memory to the device, the driver first has to move it into an internal pinned buffer, which introduces extra overhead and slows the transfer down.

If you explicitly allocate host memory with `cudaMallocHost`, you get pinned (page-locked) memory directly. In that case, the intermediate copy step is skippedand transfers can proceed faster. For example, when copying 1 GB of memory to the device, the performance difference between the two approaches looks like this:

```bash
$ ./pinned_memory 
Pinned memory
Total time: 185.875 ms
Average time per copy: 18.5875 ms
Data size: 1 GB
Bandwidth: 53.7997 GB/s

Pageable memory
Total time: 367.491 ms
Average time per copy: 36.7491 ms
Data size: 1 GB
Bandwidth: 27.2116 GB/s
```

We can see that using pinned memory significantly improves throughput. However, pinned memory consumes physical system RAM and cannot be paged out, so it should be used selectively and only where it makes sense.

## References
- https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
