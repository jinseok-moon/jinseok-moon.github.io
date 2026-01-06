---
title: "cuda tile"
slug: "cuda-tile"
draft: true
---

## Until now 
Traditional CUDA programming has been built around CUDA C++. You need to map your data to blocks and threads in a SIMT modeland carefully design the kernel to get good performance. With each new GPU generation, the hardware features change, so kernel authors often have to write and tune different kernels for each architecture.

To ease this burden, the Triton language was created ([not to be confused with NVIDIA Triton Inference Server](https://github.com/triton-lang/triton/issues/156)). Triton raises the abstraction level to the block/tile level: it handles memory management, synchronizationand Tensor Core scheduling for you. On top of that, it is built on MLIR, which makes it possible to target multiple backends, including NVIDIA GPUs, AMD GPUs, NPUsand more.

NVIDIA has also provided libraries such as CUTLASS to help developers write efficient kernels. CUTLASS offers reusable templates for small tile‑level operationsand highly optimized kernels like FlashAttention‑3 are built on top of it. Still, “easier” does not mean “easy”: you must understand a large template hierarchy and a lot of implementation detailsand in the end you still need to know CUDA C++.

## CUDA Tile IR
In the spring of 2025 NVIDIA announced that they were developing a tile-based programming model. I had been waiting to see it releasedand toward the end of 2025 they finally published it.

[![](/images/cuda/cuda-tile/image-1.png)](/images/cuda/cuda-tile/image-1.png)

cuTile sits at the same conceptual level as the traditional SIMT model. In practice, this means that in addition to the classic pipeline

> CUDA C++ → PTX → CUBIN

we now have a new path

> Tile IR -> CUBIN.

[![](/images/cuda/cuda-tile/image.png)](/images/cuda/cuda-tile/image.png)

Tile IR is implemented as an MLIR dialect and is stored as bytecode. Unless you are building your own compiler or library on top of it, you will typically work with cuTile Python instead.

- **NVIDIA cuTile Python**: this is what most developers will use. It is a Python frontend that uses CUDA Tile IR as its backend.
- **CUDA Tile IR**: if you are writing your own DSL compiler or library, you can target CUDA Tile IR directly.

## cuTile Python: how to install
The official documentation lists the following environment requirements:
- Linux x86_64, Linux aarch64, or Windows x86_64
- A GPU with compute capability 10.x or 12.x
- NVIDIA Driver r580 or later
- CUDA Toolkit 13.1 or later
- Python 3.10, 3.11, 3.12, or 3.13

Once your environment is ready, install cuTile with `pip`. Install PyTorch and any other dependencies appropriate for your setup.

```bash
pip install cuda-tile

pip install cupy-cuda13x  # For samples
pip install pytest numpy  # For tests
```
