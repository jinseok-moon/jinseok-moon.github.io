---
title: "cuda tile"
slug: "cuda-tile"
date: "2026-01-12"
categories:
  - cuda
draft: false
---

## Until now
Traditional CUDA programming has been based on CUDA C++. It requires mapping data to blocks and threads based on SIMT. While careful design can achieve optimal performance, it's not something everyone can easily do. Additionally, as GPU architectures evolve, hardware specifications change, so kernel developers had to develop different optimal kernels for each GPU.

In response to these challenges, triton-language emerged ([not related to NVIDIA Triton Inference Server](https://github.com/triton-lang/triton/issues/156)). Triton improved accessibility through block-level abstractions such as memory management, synchronization, and tensor core scheduling. Furthermore, through MLIR, it has expanded as a means to connect to various hardware backends including not only NVIDIA GPUs but also AMD and NPUs.

Of course, NVIDIA also provides libraries like CUTLASS to overcome these difficulties. By templating small tile-level operations, developers can relatively easily develop optimal kernels. FlashAttention-3 is also written based on CUTLASS. However, "relatively easy" is not truly easy. You still need to know numerous templates and logic. This also ultimately requires knowledge of CUDA C++.

## CUDA Tile IR
Then came the announcement in spring 2025. News that NVIDIA is developing a tile-based programming model. I had been waiting for it to be released, and they released it before 2025 ended.

![](images/image-1.png)

cuTile remains at the same level as the existing SIMT-based programming model. This means that in addition to the existing path of CUDA C++ code development -> PTX -> CUBIN, a new path of Tile IR -> CUBIN has been added.

![](images/image.png)

Tile IR is planned to be based on MLIR Dialect and will be stored as bytecode. Unless you're developing parts that directly interact with Tile IR, you can use cuTile Python.

- **NVIDIA cuTile Python**: Most developers fall into this category. It's a Python implementation that uses CUDA Tile IR as its backend.
- **CUDA Tile IR**: Developers who want to build their own DSL compilers or libraries use CUDA Tile IR.

## cuTile Python: how-to install
The development environment is from the official documentation.
- Linux x86_64, Linux aarch64 or Windows x86_64
- A GPU with compute capability 10.x or 12.x
- NVIDIA Driver r580 or later
- CUDA Toolkit 13.1 or later
- Python version 3.10, 3.11, 3.12 or 3.13

Once the environment is set up, install via pip.
Install torch according to your environment as well.

```bash
pip install cuda-tile
pip install cupy-cuda13x  # For sample
pip install pytest numpy  # For test
```

## Key concepts: Tile vs. SIMT

![Tile-based programming](images/image-2.png)

The tile model (left) partitions data into blocks and the compiler maps them to threads. The SIMT model (right) maps data to both blocks and threads. SIMT allows control of each individual thread, but achieving optimal performance requires manual tuning that considers hardware complexity. The tile model abstracts some of the hardware complexity, allowing the CUDA compiler/runtime to handle tile algorithms internally, while users can focus on algorithm development.

### Kernel Definition
The `@ct.kernel` decorator compiles a Python function into a GPU kernel. As we learned above, the content of this kernel doesn't generate PTX through CUDA C++, but rather goes down to `Tile IR` and generates cubin through MLIR using [CUDA TILE IR](https://github.com/NVIDIA/cuda-tile).

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    ...

class kernel(TileDispatcher):
    constant_flags = get_constant_arg_flags(function)
    compiler_options = CompilerOptions(
        num_ctas=num_ctas,
        occupancy=occupancy,
        opt_level=opt_level
    )
    compile = _compile.CompileCallback(function, compiler_options)
    super().__init__(constant_flags, compile)
    self._pyfunc = function
```

## Tutorial w/ vector addition
Let's learn how to use it through vector addition kernel, the most basic CUDA programming example.

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
 /* calculate my thread index */
 int workIndex = threadIdx.x + blockIdx.x*blockDim.x;

 if(workIndex < vectorLength)
 {
  /* perform the vector addition */
  C[workIndex] = A[workIndex] + B[workIndex];
 }
}
```

### 1D Tile

This is the simplest form of a cuTile kernel.
- Load one or more tiles from GPU memory
- Perform computation on tiles to produce new tiles
- Store result tiles to GPU memory

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
  # Get the 1D pid, blockIdx.x in cuda c++
  pid = ct.bid(0)

  # Load input tiles
  a_tile = ct.load(a, index=(pid,) , shape=(tile_size, ) )
  b_tile = ct.load(b, index=(pid,) , shape=(tile_size, ) )

  # Perform elementwise addition
  result = a_tile + b_tile

  # Store result
  ct.store(c, index=(pid, ), tile=result)
```

`ct.bid(0)` gets the block ID along axis-0. It's equivalent to getting `blockIdx` in CUDA C++. `ct.load()` loads data from memory at the required index and tile shape. Adding the loaded `a_tile` and `b_tile` is done with the `+` operator, and the result is held in the `result` tile, which hasn't been stored to DRAM yet, so we store it using `ct.store()`.

### 2D Tile
Reshaping a 1D tile to 2D makes it suitable for matrix operations. Indexing is truly intuitive as follows.

![2D element and tile space](images/image-3.png)

```python
@ct.kernel
def vec_add_kernel_2d(a, b, c, TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int]):
    # Get the global IDs of the current block along the X and Y axes.
    # `ct.bid(0)` for the first grid dimension (typically rows),
    # `ct.bid(1)` for the second grid dimension (typically columns).
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # Load `TILE_X` x `TILE_Y` chunks from input matrices 'a' and 'b'.
    # The `index=(bid_x, bid_y)` specifies the 2D tile to load.
    a_tile = ct.load(a, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))
    b_tile = ct.load(b, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))

    # Perform the element-wise addition on the loaded tiles.
    sum_tile = a_tile + b_tile

    # Store the resulting `TILE_X` x `TILE_Y` chunk back to the output matrix 'c'.
    ct.store(c, index=(bid_x, bid_y), tile=sum_tile)
```

### When tiles don't fit

The two examples above are cases where the total data size divides evenly by the tile size. For cases that don't align with tiles, use `ct.gather()` and `ct.scatter()`. This is also recommended for non-power-of-two cases. When out of bounds, `padding_value (default: 0)` is returned.

![](images/image-4.png)

```python
@ct.kernel
def vec_add_kernel_2d_gather(
    a, b, c,
    TILE_X: ConstInt, TILE_Y: ConstInt  # Tile dimensions for this block
):
    # Get the global IDs of the current block along the X and Y axes.
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # Calculate X and Y indices within the current block's tile.
    x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)
    y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)

    # Reshape the X and Y indices to (TILE_X, 1) and (1, TILE_Y), respectively.
    # This way, they can be broadcasted together to a common shape (TILE_X, TILE_Y).
    x = x[:, None]
    y = y[None, :]

    # Load elements using the calculated X and Y indices.
    # Both `a_tile` and `b_tile` have shape (TILE_X, TILE_Y).
    a_tile = ct.gather(a, (x, y))
    b_tile = ct.gather(b, (x, y))

    # Perform the element-wise addition.
    sum_tile = a_tile + b_tile

    # Store the result back to `c` using the same index tiles.
    # `ct.scatter()` only writes data to positions within the array bounds.
    ct.scatter(c, (x, y), sum_tile)
```

## Performance
Running `cutile-python/test/bench_matmul.py` gives the following results. The Torch kernel internally calls the cutlass kernel. While performance is still slower compared to torch, it's promising that tile-based programming has become possible, significantly reducing kernel implementation difficulty. Since the IR has also been released as open source, performance will gradually improve going forward.

```bash
--------------------------------------------------------------------------------------------------- benchmark 'matmul': 12 tests ---------------------------------------------------------------------------------------------------
Name (time in us)                                 Min                     Max                    Mean              StdDev                  Median                 IQR            Outliers          OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bench_matmul[1K-1K-1K-f16-torch]              14.3439 (1.0)           14.3451 (1.0)           14.3443 (1.0)        0.0005 (1.0)           14.3441 (1.0)        0.0005 (1.0)           1;0  69,714.0067 (1.0)           5        5244
bench_matmul[1K-1K-1K-f32-torch]              26.6370 (1.86)          26.6386 (1.86)          26.6374 (1.86)       0.0006 (1.37)          26.6372 (1.86)       0.0005 (1.02)          1;1  37,541.1889 (0.54)          5        3218
bench_matmul[1K-1K-1K-f16-cutile]            154.3699 (10.76)        154.4450 (10.77)        154.4040 (10.76)      0.0277 (58.67)        154.4062 (10.76)      0.0332 (73.27)         2;0   6,476.5145 (0.09)          5         613
bench_matmul[1K-1K-1K-f32-cutile]            892.8400 (62.25)        893.2511 (62.27)        893.0082 (62.26)      0.1616 (342.04)       893.0112 (62.26)      0.2286 (505.17)        2;0   1,119.8106 (0.02)          5         112
```

## References
- https://docs.nvidia.com/cuda/cutile-python/
- https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware
- https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python
