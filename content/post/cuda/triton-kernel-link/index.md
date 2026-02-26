---
title: "Triton kernel linking with CUDA C++"
slug: "triton-kernel-link"
date: "2025-07-05"
draft: false
categories:
  - cuda
showToc: true
---

First, we define a Triton kernel in Python using the Triton language. A Triton kernel is declared with the `@triton.jit` decorator. (The full Triton compilation pipeline will be covered in another post.)

```python
@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse, TMP, softmax_scale,
    batch, nheads, 
    ... ,
    EVEN_M, EVEN_N, EVEN_HEADDIM,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

```

Once the function is defined, we can invoke the Triton compiler.

```bash
export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)
rm -rf aot
mkdir -p aot/fp16
python ${TRITON_ROOT}/triton/tools/compile.py \
    fmha_triton.py \
    -n _fwd_kernel \
    -o aot/fp16/fmha_kernel_d64_fp16 \
    --out-name fmha_d64_fp16 \
    -w 4 \
    -ns 1 \
    -s "*fp16, *fp16, *fp16, *fp16, *fp32, *fp32, fp32, \
     i32, i32, i32, \
     i32, i32, i32, \
     i32, i32, i32, \
     i32, i32, i32, \
     i32, i32, i32, \
     i32, i32, i32, \
     i1, i1, i1, \
     1, \
     64, \
     128, \
     128" \
    -g "(seqlen_q + 127) / 128, batch * nheads, 1"
```
- -n: name of the Triton kernel defined in Python
- -o: output directory/path
- --out-name: function name that will be visible from C++
- -w: number of warps
- -ns: number of pipeline stages
- -s: function signature (parameter data types)
- -g: CUDA grid configuration; you can use runtime parameters such as `batch` and `nheads`

Finally, run `python ${TRITON_ROOT}/triton/tools/link.py aot/fp16/*.h -o aot/fmha_kernel_fp16` to link the generated pieces. This produces the following files:

```
aot
├── fmha_kernel_fp16.c
├── fmha_kernel_fp16.h
└── fp16
    ├── fmha_kernel_d64_fp16.6979ce4b_0123456789101112131415161718192021222324252627.c
    └── fmha_kernel_d64_fp16.6979ce4b_0123456789101112131415161718192021222324252627.h
```

In your actual C++ source, include these files and declare the generated kernel with `extern "C"` to avoid name mangling. The wrapper inside uses the CUDA Driver API under the hood.

```cpp
res = fmha_d64_fp16_default(stream, 
    reinterpret_cast<CUdeviceptr>(Q), reinterpret_cast<CUdeviceptr>(K), reinterpret_cast<CUdeviceptr>(V),
    reinterpret_cast<CUdeviceptr>(output), reinterpret_cast<CUdeviceptr>(LSE), reinterpret_cast<CUdeviceptr>(TMP), mscale,
    head_num * head_dim * seq_len, head_dim, head_dim*head_num,
    head_num * head_dim * seq_len, head_dim, head_dim*head_num,
    head_num * head_dim * seq_len, head_dim, head_dim*head_num,
    head_num * head_dim * seq_len, head_dim, head_dim*head_num,
    head_num, seq_len, seq_len,
    seqlen_q_rounded, head_dim, batch_size,
    even_m, even_n, even_headdim);
```
You simply need to wire up the parameters correctly and launch the kernel as shown.
