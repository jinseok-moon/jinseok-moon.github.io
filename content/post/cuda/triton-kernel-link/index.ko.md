---
title: "Triton kernel linking with CUDA C++"
slug: "triton-kernel-link"
date: "2025-07-05"
draft: false
categories:
    - cuda
showToc: true
---

Python에서 triton 언어를 활용해서 triton 함수를 정의해준다. triton 함수는 `@triton.jit` decorator 형태로 정의됨. Triton의 컴파일 과정은 다른 포스트에서 따로 다루는걸로..

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

함수를 정의했다면, 이제 컴파일러를 호출할 차례다.

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
- -n : python에 정의한 triton 커널 이름
- -o : output 위치
- --output-name : C++에서 사용하게 될 함수 이름
- -w : warp 개수
- -ns : stage 개수 (pipelining)
- -s : signature. 함수의 파라미터 데이터 타입
- -g : cuda grid, block 설정. batch, nheads 와 같은 파라미터로 들어오는 값을 사용할 수 있음.
- -n : python에 정의한 triton 커널 이름
- -ns : stage 개수 (pipelining)
- -s : signature. 함수의 파라미터 데이터 타입
- -g : cuda grid, block 설정. batch, nheads 와 같은 파라미터로 들어오는 값을 사용할 수 있음.

마지막으로 `python ${TRITON_ROOT}/triton/tools/link.py aot/fp16/*.h -o aot/fmha_kernel_fp16` 명령어를 실행시켜주면, 아래와 같은 파일들을 얻을 수 있음.

```
aot
├── fmha_kernel_fp16.c
├── fmha_kernel_fp16.h
└── fp16
    ├── fmha_kernel_d64_fp16.6979ce4b_0123456789101112131415161718192021222324252627.c
    └── fmha_kernel_d64_fp16.6979ce4b_0123456789101112131415161718192021222324252627.h
```

이 파일들을 실제 사용할 소스코드에서 extern “C”로 디맹글링해서 include 해주면 완성. 내부적으로 cuda driver api를 활용하도록 wrapper가 동작함.

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
이런식으로 파라미터를 맞춰넣어주고 실행시켜주면 된다.
