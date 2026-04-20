---
title: "Tensor Core로 CUDA GEMM 가속하기: WMMA API와 bf16"
slug: "gemm-tensor-core"
date: "2026-04-19"
draft: true
categories:
  - cuda
tags:
  - cuda
  - nvidia
  - tensor-core
  - gemm
showToc: true
---

CUDA 코어만으로 작성한 GEMM도 [이전 글](/ko/p/gemm/)에서처럼 cuBLAS 급까지 최적화할 수 있지만, 최신 GPU의 피크 성능을 끌어내려면 결국 Tensor Core를 써야 한다. 이 글에서는 Tensor Core에 데이터를 올리는 가장 기본적인 API인 `wmma`를 활용해, bf16 입력·fp32 accumulator 구성의 GEMM 커널을 단계별로 작성하고 벤치마크까지 비교해본다.

## 0. Baseline
이건 fp32에서 다루었던 가장 기본적인 커널이다. 다만 입력의 데이터 타입이 bf16이 되었다.
하지만 accumulator와 output은 그대로 fp32이다. 이 커널은 기존의 방식대로 돌아가기에, 쿠다코어에서 동작한다.

## 1. WMMA
텐서코어를 쓰는 방법중 하나로, `wmma` api가 있다. Warp Matrix Multiply-Accumulate의 약자로,
`mma.h` 내부의 `nvcuda::wmma` namespace를 사용한다.

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#example)를 보면 아래 코드와 같이 하나의 워프(32개 스레드)에서 동작하는 구조를 알 수 있다. 하지만 이 API만으로는 실질적으로 내부에서 어떤 스레드가 어떤 element를 담당하는지는 지정할 수 없게 되어있고, 유저는 그저 타일 레벨에서의 프로그래밍을 하게끔 되어있다.

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

참고로 bf16 사용에 있어서 가능한 타일 모양은 다음과 같다.

| Matrix A        | Matrix B        | Accumulator | Matrix Size (m-n-k) |
| --------------- | --------------- | ----------- | ------------------- |
| __nv_bfloat16   | __nv_bfloat16   | float       | 16x16x16            |
| __nv_bfloat16   | __nv_bfloat16   | float       | 32x8x16             |
| __nv_bfloat16   | __nv_bfloat16   | float       | 8x32x16             |
| precision::tf32 | precision::tf32 | float       | 16x16x8             |

이 방법을 그대로 사용해보자. 워프 1개가 16x16 output matrix를 처리할 수 있으니 4개의 워프를 사용해서 32x32 행렬을 처리하는 코드를 작성한다. K tiling 또한 16이니 루프가 필요하다.

```cuda
for (int bk = 0; bk < K; bk += 16) {
  wmma::load_matrix_sync(a_frag, A + row * K + bk, K);
  wmma::load_matrix_sync(b_frag, B + bk * N + col, N);
  wmma::mma_sync(acc, a_frag, b_frag, acc);
}

float* C_tile = C + row * N + col;
```

```bash
$ ./src/cuda/gemm/gemm_bf16 
BF16 GEMM - Matrix dimensions: M=1024, N=1024, K=1024
[BENCHMARK]      CUBLAS FP32 REF │ 0.044890 ms (w:10 r:20)
[BENCHMARK]               CUBLAS │ 0.021453 ms (w:10 r:20) [PASSED]
[BENCHMARK]           0 BASELINE │ 0.332853 ms (w:10 r:20) [PASSED]
[BENCHMARK]      1 WMMA 16x16x16 │ 0.045811 ms (w:10 r:20) [PASSED]
```