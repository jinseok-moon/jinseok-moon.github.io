---
title: "GEMM 3"
slug: "gemm-3"
date: "2026-02-23"
draft: false
categories:
  - cuda
showToc: true
---

지난 포스트에서는 `Vectorized SRAM 2D Tiling` 까지 진행했다. 128비트를 한번에 로드하는 명령어를 이용함과 동시에 DRAM->SRAM 과정에서 A를 Transpose 함으로써 warp내의 데이터 접근 효율을 높였다. 하지만 결국 thread 단위의 tiling 로직에는 한계가 존재하는데, 이러한 연유로 많은 커널들이 warp level tiling을 하게 된다. 이번 포스트의 주된 확장은 thread-level to warp-level tiling 이다.

![](images/image.png)

thread-level 2d tiling 커널의 로직을 정리하면 다음과 같다.
- shared memory 캐싱 (`sA`, `sB`)
- `float4` Vectorized load/store
- A를 global->shared 이동 시 transpose하여 계산 접근 패턴 개선

블록 파라미터는 고정값으로:
- `BM=64`, `BN=128`, `BK=16`
- `TM=16`, `TN=4`
- `blockDim.x = (BM*BN)/(TM*TN) = 128`

블록 하나가 `64x128` C 타일을 계산하고 thread 하나는 `16x4` 조각을 담당한다. (SM 내부에는 warp scheduler가 4개가 존재하기 때문에 thread는 128개 이상으로 잡는게 좋다.) 

Thread index를 바로 `tRow`, `tCol`로 변환해 블록 전체 타일을 나눈다.
- `int tRow = threadIdx.x / (BN / TN);`
- `int tCol = threadIdx.x % (BN / TN);`

이 구조는 동작은 단순하지만, warp가 어떤 서브타일을 책임지는지 명시적이지 않다. 결과적으로 warp 단위 데이터/계산 재사용을 설계적으로 통제하기 어렵다.

## 5. Warp Tiling

대부분의 GEMM 연산은 warp-level tiling을 기반으로 하며, 이에 대한 그림을 CUTLASS에서 제공한다. 이를 좀 더 알기 쉽게 주석을 조금 추가한 게 다음 그림이다.

![](images/image-1.png)

C 타일을 warp-level로 나누고 warp 내부에서 각 스레드는 정해진 구역을 계산하도록 통제함으로써 연산 효율을 올리는 것이다.

실제 코드를 돌려보면 다음과 같은 성능 차이가 난다.

```bash
$ ./src/cuda/gemm/gemm 
Matrix dimensions: M=1024, N=1024, K=1024
[BENCHMARK]                              CUBLAS GEMM │ 0.045798 ms (w:10 r:20)
[BENCHMARK]         GEMM 4 VECTORIZED SRAM 2D TILING │ 0.099450 ms (w:10 r:20) [PASSED]
[BENCHMARK]                       GEMM 5 WARP TILING │ 0.082891 ms (w:10 r:20) [PASSED]
```

## Arithmetic Intensity (AI) 비교

thread-level과 warp-level을 비교해보자.

비교 전제:
- 공통 블록 타일: `BM=64`, `BN=128`, `BK=16`
- `thread-level tiling`: `TM=16`, `TN=4`, `blockDim=128`
- `warp-level tiling`: `TM=4`, `TN=4`, `numWarps=4`, `numThreads=128`
- `warp-level` 유도값: `WM=32`, `WN=64`, `WNITER=2`, `WMITER=2`
- 기본 정의: FMA 1개 = 2 FLOPs, `float` 1개 = 4 Bytes


### 1) DRAM 기준 AI

- FLOPs: `2 * BM * BN * BK`
- DRAM bytes: `4 * (BM*BK + BK*BN)`
- 식: `AI_dram_tile = (2*BM*BN*BK) / (4*(BM*BK + BK*BN))`

값 대입:
- FLOPs = `2*64*128*16 = 262,144`
- Bytes = `4*(64*16 + 16*128) = 12,288`
- AI = `262,144 / 12,288 = 21.333... FLOP/Byte`

| 커널 | DRAM AI (tile) |
|---|---|
| thread-level | `21.33` |
| warp-level  | `21.33` |

동일한 블록 타일과 A/B 로딩량을 쓰기 때문에 값이 같다. 이는 전체 C를 포함해서, 블록 전체에서 곘나해도 동일한 값이 나온다.

### 2) SMEM/레지스터 메인루프 기준 AI
차이가 나는 부분은 SMEM 활용부분이다.
`dotIdx` 루프에서 `As/Bs -> reg` 로드 후 FMA 누적 구간만 본다.

| 항목 (`dotIdx` 1회, thread당) | thread-level  | warp-level  |
|---|---|---|
| SMEM read floats | `regM 16 + regN 4 = 20` | `regA 8 + regB 8 = 16` |
| SMEM read bytes | `80` Bytes | `64` Bytes |
| FLOPs | `64 FMA = 128 FLOPs` | `64 FMA = 128 FLOPs` |
| AI | `128 / 80 = 1.60` | `128 / 64 = 2.00` |

BK 전체로 계산해도 동일 비율이 유지된다.
- `thread-level`: `2048 / 1280 = 1.60`
- `warp-level`: `2048 / 1024 = 2.00`

### 3) 결론

- DRAM AI는 두 커널이 사실상 같다.
- 성능 차이는 주로 메인루프 내부 데이터 재사용에서 발생한다.
- warp-level tiling은 thread당 SMEM read를 `80B -> 64B`로 줄여 AI를 `1.60 -> 2.00`으로 높인다.

즉, warp-level tiling에서는 보다 세밀한 제어를 통해서 성능을 올리는 것이다.

## 그 다음은?
- Ampere 부터는 DRAM -> SMEM memcpy / computation 을 asynchronous 하게 진행할 수 있다. 이를 이용한 double-buffering을 통해서 보다 성능을 높일 수 있다. 
- 이번 warp-tiling은 일반화된 최적화커널이 아닌 특정 MNK 사이즈에 대해서 최적화된 커널로 볼 수 있다. 실제로 cuBLAS는 사이즈에 따라 미리 정의해둔 수많은 커널들중에서 최적의 커널을 사용한다. 이러한 auto tuning 또한 중요하다.
- FP32 는 텐서코어 MMA 연산을 지원하지 않는 대신 TF32를 지원한다. Mantissa가 23 bits -> 10 bits 로 줄어들지만, 연산 속도는 훨씬 빨라질 수 있다.
- FP32가 아닌 보다 낮은 정밀도의 데이터타입 (BF16/FP16/NVFP4/...) 을 활용하여 tensor core를 활용하는 방향이 있다.

다음은 BF16 데이터 타입의 GEMM에 대해서, Tensor core를 활용하는 글을 작성할 예정이다.