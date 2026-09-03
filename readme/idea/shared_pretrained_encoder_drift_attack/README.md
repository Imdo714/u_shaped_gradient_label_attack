# Shared pretrained encoder drift attack

## 연구 목표

동일한 사전학습 Encoder를 여러 클라이언트에 배포하는 Split Learning 환경에서,
악성 클라이언트가 자신의 데이터와 정상적인 학습 transcript로 복원 공격기를
학습할 수 있는지 확인한다. 특히 피해 클라이언트의 Encoder가 개별적으로
fine-tuning되어 drift한 이후에도 cut-layer gradient가 복원 성능 저하를
완화하는지 분석한다.

## 위협 모델

악성 클라이언트는 정상 참여자로서 다음 정보를 가진다.

- 공통 초기 Encoder `E⁽⁰⁾`의 구조와 가중치
- 자신의 auxiliary 이미지와 라벨
- 자신의 `z`와 `dL/dz` transcript
- cut layer와 tensor 규격

공격자는 피해 원본, 라벨, 로컬 데이터와 현재 Encoder 가중치를 모른다. 피해
클라이언트의 `z`와 `dL/dz`는 연구 환경에서 복호화 이후의 공유 relay 또는
오케스트레이션 logging 경계에 노출된다고 가정한다. 악성 클라이언트라는 이유만으로
다른 클라이언트의 통신을 자동으로 볼 수 있다고 가정하지 않는다.

Gradient 한 번으로 서버 가중치를 복구한다고 주장하지 않는다. 반복 transcript는
서버의 국소적인 입출력 및 Jacobian 동작을 반영하지만, 정확한 서버 가중치 복구는
별도의 model extraction 문제로 구분한다.

## 공격 절차

1. Autoencoder를 사전학습하고 Encoder `E⁽⁰⁾`를 모든 클라이언트에 배포한다.
2. 악성 클라이언트는 자신의 데이터로 `(x_A, z_A, dL/dz_A)`를 수집한다.
3. `z` branch와 gradient branch를 분리한 공격 Decoder를 학습한다.
4. 각 피해 클라이언트는 로컬 데이터로 Encoder를 독립적으로 fine-tuning한다.
5. 고정된 공격 Decoder에 epoch별 피해 transcript를 입력해 원본을 복원한다.

## 비교 조건

| 조건 | 입력 | 목적 |
|---|---|---|
| Z-only | `z` | 초기 Encoder 기반 복원 기준선 |
| Gradient-only | `dL/dz` | gradient 단독 정보량 |
| Proposed | `z + dL/dz` | gradient의 drift 보정 효과 |
| Oracle | 현재 피해 Encoder에 맞춘 Decoder | 비현실적 성능 상한 |

공격기는 피해 holdout 평가 전에 고정한다. Epoch 0, 1, 5, 10, 20 및 Final에서
동일한 holdout을 평가하고, Frozen Encoder와 fine-tuned Encoder도 비교한다.

## 평가

- 복원: MSE, PSNR, SSIM, LPIPS
- Encoder drift: weight L2, representation cosine/L2, 가능하면 CKA
- 문서 이미지: OCR CER/WER와 문서 layout 복원
- 확장 실험: cut 위치, auxiliary 크기, IID/Non-IID, client 간 전이

핵심 가설은 `z + dL/dz`가 `z only`보다 drift 이후의 복원 성능 감소를 유의하게
완화한다는 것이다.
