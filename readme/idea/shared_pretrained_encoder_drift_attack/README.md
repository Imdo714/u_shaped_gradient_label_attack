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

## 동물 데이터 실행

먼저 중앙 Autoencoder를 학습해 공통 초기 Encoder를 만든다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.pretrain_autoencoder `
  --data workspace\data\dataset `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal\autoencoder `
  --cut-config middle `
  --image-size 128 `
  --epochs 30 `
  --batch-size 16 `
  --device cuda
```

다음 명령은 공격기를 epoch 0에서 고정한 뒤 client 3개를 독립적으로 학습하고
Epoch 0/1/5/10/20의 복원 및 drift를 평가한다. `--oracle-epochs 5`는 Final epoch에만
현재 victim Encoder에 맞춘 비현실적 상한을 학습한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_drift_attack `
  --pretrained-autoencoder workspace\results\shared_pretrained_encoder_drift_attack\animal\autoencoder\pretrained_autoencoder_best.pt `
  --victim-data workspace\data\dataset `
  --aux-data workspace\data\dataset_aux_10k `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal\drift_attack `
  --num-clients 3 `
  --warmup-epochs 5 `
  --client-epochs 20 `
  --capture-epochs 0 1 5 10 20 `
  --attack-epochs 30 `
  --oracle-epochs 5 `
  --image-size 128 `
  --batch-size 16 `
  --attack-batch-size 8 `
  --max-holdout-samples 100 `
  --device cuda
```

주요 결과는 `summary.csv`, `drift_metrics.csv`, `reconstruction_metrics.csv`이며,
client/epoch별 공격자 transcript와 비교 이미지도 별도로 저장된다.

## UnSplit 동일 샘플 비교

`run_unsplit_comparison`은 MNIST, Fashion-MNIST, CIFAR-10을 원래 해상도로 내려받고
공식 UnSplit demo와 동일하게 test set의 클래스별 첫 이미지를 선택한다. 선택한 index는
`target_manifest.csv`로 고정되며 다음 방법이 모두 같은 victim `z`와 같은 원본으로
평가된다.

- `ours_z_only`, `ours_gradient_only`, `ours_z_gradient`: 초기 공유 Encoder에서 학습한 고정 Decoder
- `ours_oracle_z_only`: 최종 victim Encoder에 다시 맞춘 선택적 상한
- `unsplit`: 샘플별 dummy input과 clone Encoder를 교대 최적화하는 공개 UnSplit 방식

우리 방법은 auxiliary 원본을 사용하지만 UnSplit은 data-oblivious이므로 MSE뿐 아니라
공격자 지식과 샘플당 실행 시간도 함께 비교해야 한다. 논문의 보고값은 여러 split depth의
평균이고 이 pipeline의 `summary.csv`는 지정한 하나의 split 결과다. 공개 구현과 논문은
[UnSplit repository](https://github.com/ege-erdogan/unsplit)와
[논문](https://arxiv.org/abs/2108.09033)을 기준으로 했다.

먼저 MNIST에서 전체 흐름을 빠르게 확인한다. 아래 반복 횟수는 smoke용이며 논문 설정이
아니다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_unsplit_comparison `
  --dataset mnist `
  --output workspace\results\shared_pretrained_encoder_drift_attack\unsplit_comparison_quick `
  --pretrain-samples 2000 `
  --aux-samples 2000 `
  --victim-samples 2000 `
  --pretrain-epochs 2 `
  --warmup-epochs 1 `
  --victim-epochs 2 `
  --attack-epochs 3 `
  --max-targets 3 `
  --unsplit-main-iters 10 `
  --unsplit-input-iters 10 `
  --unsplit-model-iters 10 `
  --device cuda
```

다음은 논문 공개 코드의 UnSplit 반복값 `1000 × 100 × 100`과 클래스별 1장, 총 10장을
사용하는 비교 명령이다. 샘플별 교대 최적화 때문에 오래 걸린다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_unsplit_comparison `
  --dataset mnist `
  --split-depth 2 `
  --output workspace\results\shared_pretrained_encoder_drift_attack\unsplit_comparison_full `
  --oracle-epochs 20 `
  --unsplit-main-iters 1000 `
  --unsplit-input-iters 100 `
  --unsplit-model-iters 100 `
  --device cuda
```

Fashion-MNIST와 CIFAR-10은 `--dataset`과 split depth를 바꿔 실행한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_unsplit_comparison `
  --dataset fashion_mnist `
  --split-depth 2 `
  --output workspace\results\shared_pretrained_encoder_drift_attack\unsplit_comparison_full `
  --oracle-epochs 20 `
  --device cuda

.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_unsplit_comparison `
  --dataset cifar10 `
  --split-depth 4 `
  --output workspace\results\shared_pretrained_encoder_drift_attack\unsplit_comparison_full `
  --oracle-epochs 20 `
  --device cuda
```

주요 출력은 다음과 같다.

- `target_manifest.csv`: 모든 공격에 공통으로 사용한 test index와 클래스
- `summary.csv`: stage/method별 평균 MSE, MAE, PSNR, SSIM 및 UnSplit 실행 시간
- `per_sample_metrics.csv`: 동일 이미지별 상세 지표
- `victim_and_drift_metrics.csv`: 분류 정확도와 representation drift
- `final/comparison_grid.png`: Reference와 모든 공격 결과를 같은 순서로 배치한 그림
- `paper_reference.csv`: 논문에 보고된 split-depth 평균 MSE와 비교 범위 설명

첨부된 논문 그림의 정확한 index를 나중에 확인한 경우 `--target-indices`로 그대로 고정할
수 있다. 예를 들어 test index 7, 21, 42를 비교하려면 다음 옵션을 추가한다.

```powershell
--target-indices 7 21 42 --max-targets 3
```
