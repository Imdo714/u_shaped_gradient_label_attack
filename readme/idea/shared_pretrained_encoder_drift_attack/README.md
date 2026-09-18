# Shared pretrained encoder drift attack

`u + dL/dz`로 5-class 라벨 추론과 이미지 복원을 함께 평가하는 새 실험은
[JOINT_LABEL_RECONSTRUCTION.md](JOINT_LABEL_RECONSTRUCTION.md)를 참고한다.

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

## 실제 Online `z`-only 라벨 추론 비교

다른 `z` 기반 라벨 추론 연구와 같은 입력 조건을 만들기 위해, 서버가 실제로 수신한
`z=f(x)`만 공격 입력으로 사용하는 전용 pipeline을 제공한다. Learned `C_psi`와
Cosine Class Prototype을 모두 평가하며, `u`와 `dL/dz`는 어떤 공격기에도 전달하지
않는다. 출력은 이미지가 아니라 5개 클래스 중 하나의 예측 라벨이다.

기존 Autoencoder 결과를 삭제한 경우 먼저 동일한 Encoder checkpoint를 다시 만든다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.pretrain_autoencoder `
  --data workspace\data\shared_pretrained_encoder_joint_attack\animal5\pretrain `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_z_only_autoencoder `
  --cut-config middle `
  --image-size 128 `
  --epochs 30 `
  --batch-size 16 `
  --device cuda
```

이어서 `None/Fixed/Dynamic × Single-Latest/Multi-Matched/Multi-Natural`의 9개 조건만
실행한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_online_z_attack `
  --pretrained-autoencoder workspace\results\shared_pretrained_encoder_drift_attack\animal5_z_only_autoencoder\pretrained_autoencoder_best.pt `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_online_z_only `
  --seeds 42 43 44 45 46 `
  --augmentation-modes none fixed dynamic `
  --global-rounds 20 `
  --collection-rounds 1 5 10 20 `
  --attack-epochs 30 `
  --device cuda
```

`README.md`에는 요청한 3×3 형식의 표만 생성한다.

- Learned `C_psi`: `z`-only 라벨 추론 평균 정확도
- Cosine Prototype: `z`-only 라벨 추론 평균 정확도

상세 수치는 `aggregate_label_summary.csv`, `label_inference_summary.csv`,
`per_class_metrics.csv`에 저장한다.
`z`는 표준 Split Learning에서 서버가 직접 관찰하지만, 클라이언트 공격자가 다른
Victim의 `z`를 얻으려면 별도의 relay/logging 노출 가정이 필요하다.

## Online `z + dL/dz` 결합 라벨 추론 비교

동일한 Encoder checkpoint, 데이터 분할, seed, 증강 및 Window 조건에서 공격 입력만
`z + dL/dz`로 결합하려면 다음 명령을 실행한다. Learned `C_psi`와 Cosine Prototype을
모두 평가하며 `u`와 `dL/du`는 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_online_z_gradient_attack `
  --pretrained-autoencoder workspace\results\shared_pretrained_encoder_drift_attack\animal5_z_only_autoencoder\pretrained_autoencoder_best.pt `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_online_z_gradient `
  --seeds 42 43 44 45 46 `
  --augmentation-modes none fixed dynamic `
  --global-rounds 20 `
  --collection-rounds 1 5 10 20 `
  --attack-epochs 30 `
  --device cuda
```

완료되면 `animal5_online_z_gradient/README.md`에 두 공격기의 3×3 라벨 추론 정확도
표가 생성된다.

## 동일 Split 실행을 공유하는 최종 공정 비교

논문용 최종 비교에서는 신호별로 Split Learning을 따로 실행하지 않는다. 한 번의 동일한
Split Learning 실행에서 `z`와 `dL/dz`를 동시에 수집하고, 동일한 sample ID와 Window를
다음 세 공격 입력 view로 나누어 학습한다.

- `gradient_only`: `dL/dz`
- `u_gradient`: `u + dL/dz`
- `z_only`: `z`
- `z_gradient`: `z + dL/dz`

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_online_fair_z_comparison `
  --pretrained-autoencoder workspace\results\shared_pretrained_encoder_drift_attack\animal5_z_only_autoencoder\pretrained_autoencoder_best.pt `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_online_fair_z_comparison `
  --signal-modes gradient_only u_gradient z_only z_gradient `
  --seeds 42 43 44 45 46 `
  --augmentation-modes none fixed dynamic `
  --global-rounds 20 `
  --collection-rounds 1 5 10 20 `
  --attack-epochs 30 `
  --device cuda
```

이 실행에서 새로 얻은 `gradient_only` 결과를 공정 비교 기준값으로 사용한다. 기존
86.6% 결과는 원래 실행의 유효한 결과이지만, 당시 checkpoint가 삭제되고 `z` transcript도
저장되지 않았기 때문에 새 `z` 조건과 완전히 동일한 단일 실행 기준값으로 재사용할 수 없다.

## PCAT·SDAR·현재 공격 전체 공정 비교 스위프

이 스위프는 U-Shaped ResNet-20 한 학습 world 안에서 다음 라벨 공격을 동시에 학습한다.

- Learned `C_psi`: 공격자 client의 라벨 있는 `dL/dz` 또는 `u + dL/dz`
- Cosine Prototype: 위와 동일한 client-side transcript
- `pcat_label`: shared `g`에 맞춘 pseudo-front/pseudo-tail, 공통 100-step burn-in
- `sdar_label`: pseudo-front/pseudo-tail + representation discriminator + label flipping

PCAT·SDAR의 전체 이미지 복원 decoder는 포함하지 않는다. 이 실행은 네 계열의
**U-Shaped 라벨 추론 성능**만 비교한다. ResNet-20 split level 의미는 SDAR 공식 구현의
level 4·5·6·7과 맞췄고, SDAR U-Shaped CIFAR-10 설정인 `lambda=0.02`, label flip `0.2`를
기본값으로 사용한다. 참고한 공식 구현 revision은
`zhxchd/SDAR_SplitNN@3f0a0a8e119c9c12113b538bf3c120bd183de771`이다.

### 먼저 실행 계획만 확인

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_sweep `
  --plan-only `
  --device cuda
```

기본 설정에서는 다음 120개 조건이 `experiment_plan.csv`에 생성된다.

```text
2 datasets × 4 split levels × 3 auxiliary ratios × 5 seeds = 120 conditions
```

### 권장 Pilot

먼저 CIFAR-10의 가장 깊은 split, auxiliary 5%, seed 42 하나로 실행 경로와 GPU 시간을
확인한다. 관측량 네 개는 별도 재학습이 아니라 동일 실행의 중간 checkpoint 평가다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_sweep `
  --datasets cifar10 `
  --split-levels 7 `
  --aux-fractions 0.05 `
  --seeds 42 `
  --observation-budgets 200 1000 5000 all `
  --max-steps 20000 `
  --attack-start-step 101 `
  --batch-size 128 `
  --device cuda `
  --output workspace\results\shared_pretrained_encoder_drift_attack\pcat_sdar_label_pilot
```

### CIFAR-10 + Animal5 전체 실행

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_sweep `
  --datasets cifar10 animal5 `
  --split-levels 4 5 6 7 `
  --aux-fractions 0.01 0.05 1.0 `
  --seeds 42 43 44 45 46 `
  --observation-budgets 200 1000 5000 all `
  --max-steps 20000 `
  --attack-start-step 101 `
  --batch-size 128 `
  --eval-batch-size 128 `
  --target-learning-rate 0.001 `
  --attack-learning-rate 0.001 `
  --sdar-lambda 0.02 `
  --sdar-label-flip 0.2 `
  --signal-modes gradient_only u_gradient `
  --device cuda `
  --output workspace\results\shared_pretrained_encoder_drift_attack\pcat_sdar_fair_label_sweep
```

기본 `--resume`이 활성화되어 있다. 중단 후 같은 명령을 다시 실행하면
`COMPLETED.json`이 있는 조건은 건너뛴다. 실행 중인 한 condition 안에서 중단된 경우에는
그 condition만 처음부터 다시 시작한다. 실패 후 다음 조건까지 계속하려면
`--continue-on-error`를 추가한다.

관측량 `200`, `1000`, `5000`, `all`의 단위는 개별 이미지 수가 아니라
**Communication Step(batch)** 이다. 기본 batch 128에서는 각각 최대 25,600,
128,000, 640,000, 2,560,000 sample exposure다. 반복 epoch의 동일 이미지도 exposure에
포함되므로 고유 이미지 수와는 다르다.

Animal5의 공격자 Train은 총 200장이므로 요청 비율 1%는 2장이지만 5개 클래스를 모두
포함할 수 없다. 비교기가 클래스당 최소 1장을 선택하여 실제로는 5장, 즉 2.5%를 사용하고
`effective_aux_fraction`에 기록한다.

전체 집계는 다음 파일에 갱신된다.

- `experiment_plan.csv`: 120개 실행 조건과 완료 상태
- `all_metrics.csv`: seed별 Accuracy와 Macro-F1
- `aggregate_metrics.csv`: seed 평균, 표준편차, 95% CI
- 각 조건의 `run_config.json`: 실제 데이터 비율, 입력 권한, 적용 hyperparameter
