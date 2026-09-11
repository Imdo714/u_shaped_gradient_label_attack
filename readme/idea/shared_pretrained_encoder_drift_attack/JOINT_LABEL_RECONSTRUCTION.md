# `u + dL/dz` 라벨 추론 및 이미지 복원 실험

## 설정

Autoencoder로 사전학습한 Encoder를 모든 클라이언트에 배포한다. 악성 클라이언트는
자신의 원본 이미지와 라벨로 정상 fine-tuning에 참여한 후 자신이 관찰할 수 있는
`u=g(z)`와 서버가 반환한 `dL/dz`를 수집한다.

악성 클라이언트의 transcript로 다음 두 공격기를 학습한다.

\[
\hat y_i=C_\psi(u_i,\partial L_i/\partial z_i),\qquad
\hat x_i=D_\phi(u_i,\partial L_i/\partial z_i)
\]

이후 다른 victim 클라이언트의 fine-tuning 중 같은 두 신호만 수집하여 고정된
공격기에 입력한다. victim 라벨은 gradient 생성과 평가에만 사용하며 공격기 입력에는
전달하지 않는다.

비교 조건은 `u_only`, `grad_z_only`, `u_grad_z`이고 제안 조건은 `u_grad_z`이다.

## 데이터

다섯 품종은 `beagle`, `bengal`, `pug`, `samoyed`, `siamese`이다. 데이터는 다음
경로에 준비되어 있다.

```text
workspace/data/shared_pretrained_encoder_joint_attack/animal5/
├─ pretrain/ train 250, val 50
├─ attacker/ train 200, val 50
└─ victim/   train 200, val 50, new_holdout 100
```

holdout은 클래스마다 20장씩 총 100장이다. 모든 pretrain, attacker, victim 학습
이미지와 holdout은 서로 다른 Oxford-IIIT Pet 원본이며 `dataset_manifest.csv`에
분할을 기록한다.

새 경로에 데이터를 다시 준비하려면 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.data_generation.prepare_animal5_dataset `
  --output workspace\data\shared_pretrained_encoder_joint_attack\animal5_new `
  --holdout-per-class 20 `
  --workers 8
```

## 실행

먼저 5-class 데이터로 Autoencoder를 사전학습한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.pretrain_autoencoder `
  --data workspace\data\shared_pretrained_encoder_joint_attack\animal5\pretrain `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_joint\autoencoder `
  --cut-config middle `
  --image-size 128 `
  --epochs 30 `
  --batch-size 16 `
  --device cuda
```

이어서 악성 클라이언트 fine-tuning, 공격기 학습, victim fine-tuning 및 holdout 평가를
한 번에 실행한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_joint_transcript_attack `
  --pretrained-autoencoder workspace\results\shared_pretrained_encoder_drift_attack\animal5_joint\autoencoder\pretrained_autoencoder_best.pt `
  --pretrain-data workspace\data\shared_pretrained_encoder_joint_attack\animal5\pretrain `
  --attacker-data workspace\data\shared_pretrained_encoder_joint_attack\animal5\attacker `
  --victim-data workspace\data\shared_pretrained_encoder_joint_attack\animal5\victim `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_joint\joint_attack `
  --warmup-epochs 5 `
  --attacker-finetune-epochs 10 `
  --victim-finetune-epochs 20 `
  --capture-epochs 0 1 5 10 20 `
  --attack-epochs 30 `
  --holdout-per-class 20 `
  --image-size 128 `
  --batch-size 16 `
  --attack-batch-size 8 `
  --device cuda
```

## 주요 결과

- `label_inference_summary.csv`: epoch/조건별 accuracy, balanced accuracy, macro F1
- `reconstruction_summary.csv`: epoch/조건별 MSE, MAE, PSNR, SSIM
- `victim_task_accuracy.csv`: victim의 원래 5-class 분류 정확도
- `victim/epoch_020/label_inference/u_grad_z/confusion_matrix.png`: 라벨 혼동행렬
- `victim/epoch_020/reconstruction/u_grad_z/comparison_grid.png`: 원본과 복원 비교
- `victim/epoch_020/label_predictions.csv`: 100장 각각의 정답과 추론 라벨
- `data_leakage_audit.json`: holdout 클래스 균형 및 학습 데이터 중복 검사

5-class random 라벨 추론 정확도는 20%이다. 따라서 accuracy와 macro F1이 모두
20% 수준을 충분히 넘는지 확인하고, 복원 평가는 낮은 MSE와 높은 PSNR·SSIM을 함께
확인한다.

## 라벨 추론 개선 ablation

다음 파이프라인은 아래 다섯 조건을 한 번에 학습하고 동일한 victim holdout 100장에
평가한다.

| 조건 | 학습 신호와 데이터 |
|---|---|
| `baseline_gradient_direction` | 정규화한 `dL/dz`, 공격자 최종 snapshot |
| `gradient_direction_norm` | gradient 방향과 `log10(||dL/dz||)` |
| `multi_snapshot_gradient_norm` | epoch 0/1/5/10/20 방향·크기 transcript |
| `augmented_multi_snapshot_gradient_norm` | multi-snapshot과 random crop/flip/color 증강 |
| `gated_u_gradient` | gradient logits을 기본으로 학습된 gate가 `u` logits을 선택적으로 결합 |

증강 조건은 매 attack epoch마다 새로운 random crop, 좌우 반전, 색상 변형을 적용한
이미지로 transcript를 다시 계산한다. Victim holdout은 학습, checkpoint 선택 및
unlabeled adaptation에도 사용하지 않는다.

기존 30-epoch Autoencoder checkpoint를 그대로 사용하여 다음 명령 하나만 실행한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_pretrained_encoder_drift_attack.pipeline.run_label_inference_improvements `
  --pretrained-autoencoder workspace\results\shared_pretrained_encoder_drift_attack\animal5_joint\autoencoder\pretrained_autoencoder_best.pt `
  --pretrain-data workspace\data\shared_pretrained_encoder_joint_attack\animal5\pretrain `
  --attacker-data workspace\data\shared_pretrained_encoder_joint_attack\animal5\attacker `
  --victim-data workspace\data\shared_pretrained_encoder_joint_attack\animal5\victim `
  --output workspace\results\shared_pretrained_encoder_drift_attack\animal5_label_improvements `
  --warmup-epochs 5 `
  --attacker-finetune-epochs 20 `
  --attacker-snapshot-epochs 0 1 5 10 20 `
  --victim-finetune-epochs 20 `
  --capture-epochs 0 1 5 10 20 `
  --attack-epochs 30 `
  --holdout-per-class 20 `
  --image-size 128 `
  --batch-size 16 `
  --attack-batch-size 8 `
  --device cuda
```

주요 출력은 다음과 같다.

- `label_inference_improvement_summary.csv`: victim epoch별 다섯 조건의 accuracy와 macro F1
- `attack_training_history.csv`: 공격자 validation 결과와 gated 조건의 평균 gate
- `victim/epoch_020/label_predictions.csv`: holdout 100장의 조건별 추론 결과와 확률
- `victim/epoch_020/label_inference/<조건>/confusion_matrix.png`: 조건별 혼동행렬
- `label_inference_class_metrics.csv`: epoch·조건·클래스별 precision, recall, F1
- `data_leakage_audit.json`: holdout 100장 균형과 학습 데이터 SHA-256 중복 검사
