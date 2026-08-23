# Gradient 군집화 실험 가이드

이 문서는 U-shaped Split Learning에서 h 모델이 g 모델로 반환하는
gradient dL/du를 군집화하여 라벨을 구분하는 실험의 실행 방법과 현재 저장된
결과를 정리합니다.

## 1. 실험 흐름

```text
이미지와 실제 라벨
        ↓
f 모델: x -> z
        ↓
g 모델: z -> u
        ↓
h 모델과 CrossEntropyLoss
        ↓
grad_h_to_g = dL/du 수집
        ↓
gradient 평탄화 및 L2 정규화
        ↓
K-Means 군집화, K = 클래스 개수
        ↓
anchor gradient로 cluster 번호와 실제 클래스 매핑
        ↓
Accuracy, Macro F1, Purity, ARI, NMI 평가
```

K-Means 학습에는 실제 라벨을 전달하지 않습니다. 실제 라벨은 정상적인 h 모델의
loss 계산, anchor의 알려진 클래스 지정, 마지막 평가 단계에서만 사용합니다.

## 2. 관련 경로

| 역할 | 경로 |
|---|---|
| 학습·평가 데이터 | workspace/data/dataset |
| 클래스별 anchor | workspace/data/anchors |
| 모델 checkpoint | workspace/results/checkpoints |
| 서버 관찰 gradient | workspace/results/transcripts |
| 군집·평가 결과 | workspace/results/reports |
| 독립 실험 실행 | workspace/results/runs |

핵심 구현 파일은 다음과 같습니다.

- Gradient 반환: src/split_learning/gradient_flow/gradient_exchange.py
- Gradient 기록: src/split_learning/logging/gradient_transcript_logger.py
- Gradient 특징: src/experiments/attacks/gradient_features.py
- K-Means 군집화: src/experiments/attacks/gradient_clustering.py
- Anchor 매핑: src/experiments/attacks/anchor_mapping.py
- 공격 평가: src/experiments/evaluate_attack.py

## 3. 전체 실험 한 번에 실행

프로젝트 루트에서 다음 명령을 실행합니다. 출력 폴더는 이전 기록이 없는 새
폴더를 사용하는 것이 좋습니다.

```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --epochs 10 --batch-size 1 --cut-config middle --debug-samples 0 --output-dir ./workspace/results/runs/manual_middle_01
```

기본 입력 경로는 자동으로 다음 값이 사용됩니다.

```text
--data       workspace/data/dataset
--anchor-dir workspace/data/anchors
```

실행 결과는 다음 위치에 저장됩니다.

```text
workspace/results/runs/manual_middle_01/
├─ checkpoints/
├─ transcripts/
└─ reports/
```

## 4. 단계별 실행

### 4.1 f/g/h 모델 학습

```powershell
.\.venv\Scripts\python.exe -m src.split_learning.training.train --epochs 1 --batch-size 1 --cut-config middle --debug-samples 0 --output-dir ./workspace/results/runs/manual_step_01
```

### 4.2 고정된 checkpoint에서 gradient 수집

```powershell
.\.venv\Scripts\python.exe -m src.experiments.collect_training_gradients --checkpoint ./workspace/results/runs/manual_step_01/checkpoints/model.pt --output-dir ./workspace/results/runs/manual_step_01 --epoch 1
```

공격자가 보는 기록은 attacker_transcript에 저장되고, 실제 라벨은 분리된
evaluator_ground_truth에 저장됩니다.

### 4.3 라벨 없이 gradient 군집화

```powershell
.\.venv\Scripts\python.exe -m src.experiments.cluster_gradients --transcripts ./workspace/results/runs/manual_step_01/transcripts --results ./workspace/results/runs/manual_step_01/reports --epoch 1
```

출력 예시는 다음과 같습니다.

```text
Epoch 1: clustered 90 gradients; counts=[30, 30, 30]
```

이 출력은 epoch 1에서 gradient 90개를 읽었으며 세 cluster에 각각 30개가
배정되었다는 뜻입니다. 이 단계의 cluster 0, 1, 2는 아직 cat, dog, pug라는
의미를 갖지 않습니다.

### 4.4 Anchor로 cluster 의미 결정

```powershell
.\.venv\Scripts\python.exe -m src.experiments.identify_anchor --anchor-dir ./workspace/data/anchors --checkpoint ./workspace/results/runs/manual_step_01/checkpoints/epoch_001.pt --centroids ./workspace/results/runs/manual_step_01/reports/gradient_centroids_epoch_001.npy --mapping-output ./workspace/results/runs/manual_step_01/reports/cluster_mapping.json
```

Cluster 번호는 K-Means 실행마다 바뀔 수 있으므로 번호 자체를 클래스 이름으로
간주하면 안 됩니다. 반드시 anchor 매핑 결과를 사용해야 합니다.

### 4.5 공격 성능 평가

```powershell
.\.venv\Scripts\python.exe -m src.experiments.evaluate_attack --clusters ./workspace/results/runs/manual_step_01/reports/gradient_clusters.csv --ground-truth ./workspace/results/runs/manual_step_01/transcripts/evaluator_ground_truth/ground_truth.csv --mapping ./workspace/results/runs/manual_step_01/reports/cluster_mapping.json --results ./workspace/results/runs/manual_step_01/reports --transcripts ./workspace/results/runs/manual_step_01/transcripts --epoch 1
```

### 4.6 Epoch별 성능 확인

```powershell
.\.venv\Scripts\python.exe -m src.experiments.epoch_analysis --transcripts ./workspace/results/runs/manual_middle_01/transcripts --checkpoints ./workspace/results/runs/manual_middle_01/checkpoints --results ./workspace/results/runs/manual_middle_01/reports --anchor-dir ./workspace/data/anchors
```

## 5. Early·Middle·Late cut 비교 실행

각 실험은 transcript 누적을 방지하기 위해 서로 다른 새 출력 폴더를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --epochs 10 --batch-size 1 --cut-config early --debug-samples 0 --output-dir ./workspace/results/runs/repro_early_01
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --epochs 10 --batch-size 1 --cut-config middle --debug-samples 0 --output-dir ./workspace/results/runs/repro_middle_01
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --epochs 10 --batch-size 1 --cut-config late --debug-samples 0 --output-dir ./workspace/results/runs/repro_late_01
```

## 6. 현재 저장된 실험 결과

workspace/results/runs의 기존 cut 실험에서 epoch 10 결과는 다음과 같습니다.

| Cut | Samples | Attack Accuracy | Macro F1 | Purity | ARI | NMI |
|---|---:|---:|---:|---:|---:|---:|
| early | 90 | 82.22% | 80.86% | 0.8222 | 0.6192 | 0.7327 |
| middle | 90 | 82.22% | 80.86% | 0.8222 | 0.6192 | 0.7327 |
| late | 90 | 82.22% | 80.86% | 0.8222 | 0.6192 | 0.7327 |

세 실험 모두 90개 중 74개를 올바르게 추론하고 16개를 잘못 추론했습니다.

- early와 middle: dog 30개 중 16개를 cat으로 추론
- late: cat 30개 중 16개를 dog으로 추론
- pug: 세 cut 모두 30개를 전부 올바르게 추론

세 cut의 지표가 같다는 것은 현재 저장된 실행에서는 cut 위치에 따른 공격 성능
차이가 관찰되지 않았다는 뜻입니다. 항상 같은 결과가 나온다는 일반적인 결론은
아니며, 여러 random seed의 독립 실행으로 다시 확인해야 합니다.

## 7. 기본 reports의 210개 결과 주의사항

workspace/results/reports에는 다음 결과도 저장되어 있습니다.

| Samples | Attack Accuracy | Macro F1 | Purity | ARI | NMI |
|---:|---:|---:|---:|---:|---:|
| 210 | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |

하지만 해당 attacker transcript는 210행 중 고유한 sample_id와 epoch 조합이
90개뿐입니다. 같은 epoch의 일부 표본 기록이 이전 실행에서 누적된 상태이므로,
이 결과를 독립적인 210개 표본 실험으로 해석하면 안 됩니다.

정확한 재실험을 위해서는 비어 있는 새 output-dir을 사용하십시오. 기존
transcript 폴더에 다시 수집하면 logger가 index.csv에 행을 추가하므로 같은
표본이 중복될 수 있습니다.

## 8. 주요 지표 해석

| 지표 | 의미 |
|---|---|
| Attack Accuracy | anchor 매핑 후 추론 라벨이 실제 라벨과 일치한 비율 |
| Macro F1 | 클래스별 F1을 동일한 비중으로 평균한 값 |
| Purity | 각 cluster에서 가장 많은 실제 클래스가 차지하는 비율 |
| ARI | 우연한 군집 일치를 보정한 군집 유사도 |
| NMI | 실제 클래스와 cluster가 공유하는 정보량을 정규화한 값 |

Purity, ARI, NMI는 cluster 구조를 평가하고, Attack Accuracy와 Macro F1은
anchor 매핑 이후의 실제 라벨 추론 성능을 평가합니다.

## 9. 주요 결과 파일

```text
reports/
├─ gradient_clusters.csv
├─ gradient_centroids_epoch_XXX.npy
├─ cluster_mapping.json
├─ evaluation_summary.json
├─ evaluation_report.md
├─ confusion_matrix.png
├─ pca_gradient_clusters.png
├─ gradient_cosine_similarity_heatmap.png
├─ attack_f1_by_epoch.csv
└─ attack_f1_by_epoch.png
```

가장 먼저 확인할 파일은 evaluation_report.md이고, 프로그램에서 수치를
읽으려면 evaluation_summary.json을 사용하면 됩니다.
