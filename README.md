# U-shaped Split Learning의 Gradient Label Inference와 Label-Conditioned Image Reconstruction

## 연구 개요

이 프로젝트는 라벨을 서버와 직접 공유하지 않는 U-shaped Split Learning에서도
클라이언트의 h 모델이 서버의 g 모델로 반환하는 gradient에 라벨 정보가 남을 수
있는지를 분석하는 프라이버시 공격 연구입니다.

## 연구 배경: UnSplit

이 연구의 출발점은 Ege Erdoğan, Alptekin Küpçü, A. Ercüment Çiçek의
[UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label Inference Attacks against Split Learning](https://dl.acm.org/doi/abs/10.1145/3559613.3563201)
(WPES 2022, DOI: 10.1145/3559613.3563201)입니다.

UnSplit은 실제 라벨 자체를 서버에 보내지 않더라도, 역전파 과정에서 클라이언트가
서버로 반환하는 gradient가 라벨에 관한 정보를 포함할 수 있음을 보여줍니다.
논문의 label inference 공격은 단일 학습 표본에서 받은 gradient와 후보 라벨별로
계산한 gradient를 비교하여 가장 가까운 후보를 실제 라벨로 추론합니다.

논문이 보고한 100% 라벨 추론 정확도는 클라이언트가 마지막 출력층 한 층만 로컬에
보유하는 조건에 해당합니다. 클라이언트 측 모델이 두 층으로 깊어졌을 때에는 해당
공격의 성능이 무작위 추측 수준으로 감소했습니다. 따라서 이 결과를 모든 h 모델에서
라벨이 항상 완벽하게 노출된다는 뜻으로 일반화해서는 안 됩니다.

이 프로젝트는 UnSplit이 밝힌 gradient 기반 라벨 누출 가능성을 연구 동기로 삼아,
여러 층으로 구성된 h 모델이 반환하는 dL/du에도 클래스별 잔향이 남는지를 gradient
군집화로 측정합니다. 이후 추론된 라벨을 별도의 label-conditioned decoder에
제공하여 이미지의 근사 복원 품질을 높일 수 있는지 연구합니다.

연구의 최종 목표는 다음 두 단계를 연결하는 것입니다.

1. h 모델이 g 모델로 반환하는 grad_h_to_g = dL/du를 관찰하고, gradient
   군집화와 anchor 매핑을 이용하여 입력 이미지의 라벨을 추론합니다.
2. 추론한 라벨을 별도의 label-conditioned decoder에 조건으로 제공하여,
   서버가 관찰할 수 있는 중간 정보로부터 원본과 시각적으로 유사한 이미지를
   근사 복원합니다.

핵심 주제는 다음 한 문장으로 요약할 수 있습니다.

> h가 반환한 gradient로 라벨을 얻고, 별도의 label-conditioned decoder로 이미지를 복원한다.

## 전체 연구 파이프라인

    원본 이미지 x와 실제 라벨 y
            ↓
    Client f → smashed data z
            ↓
    Server g → server output u
            ↓
    Client h → logits와 loss
            ↓
    h가 g로 dL/du 반환
            ↓
    gradient 정규화·군집화
            ↓
    anchor를 이용한 라벨 추론
            ↓
    추론 라벨 + 서버 관찰 중간 정보
            ↓
    별도의 label-conditioned decoder
            ↓
    원본과 유사한 복원 이미지

이 연구에서 라벨 추론은 gradient가 클래스별로 서로 다른 방향과 분포를 형성하는지
측정합니다. 이미지 복원 단계에서는 추론 라벨이 decoder의 조건으로 작용할 때
라벨 없이 복원하는 decoder보다 형태와 클래스 특징을 더 잘 복원할 수 있는지
비교합니다.

## 핵심 연구 질문

- 실제 라벨을 직접 받지 않는 서버가 dL/du만으로 클래스 라벨을 구분할 수 있는가?
- f와 g의 cut 위치가 gradient 기반 라벨 추론 성능에 어떤 영향을 주는가?
- 추론된 라벨을 decoder에 조건으로 제공하면 이미지 복원 품질이 향상되는가?
- 실제 라벨을 사용한 decoder와 추론 라벨을 사용한 decoder 사이의 품질 차이는 얼마인가?
- 라벨 추론 오류가 최종 이미지 복원 결과에 어떤 형태로 전파되는가?

## 구현 현황

현재 저장소에는 첫 번째 단계인 gradient 기반 라벨 추론 파이프라인이 구현되어 있습니다.

- h → g gradient 수집
- gradient 평탄화와 L2 정규화
- 라벨을 사용하지 않는 K-Means 군집화
- 클래스별 anchor를 이용한 cluster-to-label 매핑
- Accuracy, Macro F1, Purity, ARI, NMI 평가
- early, middle, late cut별 비교

별도의 label-conditioned decoder와 복원 품질 평가는 연구의 두 번째 구현 단계입니다.
decoder 구현 시에는 실제 원본과 완전히 동일한 이미지를 복구한다고 주장하지 않고,
원본과 의미적·시각적으로 유사한 근사 복원을 목표로 합니다. 복원 품질은 PSNR,
SSIM, LPIPS와 분류 일치율 등을 이용해 비교할 예정입니다.

실험 실행 방법과 현재 gradient 군집화 결과는
[readme/gradient_clustering_experiment_guide.md](readme/gradient_clustering_experiment_guide.md)에서
확인할 수 있습니다.

추론 라벨, gradient, smashed data와 server output을 결합하는 조건부 이미지
복원 실험 설계는 [readme/idea1.md](readme/idea1.md)에서 확인할 수 있습니다.

---

## U자형 분할 학습 구조

시스템은 서로 분리된 세 모델로 구성됩니다.

```text
순전파:
이미지 -> ClientFront f -> z --전송--> ServerMiddle g
       -> u --전송--> ClientTail h -> logits -> CE(logits, y)

역전파:
loss -> ClientTail -> dL/du --전송--> ServerMiddle
     -> dL/dz --전송--> ClientFront
```

두 통신 경계에서는 텐서를 `detach()`한 뒤 새로운 leaf tensor로 연결합니다. 따라서 autograd 그래프가 통신 경계를 그대로 통과하지 않으며, 수신한 그래디언트를 이용해 각 모델에서 역전파를 명시적으로 다시 시작합니다.

공격에 사용하는 핵심 텐서는 다음과 같습니다.

```text
grad_h_to_g = dL/du
```

이 그래디언트는 실제 레이블이 포함된 교차 엔트로피 손실로부터 ClientTail이 생성하고, ServerMiddle의 역전파를 위해 서버에 전달합니다. 서버는 이 그래디언트를 관찰할 수 있지만 실제 레이블은 전달받지 않습니다.

`ServerMiddle.forward(z)`, 서버 transcript logger, K-means 함수에는 레이블 인자가 존재하지 않습니다. 또한 `ImageFolder` 경로에는 클래스 디렉터리 이름이 포함되므로, 공격자에게 노출되는 `sample_id`는 파일 경로가 아닌 결정론적인 불투명 해시를 사용합니다.

## 데이터셋과 앵커 이미지 준비

Dogs-vs-Cats 데이터셋을 다음과 같은 소문자 `ImageFolder` 구조로 배치합니다.

```text
workspace/data/dataset/
  train/cat/*.jpg
  train/dog/*.jpg
  val/cat/*.jpg
  val/dog/*.jpg
  test/cat/*.jpg
  test/dog/*.jpg
```

클러스터의 의미를 결정하기 위한 실제 보조 앵커 이미지는 다음 위치에 배치합니다.

```text
workspace/data/anchors/cat/cat_anchor.jpg
workspace/data/anchors/dog/dog_anchor.jpg
```

클래스 인덱스는 다음과 같이 검증됩니다.

```text
cat = 0
dog = 1
pug = 2
```

현재 프로젝트에는 공개 학술 데이터셋인 Oxford-IIIT Pet에서 선택한 실제 JPEG가 Cat 50장, Dog 50장 준비되어 있습니다. 클래스별로 train 30장, val 10장, test 10장으로 분할했으며, 이 50장과 겹치지 않는 앵커 1장도 클래스마다 별도로 사용합니다. 출처 URL과 파일별 SHA-256은 `workspace/data/dataset/public_subset_manifest.csv` 및 `workspace/data/dataset/DATASET_SOURCE.md`에서 확인할 수 있습니다.

동일한 부분집합을 다시 준비하려면 다음 명령을 실행합니다.

```bash
.\.venv\Scripts\python.exe -m src.shared.data.prepare_public_dataset
```

## 설치 및 테스트

현재 프로젝트 위치에서 실행하려면 PowerShell에 다음 두 줄을 그대로 입력합니다.

```powershell
cd C:\Class\u_shaped_gradient_label_attack
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --data ./workspace/data/dataset --epochs 1 --batch-size 1 --cut-config middle
```

프로젝트에는 `.venv` 가상환경이 준비되어 있습니다. PowerShell에서 다음과 같이 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell 실행 정책 때문에 활성화할 수 없는 경우에는 `.venv`의 Python을 직접 사용해도 됩니다.

```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --data ./workspace/data/dataset
```

환경을 새로 만드는 경우 프로젝트 디렉터리에서 다음 명령을 실행합니다.

```bash
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

기본 epoch은 `1`로 설정되어 있으므로 `--epochs`를 생략해도 1 epoch만 실행됩니다.

## 실행 명령

### 1. U자형 분할 학습 훈련

샘플 단위 그래디언트를 가장 명확하게 관찰하려면 최초 기준 실험에서 `batch-size=1`을 권장합니다.

```bash
.\.venv\Scripts\python.exe -m src.split_learning.training.train --data ./workspace/data/dataset --epochs 1 --batch-size 1 --cut-config middle
```

실제 학습 샘플의 `x -> z -> u -> logits -> dL/du -> dL/dz` 중간값을
터미널에서 자세히 보려면 다음처럼 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m src.split_learning.training.train --data ./workspace/data/dataset --epochs 1 --batch-size 1 --cut-config middle --debug-samples 1 --debug-values 10
```

- `--debug-samples`: 상세 로그를 출력할 첫 epoch의 샘플 수
- `--debug-values`: 각 텐서에서 표시할 앞부분 값의 개수
- `--debug-samples 0`: 상세 로그를 완전히 끔

상세 로그에는 shape, min/max/mean, L2 norm, 앞부분 값, logits, softmax
확률, loss, `dL/du`, `dL/dz`, f/g/h 파라미터 gradient norm이 표시됩니다.
실제 레이블이 포함된 이 화면은 연구자 디버그 view이며 서버 transcript에는
레이블이나 확률이 저장되지 않습니다.

각 값의 의미와 실제 출력 예시는
[`readme/training_debug_values_guide.md`](readme/training_debug_values_guide.md)에서
단계별로 확인할 수 있습니다.

Gradient 군집화 실행 명령, cut별 결과와 지표 해석은
[`readme/gradient_clustering_experiment_guide.md`](readme/gradient_clustering_experiment_guide.md)에서
확인할 수 있습니다.

### 2. 동일 이미지에 서로 다른 레이블을 적용하는 인과 실험

동일한 이미지와 동일한 모델 출력에 CAT과 DOG 레이블만 번갈아 적용해 `dL/du`가 달라지는지 확인합니다.

```bash
.\.venv\Scripts\python.exe -m src.experiments.same_image_different_label --image ./workspace/data/anchors/cat/cat_anchor.jpg --checkpoint ./workspace/results/checkpoints/model.pt
```

이 실험은 레이블이 그래디언트에 영향을 준다는 사실을 확인하기 위한 인과적 sanity check이며, 그 자체가 레이블 추론 공격은 아닙니다.

### 3. 서버 관찰 transcript 수집

하나의 고정된 체크포인트에서 샘플별 학습 손실 그래디언트를 수집합니다. 실제 레이블은 ClientTail만 사용하며 공격자 transcript 파일에는 저장되지 않습니다.

```bash
.\.venv\Scripts\python.exe -m src.experiments.collect_training_gradients --data ./workspace/data/dataset --split train --checkpoint ./workspace/results/checkpoints/model.pt --epoch 1
```

### 4. 레이블 없이 그래디언트 클러스터링

```bash
.\.venv\Scripts\python.exe -m src.experiments.cluster_gradients --transcripts ./workspace/results/transcripts --results ./workspace/results/reports --epoch 1 --data ./workspace/data/dataset
```

그래디언트는 평탄화한 다음 다음 식으로 L2 정규화합니다.

```text
g_hat = g / (||g||_2 + 1e-12)
```

K-means에는 실제 레이블이 전달되지 않습니다. 이 단계에서 Cluster 0과 Cluster 1은 단순한 집합 구분일 뿐 CAT 또는 DOG라는 의미를 갖지 않습니다.

### 5. 실제 Cat/Dog 앵커로 클러스터 의미 결정

체크포인트와 centroid의 epoch을 동일하게 지정해야 합니다.

```bash
.\.venv\Scripts\python.exe -m src.experiments.identify_anchor --anchor-dir ./workspace/data/anchors --checkpoint ./workspace/results/checkpoints/epoch_001.pt --centroids ./workspace/results/reports/gradient_centroids_epoch_001.npy
```

클래스당 하나의 알려진 앵커를 사용하며, 일반적인 K개 클래스에서도 일대일 대응이 유지되도록 Hungarian 알고리즘으로 클러스터와 클래스 사이의 할당을 계산합니다.

### 6. 교체 가능한 단일 Cat 이미지 실험

```bash
.\.venv\Scripts\python.exe -m src.experiments.identify_anchor --image ./my_images/real_cat.jpg --label cat --checkpoint ./workspace/results/checkpoints/epoch_001.pt --centroids ./workspace/results/reports/gradient_centroids_epoch_001.npy
```

프로그램은 이미지 파일명, 알려진 앵커 레이블, 각 클러스터와의 코사인 유사도 및 정규화 유클리드 거리, 최종 할당 클러스터를 출력합니다.

### 7. 공격 결과 평가

평가는 비지도 클러스터링과 앵커 매핑이 완료된 뒤에만 실제 레이블을 사용합니다.

```bash
.\.venv\Scripts\python.exe -m src.experiments.evaluate_attack --epoch 1 --mapping ./workspace/results/reports/cluster_mapping.json
```

다음 지표를 계산합니다.

- Clustering purity
- Adjusted Rand Index(ARI)
- Normalized Mutual Information(NMI)
- 공격 정확도
- 정밀도(precision)
- 재현율(recall)
- F1 점수
- Confusion matrix

### 8. epoch별 공격 성능 분석

```bash
.\.venv\Scripts\python.exe -m src.experiments.epoch_analysis --anchor-dir ./workspace/data/anchors --data ./workspace/data/dataset
```

각 epoch의 그래디언트는 서로 섞지 않고 독립적으로 클러스터링합니다. 서로 다른 모델 상태의 그래디언트를 섞으면 모델 변화가 교란 요인이 될 수 있기 때문입니다.

### 9. 전체 실험 실행

```bash
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --data ./workspace/data/dataset --epochs 1 --batch-size 1 --anchor-dir ./workspace/data/anchors
```

전체 실행기는 다음 단계를 순서대로 수행합니다.

```text
[1/7] U자형 분할 학습 훈련
[2/7] 서버에서 관찰 가능한 그래디언트 수집
[3/7] 레이블에 따른 그래디언트 변화 검증
[4/7] 그래디언트 벡터 정규화
[5/7] K=num_classes K-means 클러스터링
[6/7] Cat/Dog 앵커를 이용한 클러스터 매핑
[7/7] 추론 레이블 평가
```

훈련을 마친 뒤 각 epoch의 체크포인트를 다시 불러오고, 동일하게 고정된 모델 상태에서 target 및 anchor 그래디언트를 추출합니다. 이 방식은 epoch 내부의 모델 변화가 비교 결과를 교란하는 문제를 줄입니다.

## 훈련 그래디언트 공격과 순수 추론 공격의 차이

`training_gradient` 모드는 정상적인 ClientTail이 실제 레이블을 알고 손실을 계산하는 훈련 과정에서만 사용할 수 있습니다.

순수 추론 단계에서는 일반적으로 실제 레이블과 훈련 손실이 없으므로 손실 그래디언트도 존재하지 않습니다. 따라서 `inference_smashed` 모드는 존재하지 않는 그래디언트를 임의로 만들지 않고 `f(image)`의 smashed data를 특징으로 사용합니다.

```bash
.\.venv\Scripts\python.exe -m src.experiments.attack_image --attack-mode training_gradient --image ./workspace/data/anchors/cat/cat_anchor.jpg --label cat --centroids ./workspace/results/reports/gradient_centroids_epoch_001.npy
.\.venv\Scripts\python.exe -m src.experiments.cluster_smashed_data --transcripts ./workspace/results/transcripts --epoch 1
.\.venv\Scripts\python.exe -m src.experiments.attack_image --attack-mode inference_smashed --image ./unknown.jpg --centroids ./workspace/results/reports/smashed_centroids.npy
```

추론용 centroid는 smashed-data 특징으로 별도 학습해야 합니다. 그래디언트 특징과 smashed-data 특징은 차원과 의미가 다르므로 서로 혼용할 수 없습니다.

## 공격자와 평가자 데이터 분리

```text
workspace/results/transcripts/attacker_transcript/
    z, u, dL/du, dL/dz 저장
    실제 레이블 없음

workspace/results/transcripts/evaluator_ground_truth/
    불투명 sample_id와 실제 레이블 저장
    최종 평가 전용

workspace/results/reports/gradient_clusters.csv
    레이블 없이 계산한 클러스터 할당

workspace/results/reports/evaluation_clusters.csv
    평가 단계에서만 생성하는 cluster/true-label 결합 결과
```

다음 과정에서는 실제 레이블을 절대 사용하지 않습니다.

- 그래디언트 특징 생성
- 유사도 계산
- K-means 학습 및 클러스터 할당
- PCA 학습
- 공격 대상 샘플의 클러스터 결정

실제 레이블은 다음 세 경우에만 사용합니다.

1. 정상적인 ClientTail이 훈련 손실을 계산할 때
2. 공개된 보조 앵커 샘플의 의미를 지정할 때
3. 클러스터링이 끝난 뒤 평가자가 성능 지표를 계산할 때

예측 클러스터를 색으로 표시한 PCA와 실제 레이블을 색으로 표시한 evaluator-only PCA도 별도 파일로 생성됩니다.

## 주요 출력 파일

```text
workspace/results/reports/gradient_clusters.csv
workspace/results/reports/evaluation_clusters.csv
workspace/results/reports/evaluation_report.md
workspace/results/reports/evaluation_summary.json
workspace/results/reports/gradient_cosine_similarity_matrix.npy
workspace/results/reports/gradient_cosine_similarity_heatmap.png
workspace/results/reports/pca_gradient_clusters.png
workspace/results/reports/pca_gradient_ground_truth.png
workspace/results/reports/confusion_matrix.png
workspace/results/reports/attack_f1_by_epoch.csv
workspace/results/reports/attack_f1_by_epoch.png
workspace/results/reports/cluster_mapping.json
```

정확도나 공격 성능은 하드코딩하지 않으며 실제 실행 결과만 출력합니다.

## 모델 구조와 논문 대비 차이점

`early`, `middle`, `late` 설정에 따라 세 개의 convolution block을 ClientFront와 ServerMiddle 사이에 다르게 배치할 수 있습니다.

ClientTail은 다음 구조를 사용합니다.

```text
Adaptive Average Pooling
Flatten
Linear -> ReLU
Linear -> ReLU
Linear -> num_classes logits
```

## 소스 패키지 구조

```text
src/
  split_learning/   # f/g/h 모델, 통신 경계, trainer, 학습 CLI
    models/
  shared/           # 설정, 데이터 로딩, 클래스 카탈로그, 지표, 시각화
  experiments/      # 공격 알고리즘, 분석 CLI, 전체 실험 오케스트레이션
    attacks/
```

의존 방향은 `experiments -> split_learning -> shared`입니다. 데이터, 앵커,
체크포인트, transcript, 결과와 테스트는 실행 소스가 아니므로 프로젝트 루트에서 관리합니다.
Python 예약어인 `global`은 import 가능한 패키지명이 아니므로 전역 공용 책임 영역의 이름은
`shared`를 사용합니다. 모든 CLI는 프로젝트 루트에서 `python -m src...` 형태로 실행합니다.

## 다중 클래스 확장

현재 예제 데이터는 `cat`, `dog`, `pug` 세 클래스입니다. 런타임 클래스 순서와 수는
`workspace/data/dataset/train`의 하위 디렉터리에서 자동으로 발견되며, 모델 출력 수, K-Means의 `K`,
앵커 매핑, macro precision/recall/F1 및 혼동행렬에 동일하게 적용됩니다.

데이터 소스 그룹은 `workspace/data/dataset_classes.json`에서 관리합니다. 코드를 수정하지 않고 새 클래스를
추가하려면 이 파일에 알파벳순으로 클래스를 정의하고 각 split 및 아래 규칙의 앵커를 준비합니다.

```text
workspace/data/dataset/{train,val,test}/{class_name}/*.jpg
workspace/data/anchors/{class_name}/{class_name}_anchor.jpg
```

전체 실험은 클래스별 CLI 인자 대신 앵커 루트 하나를 받습니다.

```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_full_experiment --data ./workspace/data/dataset --anchor-dir ./workspace/data/anchors
```

이는 요청된 SimpleNet 계열 CNN과 세 개의 fully connected layer라는 전반적인 설명을 반영한 것입니다. 다만 논문에 명시되지 않은 정확한 채널 수, 은닉 차원, 세부 하이퍼파라미터까지 동일하다고 주장하지 않습니다.

최초 기준 실험에서는 각 epoch을 독립적으로 분석합니다. 서로 다른 epoch의 그래디언트를 결합하는 cross-epoch aggregation은 모델 상태 차이가 결과에 미치는 영향을 분리하기 위해 기본 구현에서 제외했습니다.
