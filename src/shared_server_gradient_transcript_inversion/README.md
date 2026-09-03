# Shared-server gradient transcript inversion 실험 코드

이 패키지는 아이디어 문서의 두 핵심 실험을 실행한다.

1. 동일한 public auxiliary transcript로 A(`u`), B(`dL/dz`),
   C(`u+dL/dz`), D(`z+u+dL/dz`)를 학습해 신호 기여도를 비교한다.
2. 보조 클라이언트 한 곳에서 학습한 공격기를 같은 `ServerMiddle g`를
   사용하는 여러 피해 클라이언트에 **재학습 없이** 적용해 전이 행렬을 만든다.

E(`z+u+dL/du+dL/dz`)는 현재 passive downlink 위협 모델보다 강하고 기존 RPC
relay가 `dL/du`를 저장하지 않으므로 이 패키지의 주 실험에서 제외한다. F는
checkpoint를 공격기에 직접 제공하는 별도 white-box 상한이며 black-box 결과와
합치지 않는다.

## 현재 저장된 10k/holdout100으로 바로 실행

저장소에 있는 `rpc_z_u_grad_z_128_aux10k`의 public train 10,000개와 validation
75개, 별도 holdout 100개를 그대로 사용한다. 아래 실행은 victim checkpoint를
읽지 않고 transcript manifest만으로 A/B/C/D 공격기를 각각 학습한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.run_ablation_suite `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k\observations\aux_train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k\observations\aux_validation\evaluator_manifest.csv `
  --victim-manifests client_1 workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k_holdout100\observations\victim_holdout\attacker_manifest.csv workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k_holdout100\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --conditions A B C D `
  --output workspace\results\shared_server_gradient_transcript_inversion\ablation_10k `
  --image-size 128 `
  --epochs 50 `
  --batch-size 8 `
  --early-stopping-patience 8 `
  --device cuda
```

주 가설만 먼저 확인하려면 `--conditions A C`를 사용한다. 조건 C가 A보다
PSNR/SSIM은 높고 MSE/MAE는 낮은지 paired bootstrap 결과로 확인할 수 있다.

## 하나의 공유 서버에서 1:N transcript 새로 수집

`--victim-client NAME=CHECKPOINT`를 반복하면 **서버 프로세스 하나**가 보조
train/validation 및 모든 피해 클라이언트 연결을 순서대로 처리한다. A/B/C만
실험할 때는 `--capture-z`를 빼고, D도 실험할 때만 추가한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.collect_shared_server `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --aux-client-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --victim-client client_1=workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --victim-client client_2=PATH_TO_SECOND_CLIENT_ROLE_CHECKPOINT `
  --aux-data workspace\data\dataset_aux_10k `
  --victim-data workspace\data\dataset `
  --output workspace\results\shared_server_gradient_transcript_inversion\collection_1n `
  --holdout-count 100 `
  --holdout-labels cat dog `
  --image-size 128 `
  --capture-z `
  --device cuda
```

수집 후 생성되는 `manifest_index.json`의 auxiliary/victims 경로를
`run_ablation_suite`에 넘긴다. 피해 클라이언트마다 다음 옵션을 한 번씩 반복한다.

```text
--victim-manifests CLIENT_NAME ATTACKER_MANIFEST EVALUATOR_MANIFEST
```

클라이언트 간 전이를 주장하려면 `client_2` 등에 보조 클라이언트와 다른
`client_front`/`client_tail` 가중치를 가진 checkpoint를 사용해야 한다. 같은
checkpoint를 이름만 바꾸어 반복한 결과는 파이프라인 검증일 뿐 전이 증거가 아니다.

## 출력

```text
OUTPUT/
├── conditions/A|B|C|D/
│   ├── checkpoints/client_received_decoder_best.pt
│   ├── condition_config.json
│   └── clients/CLIENT/
│       ├── reconstruction_metrics.csv
│       ├── reconstruction_summary.json
│       ├── label_inference_report.json
│       └── label_confusion_matrix.csv
├── reports/transfer_matrix.csv
├── reports/CLIENT/paired_bootstrap_comparisons.json
└── run_config.json
```

`transfer_matrix.csv`는 같은 auxiliary client에서 학습한 공격기의 client별 평균
성능이다. `paired_bootstrap_comparisons.json`은 동일 transcript ID에 대해 A→C,
B→C, C→D 차이의 95% bootstrap CI와 표본별 승률을 기록한다. 라벨 head는
`dL/dz` branch만 사용하므로 A에서는 꺼지고 B/C/D에서는 gradient 기반 라벨
정확도, macro F1, 클래스별 precision/recall/F1 및 confusion matrix를 기록한다.

## 빠른 CPU 검증

전체 10k 학습 전에 기존 소형 RPC fixture로 코드 경로만 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.run_ablation_suite `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_smoke_validation\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_smoke_validation\observations\aux_train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_smoke_validation\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_smoke_validation\observations\aux_validation\evaluator_manifest.csv `
  --victim-manifests smoke workspace\results\client_received_transcript_attack\rpc_smoke_validation\observations\victim_holdout\attacker_manifest.csv workspace\results\client_received_transcript_attack\rpc_smoke_validation\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --conditions A C `
  --output workspace\results\shared_server_gradient_transcript_inversion\smoke `
  --image-size 64 `
  --signal-spatial-size 8 `
  --signal-channels 8 `
  --decoder-base-channels 16 `
  --decoder-min-channels 8 `
  --refinement-blocks 0 `
  --epochs 1 `
  --batch-size 2 `
  --ssim-weight 0 `
  --edge-weight 0 `
  --perceptual-weight 0 `
  --laplacian-weight 0 `
  --bootstrap-samples 100 `
  --max-grid-images 2 `
  --device cpu
```

스모크 실험은 실행 가능성만 검증하며 연구 결과로 사용하지 않는다.

## 합성 글자 A/B 데이터에서 조건 C 실행

동물 checkpoint는 `cat/dog/pug` 분류기이므로 글자 이미지에 그대로 사용하지 않는다.
글자용 victim `f-g-h`를 새로 학습한 뒤, 별도의 public auxiliary 글자 이미지로
`u+dL/dz` 공격기를 학습한다. 준비된 데이터 경로는 다음과 같다.

```text
workspace/data/letter_experiment/victim
  train=600, val=100, test=100, new_holdout=100
workspace/data/letter_experiment/auxiliary_10k
  train=10000, val=100
```

클래스는 `letter_a`, `letter_b`이며 모든 이미지는 128x128이다. 생성 설정과 font
hash는 `dataset_metadata.json`, 표본별 설정은 `generation_manifest.csv`에 기록된다.

데이터를 다시 생성해야 할 때만 실행한다. 기존 출력이 있으면 덮어쓰지 않고 중단한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.data_generation.prepare_letter_dataset `
  --output workspace\data\letter_experiment `
  --letters A B `
  --image-size 128 `
  --victim-train 600 `
  --victim-validation 100 `
  --victim-test 100 `
  --victim-holdout 100 `
  --auxiliary-train 10000 `
  --auxiliary-validation 100 `
  --seed 42 `
  --workers 8
```

### 1. 글자용 victim Split Learning 학습

글자 형태를 바꾸는 horizontal flip은 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.train_letter_victim `
  --data workspace\data\letter_experiment\victim `
  --output workspace\results\shared_server_gradient_transcript_inversion\letter_c\victim_model `
  --epochs 15 `
  --batch-size 32 `
  --image-size 128 `
  --cut-config middle `
  --early-stopping-patience 5 `
  --device cuda
```

### 2. 서버/클라이언트 역할 checkpoint 분리

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.rpc.export_roles `
  --checkpoint workspace\results\shared_server_gradient_transcript_inversion\letter_c\victim_model\checkpoints\model_best.pt `
  --output workspace\results\shared_server_gradient_transcript_inversion\letter_c\roles
```

### 3. 조건 C용 public 및 unseen transcript 수집

조건 C에는 `z`가 필요하지 않으므로 `--capture-z`를 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.collect_shared_server `
  --server-role-checkpoint workspace\results\shared_server_gradient_transcript_inversion\letter_c\roles\server_role.pt `
  --aux-client-checkpoint workspace\results\shared_server_gradient_transcript_inversion\letter_c\roles\client_role.pt `
  --victim-client letter_victim=workspace\results\shared_server_gradient_transcript_inversion\letter_c\roles\client_role.pt `
  --aux-data workspace\data\letter_experiment\auxiliary_10k `
  --victim-data workspace\data\letter_experiment\victim `
  --output workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts `
  --holdout-count 100 `
  --holdout-labels letter_a letter_b `
  --image-size 128 `
  --device cuda
```

### 4. 조건 C 공격 디코더 학습 및 holdout 평가

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.run_ablation_suite `
  --train-attacker-manifest workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts\observations\auxiliary_client\train\attacker_manifest.csv `
  --train-target-manifest workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts\observations\auxiliary_client\train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts\observations\auxiliary_client\validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts\observations\auxiliary_client\validation\evaluator_manifest.csv `
  --victim-manifests letter_victim workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts\observations\victim_clients\letter_victim\attacker_manifest.csv workspace\results\shared_server_gradient_transcript_inversion\letter_c\transcripts\observations\victim_clients\letter_victim\evaluator_manifest.csv `
  --class-names letter_a letter_b `
  --conditions C `
  --output workspace\results\shared_server_gradient_transcript_inversion\letter_c\condition_c `
  --image-size 128 `
  --epochs 50 `
  --batch-size 8 `
  --early-stopping-patience 8 `
  --save-separate-images `
  --device cuda
```

최종 수치는 `condition_c/conditions/C/clients/letter_victim/` 아래의
`reconstruction_summary.json`, `label_inference_report.json` 및 복원 이미지에서
확인한다. C만 실행하면 gradient 추가 기여도는 판별할 수 없으므로, 논문에서
“gradient가 복원을 개선했다”고 비교하려면 동일 데이터로 `--conditions A C`를
실행해야 한다.

## 합성 민감 문서에서 조건 C 실행

실제 개인정보 문서 대신 `bankbook_copy`와 `interview_form` 두 종류의 완전 합성
문서를 사용한다. 실존 은행·회사·사람·계좌·전화·이메일은 포함하지 않으며 모든
페이지에 연구용 가상 문서 watermark가 들어간다. 기본 해상도는 작은 글자와 문서
layout을 보존하기 위해 256x256이다.

```text
workspace/data/synthetic_document_experiment/victim
  train=600, val=100, test=100, new_holdout=100
workspace/data/synthetic_document_experiment/auxiliary_10k
  train=10000, val=100
```

데이터를 다시 생성할 때만 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.data_generation.prepare_document_dataset `
  --output workspace\data\synthetic_document_experiment `
  --image-size 256 `
  --victim-train 600 `
  --victim-validation 100 `
  --victim-test 100 `
  --victim-holdout 100 `
  --auxiliary-train 10000 `
  --auxiliary-validation 100 `
  --seed 42 `
  --workers 8
```

### 1. 합성 문서 victim 학습

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.train_synthetic_victim `
  --data workspace\data\synthetic_document_experiment\victim `
  --output workspace\results\shared_server_gradient_transcript_inversion\document_c\victim_model `
  --epochs 15 `
  --batch-size 16 `
  --image-size 256 `
  --cut-config middle `
  --early-stopping-patience 5 `
  --device cuda
```

### 2. 역할 checkpoint 분리

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.rpc.export_roles `
  --checkpoint workspace\results\shared_server_gradient_transcript_inversion\document_c\victim_model\checkpoints\model_best.pt `
  --output workspace\results\shared_server_gradient_transcript_inversion\document_c\roles
```

### 3. 조건 C transcript 수집

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.collect_shared_server `
  --server-role-checkpoint workspace\results\shared_server_gradient_transcript_inversion\document_c\roles\server_role.pt `
  --aux-client-checkpoint workspace\results\shared_server_gradient_transcript_inversion\document_c\roles\client_role.pt `
  --victim-client document_victim=workspace\results\shared_server_gradient_transcript_inversion\document_c\roles\client_role.pt `
  --aux-data workspace\data\synthetic_document_experiment\auxiliary_10k `
  --victim-data workspace\data\synthetic_document_experiment\victim `
  --output workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts `
  --holdout-count 100 `
  --holdout-labels bankbook_copy interview_form `
  --image-size 256 `
  --device cuda
```

### 4. 조건 C 디코더 학습 및 holdout 평가

```powershell
.\.venv\Scripts\python.exe -m src.shared_server_gradient_transcript_inversion.pipeline.run_ablation_suite `
  --train-attacker-manifest workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts\observations\auxiliary_client\train\attacker_manifest.csv `
  --train-target-manifest workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts\observations\auxiliary_client\train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts\observations\auxiliary_client\validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts\observations\auxiliary_client\validation\evaluator_manifest.csv `
  --victim-manifests document_victim workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts\observations\victim_clients\document_victim\attacker_manifest.csv workspace\results\shared_server_gradient_transcript_inversion\document_c\transcripts\observations\victim_clients\document_victim\evaluator_manifest.csv `
  --class-names bankbook_copy interview_form `
  --conditions C `
  --output workspace\results\shared_server_gradient_transcript_inversion\document_c\condition_c `
  --image-size 256 `
  --epochs 50 `
  --batch-size 4 `
  --early-stopping-patience 8 `
  --save-separate-images `
  --device cuda
```

이 실험의 MSE/PSNR/SSIM은 문서 layout과 픽셀 복원을 측정한다. 계좌번호나 문장
내용 자체의 복구를 주장하려면 별도의 OCR character/word error rate 평가가 필요하다.
