# Transcript Inversion 패키지

이 패키지는 Split Learning 서버가 관측하는 통신 transcript로 원본 이미지를
역추론하는 **Architecture-Agnostic Bidirectional Transcript Reconstruction(ABTR)**
실험을 구현합니다.

공격자는 피해자 클라이언트의 `f`, `h` 구조나 가중치를 사용하지 않습니다.
사용하는 정보는 서버가 자연스럽게 관측하거나 계산할 수 있는 다음 값입니다.

- smashed data `z`
- 서버 모델의 출력 `u = g(z)`
- 클라이언트에서 서버로 전달되는 gradient `dL/du`
- 서버가 `g`를 통해 계산해 클라이언트로 전달하는 gradient `dL/dz`
- gradient clustering 등으로 추론한 label condition

## 폴더 구성

```text
transcript_inversion/
├─ data/        transcript 입출력, 평가 원본 격리, P0-P3 pairing 통제
├─ models/      범용 f-hat, gradient-matched h-hat, 양방향 decoder
├─ losses/      gradient matching, 조건부 분포 정렬, 이미지 복원 손실
├─ training/    h-hat 준비 학습, paired 학습, strict/semi-unpaired ABTR
├─ evaluation/  holdout MSE, MAE, PSNR, SSIM 측정
└─ pipeline/    재현 가능한 P0-P5 명령행 실험
```

## 실험 조건

| 조건 | 학습 데이터 구성 | 목적 |
|---|---|---|
| P0 | 정확한 transcript–원본 pair 전체 | 복원 성능 상한선 측정 |
| P1 | 정확한 pair 일부 | pair 수에 따른 성능 변화 측정 |
| P2 | 같은 클래스 안에서 원본을 섞음 | 개별 sample pair 의존성 확인 |
| P3 | 전체 원본을 무작위로 섞음 | 클래스 평균 복원 여부 확인 |
| P4 | 피해자 pair 없이 public 이미지 사용 | strict-unpaired 핵심 공격 평가 |
| P5 | P4에 소량의 정확한 pair 추가 | semi-paired 조건 평가 |

## P0-P3 pairing 실험 실행

```powershell
python -m src.transcript_inversion.pipeline.run_pairing_suite `
  --train-manifest workspace/results/transcripts/train/manifest.csv `
  --validation-manifest workspace/results/transcripts/val/manifest.csv `
  --test-manifest workspace/results/transcripts/test/manifest.csv `
  --num-classes 2 `
  --output workspace/results/transcript_inversion/pairing
```

## P4 strict-unpaired 실험 실행

```powershell
python -m src.transcript_inversion.pipeline.run_abtr_experiment `
  --real-manifest workspace/results/transcripts/train/manifest.csv `
  --public-manifest workspace/results/transcripts/public/manifest.csv `
  --test-manifest workspace/results/transcripts/test/manifest.csv `
  --server-checkpoint workspace/results/checkpoints/model.pt `
  --num-classes 2 `
  --output workspace/results/transcript_inversion/p4
```

P5 semi-paired 실험은 P4 명령에 다음 옵션을 추가합니다.

```powershell
--paired-manifest workspace/results/transcripts/paired/manifest.csv `
--paired-fraction 0.05
```

## 공격자 데이터와 평가 데이터 분리

`TranscriptDataset(include_target=False)`는 공격자에게 보이는 transcript만
읽습니다. 피해자 원본은 학습이 끝난 후 `include_target=True`인 평가 경로에서만
사용합니다. `PublicTargetDataset`도 public 이미지와 label만 읽으며, 해당 이미지와
짝을 이루는 피해자 transcript는 열지 않습니다.

서버 체크포인트 로더는 서버가 소유한 `g`만 생성하고 `server_middle` 가중치만
불러옵니다. 피해자의 `f`, `h` 모델이나 가중치는 불러오지 않습니다.
