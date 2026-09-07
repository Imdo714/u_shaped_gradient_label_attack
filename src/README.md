# src 구조

## split_learning

```text
split_learning/
├─ f_model/          클라이언트 앞단 f: x -> z
├─ g_model/          서버 중간단 g: z -> u
├─ h_model/          클라이언트 뒷단 h: u -> logits
├─ architecture/     f/g/h 조립, 체크포인트 저장·복원, 공통 CNN 블록
├─ gradient_flow/    activation 전달과 gradient 반환
├─ logging/          서버 관찰 gradient와 평가용 정답 기록
└─ training/         optimizer와 학습·평가·CLI 조정
```

gradient를 반환하는 코드는 gradient_flow/gradient_exchange.py에만 있습니다.

- GradientExchangeResult.grad_h_to_g: h가 g로 반환하는 dL/du
- GradientExchangeResult.grad_g_to_f: g가 f로 반환하는 dL/dz
- run_gradient_exchange_step(): 한 번의 학습 및 gradient 교환
- observe_frozen_gradient_exchange(): 파라미터를 갱신하지 않는 gradient 관찰

## shared

`decoder/`는 Split Learning 구현을 import하지 않습니다. gradient의 생성·전달·기록 책임도 갖지 않습니다.

```text
shared/
├─ configuration/    실험 설정
├─ data/             데이터셋, 이미지 로딩, 클래스 카탈로그
├─ evaluation/       군집 평가 지표와 시각화
└─ reproducibility/  난수 시드 고정
```

## decoder

`decoder/`는 split_learning이 만든 관측 인터페이스를 재사용하지만, split_learning은 decoder를 import하지 않습니다.

```text
decoder/
├─ data/                 관측 수집, gradient 라벨 조건, 데이터 경계
├─ surrogate_models/     공격자 소유 f-hat, h-hat
├─ models/               신호 encoder와 label-conditioned decoder
├─ losses/               이미지 복원 손실
├─ training/             surrogate와 decoder 학습
├─ evaluation/           복원 지표와 비교 이미지
└─ pipeline/             end-to-end 복원 실험 조립
```

## client_received_transcript_attack

`client_received_transcript_attack/`는 동일 클라이언트가 서버로부터 받는 `u`와
`dL/dz`만 사용하여 공개 데이터로 공격 decoder를 학습하고 unseen holdout을
평가합니다. 공격 decoder는 Split Learning 내부 레이어 대신 관찰된 tensor shape만
사용합니다. Local provider는 고정 checkpoint에서 정확한 transcript를 만드는
기준선이고, RPC pipeline은 서버, 정상 client, passive proxy, 공격 학습기와
evaluator를 별도 process로 실행합니다. RPC 공격 학습 process에는 Victim checkpoint가
전달되지 않습니다.

```text
client_received_transcript_attack/
├─ data/          u와 dL/dz 수집, 공격자 record와 평가 원본 분리
├─ models/        bilinear baseline과 multi-scale PixelShuffle decoder
├─ training/      공개 원본 reconstruction loss를 이용한 end-to-end 학습
├─ evaluation/    holdout 지표, 고정 후처리 및 v1/v2 비교 이미지
├─ rpc/           역할 checkpoint, 안전한 TCP protocol, server/client/proxy
└─ pipeline/      local-exact 및 process-separated RPC 실험 조립
```

## client_received_transcript_refiner

`client_received_transcript_refiner/`는 고정된 공격 decoder가 만든 개별 coarse
복원 이미지를 공개 auxiliary 원본과 비교하여 image-only Residual U-Net을 학습합니다.
피해 holdout은 마지막 평가에서만 사용하며, 학습 및 검증 transcript ID와 평가 ID가
겹치면 기본적으로 실행을 중단합니다.

추가 OOF pipeline은 공개 train을 fold로 분리하여 각 샘플을 자신을 보지 않은 decoder로
복원하고, transcript-conditioned refiner는 이 coarse 영상과 raw `u`, `dL/dz`를 함께
사용합니다. 검증 기준선보다 개선되지 않은 모델은 선택하지 않습니다.

```text
client_received_transcript_refiner/
├─ data/          OOF coarse와 원래 transcript/target 결합
├─ models/        변화량이 제한된 Residual U-Net
├─ training/      image-only 및 transcript-conditioned 정제기 학습
├─ evaluation/    Original/Coarse/Refined 지표와 비교 이미지
└─ pipeline/      OOF 생성, 정제기 학습 및 unseen holdout 평가 CLI
```

## shared_server_gradient_transcript_inversion

`shared_server_gradient_transcript_inversion/`은 동일한 서버 `g`를 공유하는 1:N
환경에서 A(`u`), B(`dL/dz`), C(`u+dL/dz`), D(`z+u+dL/dz`) ablation과
클라이언트 간 공격 전이 행렬을 실행합니다. 자세한 명령은 해당 폴더의
`README.md`를 참고합니다.

## shared_pretrained_encoder_drift_attack

`shared_pretrained_encoder_drift_attack/`은 공통 Autoencoder Encoder 배포 후 발생하는
client별 representation drift와 `z + dL/dz` 복원 공격을 epoch별로 평가합니다.
MNIST, Fashion-MNIST, CIFAR-10 native 해상도에서는 동일 target manifest를 사용해
보조 데이터 기반 Decoder와 data-oblivious UnSplit baseline도 비교합니다.

## workspace

`workspace/`는 기본 경로는 shared/configuration/workspace_paths.py에서 한 번만 정의합니다.

```text
workspace/
├─ data/
│  ├─ dataset/          train, val, test 이미지와 manifest
│  ├─ anchors/          클래스별 기준 이미지
│  └─ dataset_classes.json
├─ results/
│  ├─ checkpoints/      학습된 모델
│  ├─ transcripts/      서버 관찰 기록과 평가 정답
│  ├─ reports/          표, 지표, 그림, 군집 결과
│  └─ runs/             cut 설정별 실행 결과
└─ tests/               자동화 테스트
```

## experiments

`experiments/`는 학습 자체의 핵심 구현이 아니라, 학습·공격·평가 절차를 실제로 실행하는
실험 워크플로를 모아 둔 패키지입니다. 역할에 따라 다음과 같이 구분합니다.

```text
experiments/
├─ training/          victim 모델 학습과 학습 gradient 수집
├─ analysis/          공격 성능 평가와 epoch별 변화 분석
├─ clustering/        gradient·smashed data 군집화와 label 연결
├─ inference/         학습된 산출물로 단일 이미지 공격 실행
├─ reconstruction/    decoder 학습과 holdout 이미지 복원 실험
└─ attacks/           여러 실험에서 재사용하는 공격 알고리즘
```
