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

gradient를 반환하는 코드는
gradient_flow/gradient_exchange.py에만 있습니다.

- GradientExchangeResult.grad_h_to_g: h가 g로 반환하는 dL/du
- GradientExchangeResult.grad_g_to_f: g가 f로 반환하는 dL/dz
- run_gradient_exchange_step(): 한 번의 학습 및 gradient 교환
- observe_frozen_gradient_exchange(): 파라미터를 갱신하지 않는 gradient 관찰

## shared

```text
shared/
├─ configuration/    실험 설정
├─ data/             데이터셋, 이미지 로딩, 클래스 카탈로그
├─ evaluation/       군집 평가 지표와 시각화
└─ reproducibility/  난수 시드 고정
```

shared는 Split Learning 구현을 import하지 않습니다. gradient의 생성·전달·기록
책임도 갖지 않습니다.

## decoder

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

decoder는 split_learning이 만든 관측 인터페이스를 재사용하지만,
split_learning은 decoder를 import하지 않습니다.

## workspace

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

기본 경로는 shared/configuration/workspace_paths.py에서 한 번만 정의합니다.
