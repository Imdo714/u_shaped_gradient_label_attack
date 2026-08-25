# U-shaped Split Learning의 Gradient Label Inference와 Label-Conditioned Image Reconstruction

## 연구 배경과 필요성

AI 기술은 의료, 금융, 비전 기반 서비스, LLM 기반 챗봇 등 다양한 분야에서 빠르게 활용되고 있다. 최근에는 사용자의 특성, 환경 및 행동 패턴을 반영하여 개별 사용자에게 최적화된 개인화 모델에 대한 요구가 증가하고 있다.

그러나 이러한 개인화 서비스를 제공하기 위해서는 사용자의 의료 기록, 금융 정보, 이미지, 대화 기록 등과 같은 민감한 개인 데이터를 학습에 활용해야 하므로, 데이터가 중앙 서버로 수집되는 과정에서 개인정보 및 프라이버시가 노출될 가능성 또한 증가하고
있다. 중앙집중형 학습은 데이터 관리와 연산을 단순화할 수 있지만, 원본 데이터가 한 곳에 모인다는 점에서 유출 사고의 영향 범위가 커지고 데이터 소유자가 자신의 정보 흐름을 통제하기 어려워질 수 있다.

이러한 문제를 해결하기 위해, 원본 데이터를 서버에 직접 전달하지 않고도 모델을 학습할 수 있는 `Federated Learning(FL)`과 `Split Learning(SL)`과 같은 프라이버시 보존형 분산 학습 기술이 활발하게 연구되고 있다.

`Federated Learning`은 각 클라이언트가 로컬 데이터로 모델을 학습하고 모델 업데이트를 중앙 서버에서 집계한다. 반면 Split Learning은 하나의 신경망을 여러 구간으로 나누고, 클라이언트와 서버가 순전파와 역전파를 이어서 수행한다. 두 방법 모두 원본 데이터를 중앙 서버에 직접 모으지 않는다는 공통점이 있지만, 서버에 전달되는 정보의 형태와 학습 프로토콜은 서로 다르다.

특히 `Split Learning`은 전체 신경망을 클라이언트와 서버로 분할하고, 클라이언트의 원본 데이터 자체를 서버에 전달하는 대신 중간 계층에서 생성되는 `Smashed Data`를 서버와 주고받는 방식으로 학습을 수행한다. 모델을 나누는 지점을 `cut layer`라고 하며, 클라이언트는 `cut layer`까지 계산한 `activation`을 서버에 전달한다. 서버는 나머지 계층의 순전파와 `loss` 계산을 수행하고, 역전파 과정에서 `cut layer`에 대한 `gradient`를 클라이언트로 돌려준다.

```text
Client                         Server
원본 데이터 x
    ↓
앞단 모델 f
    ↓
smashed data z  ─────────────→  뒷단 모델 g → 예측 → loss
                 ←────────────  dL/dz
```

이를 통해 서버가 사용자의 원본 데이터에 직접 접근하는 것을 제한하여 기존
중앙집중형 학습 방식에서 발생할 수 있는 개인정보 노출 위험을 완화하고자 한다. 다만 원본 데이터 대신 `activation`과 `gradient`를 공유한다는 사실이 그 자체로 완전한 프라이버시를 보장한다는 뜻은 아니다.

## U-shaped Split Learning의 등장 배경

일반적인 `Split Learning`의 대표적인 구성에서는 모델의 마지막 계층이 서버에 위치하기 때문에 학습 과정에서 사용자의 라벨 정보가 서버에 노출될 수 있다. 이러한 문제를 해결하기 위해 모델의 초기 계층뿐만 아니라 마지막 계층까지 클라이언트에 배치하는 `U-shaped Split Learning` 구조가 제안되었으며, 이를 통해 원본 데이터뿐만 아니라 라벨 정보까지 서버로부터 보호하고자 한다.

<img width="1076" height="552" alt="Image" src="https://github.com/user-attachments/assets/7ed7d095-a640-48c2-8f0a-c2d2f1c5de5e" />

```text
Client                         Server                         Client
원본 x → 앞단 f → z  ───────→  중간 모델 g → u  ───────────→  뒷단 h
                    ←───────  dL/dz       ←───────────────  dL/du ← loss(y)
```

이 구조에서 서버는 원본 이미지 `x`와 실제 라벨 `y`를 직접 받지 않는다. 서버가 보유한 중간 모델은 `z`를 입력받아 `u`를 생성하고, 학습을 계속하기 위해 클라이언트의 뒷단 모델로부터 `dL/du`를 전달받는다. 따라서 `U-shaped Split Learning` 구조는 입력과 라벨을 모두 클라이언트 측에 남길 수 있다는 장점이 있다.

그러나 `U-shaped Split Learning`에는 다음과 같은 한계가 남는다.

- 매 `mini-batch`마다 `activation`과 `gradient`를 교환하므로 통신 비용과 지연이 발생한다.
- 클라이언트와 서버의 순차적인 연산 의존성 때문에 느린 참여자가 전체 학습 속도에 영향을 줄 수 있다.
- 서버가 원본과 라벨을 직접 받지 않더라도 `smashed data`, `server output`, `gradient`에는 입력이나 라벨과 통계적으로 연관된 정보가 남을 수 있다.
- 따라서 “원본 데이터와 라벨을 전송하지 않는다”는 프로토콜 수준의 보호와 “관찰된 중간 정보로부터 이를 추론할 수 없다”는 정보 수준의 보호는 구분해야 한다.

## U-shaped Split Learning에 남아 있는 프라이버시 위험

하지만 `U-shaped Split Learning`에서도 학습 과정에서 서버와 클라이언트 사이에 `smashed data`와 `gradient`가 지속적으로 교환되므로, 이러한 중간 정보에 라벨 및 원본 데이터와 관련된 정보가 잔존할 가능성이 있다. 따라서 `U-shaped Split Learning` 구조가 라벨을 서버에 직접 공개하지 않더라도, 서버가 관찰 가능한 `gradient`를 통해 라벨을 간접적으로 추론할 수 있는지, 그리고 추론된 라벨 정보가 원본 이미지 복원 공격의 성능을 얼마나 향상시키는지에 대한 분석이 필요하다.

이 연구의 출발점은 Ege Erdoğan, Alptekin Küpçü, A. Ercüment Çiçek의 [UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label Inference Attacks against Split Learning](https://dl.acm.org/doi/abs/10.1145/3559613.3563201) (WPES 2022, DOI: 10.1145/3559613.3563201)이다. `UnSplit`은 실제 라벨을 서버에 전달하지 않더라도 역전파 `gradient`가 라벨과 원본 데이터에 관한 정보를 포함할 수 있음을 보여준다.

다만 `UnSplit`이 보고한 높은 라벨 추론 성능을 모든 모델 구조에 그대로 일반화할 수는 없다. 공격 성능은 클라이언트 뒷단 모델의 깊이, `cut` 위치, 데이터와 클래스 수, 학습 상태 등에 따라 달라질 수 있다. 본 연구는 이러한 조건을 명시적으로 통제하고, 여러 층으로 구성된 클라이언트 뒷단 모델에서도 서버로 반환되는 `dL/du`에 클래스별 구조가 남는지를 실험적으로 측정한다.

## 위협 모델

본 연구는 프로토콜을 정상적으로 수행하지만 자신이 관찰할 수 있는 통신 텐서를 분석하려는 `honest-but-curious` 서버를 가정한다. 서버는 다음 정보에 접근할 수 있다.

- 클라이언트 앞단이 전송한 `smashed data z`
- 서버 중간 모델이 생성한 `output u`
- 클라이언트 뒷단이 역전파를 위해 반환한 `gradient dL/du`
- 서버가 합법적으로 보유하거나 공개 데이터로 구성한 소규모 `auxiliary/anchor data`

반면 서버는 공격 대상 이미지의 실제 라벨과 원본 픽셀에 직접 접근할 수 없다고 가정한다. 평가자는 공격 성공 여부를 측정할 때만 실제 라벨과 원본 이미지를 사용한다.

## 제안하는 Gradient Label Inference와 이미지 복원 공격

본 연구는 `U-shaped Split Learning`의 서버가 관찰할 수 있는 `z`, `u`, `dL/du`를 이용해 먼저 라벨을 추론하고, 그 결과를 조건으로 사용하여 원본 이미지의 근사 복원을 시도하는 2단계 공격 및 평가 프레임워크를 제안한다.

### Gradient 기반 라벨 추론

- 학습 과정에서 서버로 반환되는 `dL/du`를 표본별로 수집한다.
- `Gradient`를 평탄화하고 `L2` 정규화한 뒤, 실제 라벨을 사용하지 않고 `K-Means`로 군집화한다.
- 클래스별 `anchor` 표본의 `gradient`를 각 `centroid`와 비교하여 `cluster`를 의미 있는 클래스 `label`에 연결한다.
- 새로운 표본의 `gradient`를 `centroid`와 비교해 `hard label` 또는 클래스별 확률을 나타내는 `soft label condition`을 생성한다.

### Label-conditioned 이미지 복원

- 공격자가 관찰할 수 있는 `z`, `u`, `dL/du`와 추론된 `label condition`만 `decoder` 입력으로 사용한다.
- `Decoder`는 `holdout`이 아닌 `auxiliary` 이미지로 학습하며, 원본 이미지는 이 단계에서 학습 `target`으로만 사용한다.
- 공격 대상 `holdout` 이미지는 `decoder`의 `train/validation`에서 완전히 제외하고, 마지막 평가에서 복원 품질을 측정할 때만 원본과 비교한다.
- 단일 `holdout` 실험과 클래스별 다중 `holdout` 실험을 모두 수행하여 특정 이미지에 의존하지 않는 결과를 측정한다.

### Ablation을 통한 정보 기여도 분석

단순히 복원 이미지 한 장을 제시하는 대신 다음 조건을 비교하여 각 관찰 정보가 복원에 기여하는 정도를 분리한다.

- `z + u + gradient + inferred-soft label`
- `z + u + gradient + oracle label`
- `z + u + gradient + zero label`
- `z only`, `z + u`, `z + gradient`, `gradient only`
- 동일 입력을 사용하는 `baseline decoder`와 `strong decoder`

이를 통해 추론된 `label`이 실제 `oracle label`에 얼마나 가까운 효과를 제공하는지, `label`이 없을 때보다 복원 품질이 향상되는지, 그리고 어떤 통신 텐서가 이미지 정보 누출에 가장 크게 기여하는지를 분석한다.

## 전체 연구 파이프라인

```text
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
anchor mapping으로 label 추론
        ↓
z + u + dL/du + inferred label
        ↓
label-conditioned decoder
        ↓
holdout 이미지의 근사 복원
        ↓
PSNR·SSIM·MSE·MAE·분류 일치율 및 ablation 비교
```

핵심 제안은 다음 한 문장으로 요약할 수 있다.

> `U-shaped Split Learning`에서 서버로 반환되는 `gradient`로 숨겨진 라벨을 추론하고, 그 라벨과 서버 관찰 텐서를 결합하여 원본 이미지의 근사 복원 가능성을 평가한다.

## 핵심 연구 질문

- 실제 라벨을 직접 받지 않는 서버가 `dL/du`만으로 클래스 `label`을 구분할 수 있는가?
- `Client f`와 `Server g`의 `cut` 위치가 `gradient` 기반 라벨 추론 성능에 어떤 영향을 주는가?
- 추론된 `label`을 `decoder` 조건으로 제공하면 `label`이 없는 경우보다 이미지 복원 품질이 향상되는가?
- `Oracle label`과 `inferred-soft label` 사이의 복원 품질 차이는 어느 정도인가?
- `z`, `u`, `dL/du` 가운데 어떤 정보가 복원 성능에 가장 크게 기여하는가?
- 라벨 추론 오류가 최종 이미지 복원 결과에 어떤 형태로 전파되는가?

## 연구 범위와 해석

본 연구는 `U-shaped Split Learning`이 항상 라벨이나 원본 이미지를 노출한다고 가정하지 않는다. 또한 `decoder` 결과를 원본 이미지의 완전한 복구라고 주장하지 않으며, 관찰된 중간 정보가 원본의 클래스와 시각적 특징을 어느 정도 보존하는지를 근사 복원 지표와 `ablation`으로 평가한다.

현재 저장소에는 `gradient` 수집, `K-Means` 군집화, `anchor mapping`, `label inference`, `label-conditioned decoder`, 엄격한 `holdout` 평가, 다중 `holdout ablation suite`가 구현되어 있다. `Label inference`는 `Accuracy`, `Macro F1`, `Purity`, `ARI`, `NMI`로 평가하고, 복원 결과는 `MSE`, `MAE`, `PSNR`, `SSIM`과 재분류 정확도로 평가한다.

세부 실험 방법은 [gradient 군집화 실험 가이드](readme/gradient_clustering_experiment_guide.md)에서 확인할 수 있다.

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

두 통신 경계에서는 텐서를 `detach()`한 뒤 새로운 `leaf tensor`로 연결합니다. 따라서 `autograd graph`가 통신 경계를 그대로 통과하지 않으며, 수신한 `gradient`를 이용해 각 모델에서 역전파를 명시적으로 다시 시작합니다.

공격에 사용하는 핵심 텐서는 다음과 같습니다.

```text
grad_h_to_g = dL/du
```

이 `gradient`는 실제 레이블이 포함된 `cross-entropy loss`로부터 `ClientTail`이 생성하고, `ServerMiddle`의 역전파를 위해 서버에 전달합니다. 서버는 이 `gradient`를 관찰할 수 있지만 실제 레이블은 전달받지 않습니다.

`ServerMiddle.forward(z)`, 서버 `transcript logger`, `K-Means` 함수에는 레이블 인자가 존재하지 않습니다. 또한 `ImageFolder` 경로에는 클래스 디렉터리 이름이 포함되므로, 공격자에게 노출되는 `sample_id`는 파일 경로가 아닌 결정론적인 불투명 해시를 사용합니다.
