# Paired-to-Unpaired Architecture-Agnostic Bidirectional Transcript Reconstruction 연구 계획

## 1. 연구 목표

이 연구는 U-shaped Split Learning의 honest-but-curious 서버가 client-side 모델 `f`, `h`의
구조와 가중치를 알지 못하는 조건에서, 정상적인 학습 과정 중 관측한 forward/backward
중간값으로 client 입력과 시각적ㆍ의미적으로 유사한 이미지를 복원할 수 있는지 연구한다.

본 연구의 핵심 실험축은 두 가지다.

1. 서버가 원본 이미지와 transcript의 정확한 `pair`를 얼마나 보유하는가?
2. `z`, `u`, `dL/du`, `dL/dz` 중 어떤 신호가 architecture-unknown 복원에 기여하는가?

가칭 방법명은 다음과 같다.

> ABTR: Architecture-Agnostic Bidirectional Transcript Reconstruction

논문은 exact pair를 모두 보유한 현재 Decoder를 공격 성능의 upper bound로 유지하면서,
limited-paired, semi-paired, strict-unpaired 조건까지 단계적으로 평가한다.

## 2. 표기법과 서버 관측 정보

U-shaped Split Learning 모델을 다음과 같이 정의한다.

```text
x -> client front f -> z -> server middle g -> u -> client tail h -> loss L
```

- `x`: client private image
- `y`: client private label
- `f`: client front model
- `g`: server middle model
- `h`: client tail model
- `z = f(x)`: client가 서버로 전송한 smashed data
- `u = g(z)`: 서버가 client tail로 전송한 middle output
- `r_u = dL/du`: client tail이 서버로 반환한 gradient

서버는 `g`를 소유하므로 정상적인 역전파 과정에서 다음 값도 계산한다.

```text
r_z = dL/dz = J_g(z)^T r_u
```

본 연구에서는 다음을 한 sample의 bidirectional transcript로 정의한다.

```text
T_i = (z_i, u_i, r_u_i, r_z_i)
```

`z`, `u`는 forward 정보이고 `r_u`, `r_z`는 backward sensitivity 정보다.

## 3. Pair의 정확한 의미

이 문서에서 `pair`는 후보라는 뜻이 아니다. 같은 sample에서 나온 원본 이미지와
transcript의 정확한 대응 관계를 의미한다.

```text
exact pair = (x_i, T_i)
```

Exact pair가 있으면 Decoder 출력과 대응 원본을 직접 비교할 수 있다.

```text
x_hat_i = D(T_i)
L_pair = L_reconstruction(x_hat_i, x_i)
```

현재 구현은 auxiliary image를 victim 전체 모델에 통과시켜 transcript를 만든다. 따라서
auxiliary 구간에서는 `(x_i, T_i)`가 정확히 연결되어 있다. 이 방식은 복원 가능성의
upper bound로는 유효하지만, 실제 서버가 그 pair를 어떻게 획득했는지 별도 가정이 필요하다.

Unpaired 조건에서는 다음 두 집합만 존재하며 sample 대응은 알 수 없다.

```text
X_public  = {x_1, x_2, ..., x_m}
T_private = {T_1, T_2, ..., T_n}
```

이때 임의의 `x_j`를 `T_i`의 정답으로 붙여 reconstruction loss를 계산하면 안 된다.
Unpaired 학습에서는 sample loss 대신 class-conditional distribution alignment를 사용한다.

## 4. Pair knowledge 실험축

### P0. Full-paired upper bound

- 모든 auxiliary `(x_i, T_i)`가 정확하게 대응된다.
- 현재 Decoder 학습 조건이다.
- 실제 공격의 기본 조건이 아니라 복원 성능 상한선으로 해석한다.

### P1. Limited-paired

- 정확한 pair 중 일부만 공격자에게 제공한다.
- 비율: `1%, 5%, 10%, 25%, 50%, 100%`
- 선택은 class-stratified 방식으로 수행한다.
- pair sample efficiency curve를 측정한다.

### P2. Class-shuffled negative control

- transcript와 target image의 sample identity는 다르다.
- 단, source와 target의 class는 동일하다.
- 정확한 pixel/instance pair와 class-level prior의 기여를 분리한다.

```text
(T_i, x_j), i != j, y_i = y_j
```

### P3. Global-shuffled negative control

- target image를 전체 auxiliary set에서 섞는다.
- sample identity와 class correspondence를 모두 파괴한다.
- 잘못된 pair를 사용한 supervised loss가 만드는 실패 양상을 확인한다.

```text
(T_i, x_j), i != j, class constraint 없음
```

### P4. Strict-unpaired ABTR

- public image와 private transcript 사이의 pair가 전혀 없다.
- victim `f`, `h`의 구조와 가중치도 모른다.
- 잘못된 pseudo-pair를 만들지 않고 distribution alignment로 학습한다.
- 본 연구가 최종적으로 목표하는 공격 조건이다.

### P5. Semi-paired ABTR

- 소수의 exact auxiliary pair와 다수의 unpaired public/private records를 함께 사용한다.
- 현실적인 calibration 또는 known-sample 공격 시나리오를 나타낸다.
- exact pair의 supervised loss와 unpaired alignment loss를 동시에 사용한다.

### Pair 조건 요약

| ID | Exact sample pair | Class 대응 | Unpaired data | 역할 |
|---|---:|---:|---:|---|
| P0 Full-paired | 100% | 있음 | 없음 | upper bound |
| P1 Limited-paired | 1~50% | 있음 | 선택 | pair budget 분석 |
| P2 Class-shuffled | 0% | 있음 | 없음 | class-only negative control |
| P3 Global-shuffled | 0% | 없음 | 없음 | random negative control |
| P4 Strict-unpaired | 0% | pseudo label만 사용 | 있음 | main attack |
| P5 Semi-paired | 소량 | 있음 | 있음 | practical mixed setting |

P2와 P3는 새로운 공격 방식이 아니라 pair가 실제로 중요한지 검증하는 통제 실험이다.

## 5. 중요한 한계: 중간값만으로는 원본을 유일하게 결정할 수 없음

Client `f`가 완전히 알려지지 않았을 때 `z = f(x)`만으로 `x`를 유일하게 결정하는 것은
일반적으로 불가능하다. 서로 다른 입력과 서로 다른 client 함수가 같은 `z`를 만들 수 있기
때문이다. `u = g(z)`도 `z`의 결정적 함수이므로 이 비식별성을 제거하지 못한다.

따라서 architecture-unknown 복원에는 최소한 다음 중 하나가 필요하다.

- victim private set과 겹치지 않는 public auxiliary images
- 공개 데이터로 사전학습된 image generator 또는 diffusion model
- 입력 domain, 해상도, 채널 수와 같은 최소 domain knowledge

본 논문의 기본 조건은 `transcript만 사용하고 어떠한 prior도 사용하지 않는다`가 아니다.
다음과 같이 제한해서 주장한다.

> Client 모델의 구조ㆍ가중치ㆍprivate 원본ㆍprivate 실제 라벨ㆍ임의 query에는 접근하지
> 않고, 서버가 정상적으로 관측한 transcript와 disjoint public auxiliary images만 사용한다.

## 6. 위협 모델

### 6.1 공격자

공격자는 protocol을 변경하지 않는 honest-but-curious 서버다. Client update를 조작하거나
악성 gradient를 전송하지 않고 관측값을 수동적으로 기록한다.

### 6.2 공격자가 아는 정보

- 서버 소유 middle model `g`의 구조와 현재 가중치
- communication interface tensor의 shape와 dtype
- 각 step에서 관측한 `z`, `u`, `dL/du`
- `g`의 역전파로 계산한 `dL/dz`
- class 수
- victim private set과 겹치지 않는 public auxiliary images
- cluster-to-class mapping에 사용하는 소수 public/anchor samples
- P0/P1/P5 실험에서 명시된 수량의 exact auxiliary pair

### 6.3 공격자가 모르는 정보

- client front `f`의 layer 구성, 깊이, 가중치
- client tail `h`의 layer 구성, 깊이, 가중치
- victim private image와 private true label
- victim client에 대한 임의 입력 query 결과
- P4 조건에서 public image와 private transcript의 sample pair

### 6.4 평가자만 사용하는 정보

- victim holdout 원본 이미지
- victim holdout 실제 라벨

이 값은 metric 계산에만 사용하고 공격 학습 모듈로 전달하지 않는다.

## 7. 연구 질문과 가설

### RQ1. Exact pair 수량은 복원 품질에 어떤 영향을 주는가?

가설: pair 수가 증가하면 instance-level metric은 향상되지만, 일정 구간 이후 수익이
감소한다. 이 curve로 공격에 필요한 최소 pair budget을 보고한다.

### RQ2. Exact identity pair가 중요한가, 같은 class pair만으로 충분한가?

가설: class-shuffled pair는 classifier consistency를 일부 유지하지만 PSNR, SSIM, LPIPS와
같은 instance metric에서는 exact pair보다 크게 낮아진다.

### RQ3. Strict-unpaired ABTR은 shuffled-pair 학습보다 우수한가?

가설: 잘못된 sample loss를 사용하는 것보다 joint distribution alignment가 transfer에
유리하다.

### RQ4. Forward와 backward transcript를 함께 사용하면 `z`만 사용할 때보다 향상되는가?

가설: `dL/du`, `dL/dz`에는 loss와 label에 민감한 channelㆍspatial 방향이 포함되어
semantic consistency와 reconstruction quality가 향상된다.

### RQ5. `dL/dz`가 `dL/du`보다 복원에 더 직접적으로 유용한가?

가설: `dL/dz`는 `z`와 동일한 shape이므로 `z * normalize(dL/dz)`와 같은 위치 기반
결합으로 중요한 latent 영역을 강조할 수 있다.

### RQ6. Gradient pseudo label이 unpaired alignment의 모호성을 줄이는가?

가설: 전체 분포만 맞추는 것보다 class-conditional alignment가 class permutation과
mode mixing을 줄인다.

### RQ7. Victim과 다른 surrogate architecture에서도 동작하는가?

가설: exact layer 복제 대신 transcript distribution을 정렬하므로 다른 architecture에서도
paired upper bound보다 낮지만 유의미한 복원이 가능하다.

## 8. 제안 방법 개요

```text
Victim private branch

private x -> unknown f -> z -> known g -> u -> unknown h -> loss
                         ^             |
                         |             v
                       dL/dz <- known g <- dL/du

T_private = (z, u, dL/du, dL/dz, inferred-label)


Public auxiliary branch

public x -> arbitrary f_hat -> z_hat -> known g -> u_hat -> arbitrary h_hat
                                                        |
                                                        v
                                                    dL/du_hat

T_public = (z_hat, u_hat, dL/du_hat, dL/dz_hat, public-label)


Training

paired subset   -> supervised reconstruction loss
unpaired sets   -> class-conditional joint transcript alignment
all conditions  -> bidirectional Decoder
```

## 9. ABTR 세부 방법

### 9.1 실제 transcript 수집

Private attacker record에는 다음 값만 저장한다.

```text
z, u, grad_u, grad_z, opaque_sample_id, training_step, model_version
```

`grad_z`는 서버가 다음과 같이 계산한다.

```python
u = g(z_leaf)
u.backward(grad_u)
grad_z = z_leaf.grad
```

Private target과 true label은 evaluator-only storage에 분리한다.

### 9.2 Gradient label inference

기존 pipeline을 재사용한다.

1. `dL/du` sample-wise normalization
2. PCA 또는 random projection
3. K-Means clustering
4. 소수 anchor로 cluster-to-class mapping
5. hard/soft pseudo label 생성

비교 조건은 `zero`, `inferred-hard`, `inferred-soft`, `oracle`로 분리한다. Oracle은 upper
bound에서만 사용한다.

### 9.3 Architecture-agnostic `f_hat`

`f_hat`은 victim `f`를 상속하거나 복사하지 않는다. Public image를 입력받아 observed `z`와
같은 tensor shape을 출력하는 임의의 encoder다.

- PlainCNN simulator
- Residual simulator

Victim과 simulator architecture가 다른 cross-architecture 설정을 main table에 포함한다.

### 9.4 Gradient-matched `h_hat`

`h_hat`도 victim `h`를 복사하지 않는다. 실제 `(u_i, r_u_i)`와 inferred label
`y_hat_i`를 이용한다.

```text
r_hat_i = d CE(h_hat(u_i), y_hat_i) / du_i

L_tail = MSE(normalize(r_hat_i), normalize(r_u_i))
       + alpha * (1 - cosine(r_hat_i, r_u_i))
```

Second-order gradient를 위해 `create_graph=True`를 사용한다.

### 9.5 Public synthetic transcript

Public sample `(x_a, y_a)`에서 다음을 계산한다.

```text
z_hat   = f_hat(x_a)
u_hat   = g(z_hat)
r_u_hat = d CE(h_hat(u_hat), y_a) / du_hat
r_z_hat = J_g(z_hat)^T r_u_hat
```

이 과정에서는 victim `f`, `h`를 호출하지 않는다.

### 9.6 Class-Conditional Joint Transcript Alignment

각 tensor를 adaptive pooling, channel statistics, fixed random projection, L2 normalization으로
embedding한다. Public synthetic transcript와 private real transcript의 class-conditional
distribution을 맞춘다.

```text
L_align = sum_c [
    JMMD(T_public | y=c, T_private | y_hat=c)
    + beta * CORAL(T_public | y=c, T_private | y_hat=c)
]
```

Joint kernel은 view 관계를 보존하도록 정의한다.

```text
k(T_i, T_j) = k_z * k_u * k_ru * k_rz
```

초기 구현에서는 안정성을 위해 view별 MMD + CORAL로 시작하고 이후 JMMD로 확장한다.

### 9.7 Bidirectional Decoder

Decoder 입력은 다음과 같다.

```text
z
u
normalize(r_u)
normalize(r_z)
z * normalize(r_z)
label condition
```

각 signal은 별도 encoder와 adaptive pooling을 거친 뒤 결합한다. 복원 loss는 기존
`L1 + SSIM + edge`를 재사용한다.

```text
L_rec = lambda_l1 * L1(x_recon, x_target)
      + lambda_ssim * (1 - SSIM(x_recon, x_target))
      + lambda_edge * L_edge
```

## 10. Pair 조건별 학습 목적함수

### 10.1 Full/Limited-paired

정확한 paired subset에서 supervised loss를 사용한다.

```text
L_P0/P1 = L_rec_paired + lambda_tail * L_tail
```

### 10.2 Class/Global-shuffled

Target correspondence만 바꾸고 다른 설정을 고정한다.

```text
L_P2/P3 = L_rec_shuffled
```

이 결과는 공격 성능으로 주장하지 않고 pair correspondence negative control로만 사용한다.

### 10.3 Strict-unpaired

Private target을 학습에 전혀 사용하지 않는다.

```text
L_P4 = lambda_align * L_align
     + lambda_public_rec * L_rec_public
     + lambda_tail * L_tail
     + lambda_reg * L_regularization
```

### 10.4 Semi-paired

소수 exact pair와 unpaired records를 결합한다.

```text
L_P5 = lambda_pair * L_rec_paired
     + lambda_align * L_align_unpaired
     + lambda_public_rec * L_rec_public
     + lambda_tail * L_tail
```

Pair fraction이 0이면 P4, 100%이고 alignment가 없으면 P0가 되므로 P5를 통해 두 조건을
하나의 연속적인 curve로 분석할 수 있다.

## 11. Pair 생성 및 실험 규칙

### 11.1 Exact pair budget

- train/validation에서 class-stratified sampling
- fraction: `0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00`
- seed별 selected sample ID 저장
- train과 validation의 pair ID 중복 금지

### 11.2 Class-shuffled mapping

- class별 sample index를 seed로 shuffle
- 같은 sample로 다시 매핑되지 않도록 derangement 또는 cyclic shift
- `source_sample_id`, `target_sample_id`를 결과 manifest에 저장
- 모든 pair에서 `source_label == target_label`, `source_id != target_id` 검증

### 11.3 Global-shuffled mapping

- 전체 sample을 seed로 shuffle
- 동일 sample mapping 금지
- class 일치는 강제하지 않음
- 실제 class 일치율을 결과에 기록

### 11.4 Strict-unpaired

- private transcript loader는 `target_image`를 반환하지 않음
- public image loader는 private transcript ID를 알지 못함
- sample-wise reconstruction loss 호출 시 test가 실패해야 함

### 11.5 Semi-paired

- paired subset과 unpaired subset의 sample ID 분리
- paired subset만 `L_rec_paired` 계산
- unpaired subset에는 `L_align`만 계산
- 같은 private record가 paired와 unpaired 양쪽에 중복되지 않도록 검증

## 12. 필수 비교와 ablation

### 12.1 Pair ablation

- P0 Full-paired
- P1 Limited-paired curve
- P2 Class-shuffled
- P3 Global-shuffled
- P4 Strict-unpaired ABTR
- P5 Semi-paired curve

Main pair figure는 x축을 exact pair fraction, y축을 PSNR/SSIM/LPIPS로 표시한다. P2와 P3는
별도 marker로 표시한다.

### 12.2 Signal ablation

```text
z
z + u
z + r_u
z + r_z
z + r_z + z*r_z
z + u + r_u + r_z
전체 + inferred label
전체 + oracle label
```

### 12.3 Alignment ablation

- alignment 없음
- `z` marginal alignment
- view별 MMD
- MMD + CORAL
- joint MMD
- class-unconditional
- class-conditional

### 12.4 Architecture ablation

- victim과 같은 simulator: upper bound
- victim ResNet / attacker PlainCNN
- victim PlainCNN / attacker Residual simulator
- width mismatch
- early/middle/late cut

Cut depth는 공격 입력으로 제공하지 않는다. 공격자는 관측 tensor shape만 사용하고,
연구자가 결과 분석 변수로 실제 depth를 기록한다.

### 12.5 Prior ablation

- public data: `1%, 5%, 10%, 25%, 100%`
- same-domain public data
- related but shifted-domain public data
- anchor: `0, 1, 3, 5` samples per class

## 13. 비교 대상

### 내부 baseline

- 현재 paired Decoder
- paired strong Decoder
- shuffled-target Decoder
- z-only unpaired alignment
- ABTR zero-label
- ABTR inferred-label
- ABTR oracle-label upper bound

### 외부 baseline

- UnSplit: exact client architecture known, auxiliary data 없음
- FORA: public data 기반 feature-oriented alignment
- SDAR: adversarially regularized simulator decoding
- MAERA: U-shaped model approximation reconstruction

| Method | Client architecture | Client weights | Exact `(x,T)` | Public data | Backward transcript |
|---|---|---|---|---|---|
| Current paired Decoder | 불필요 | observation 생성에 사용 | 필요 | 필요 | 선택 |
| UnSplit | 필요 | 불필요 | 불필요 | 불필요 | 불필요 |
| FORA/SDAR/MAERA 계열 | 불필요 또는 mismatch | 불필요 | 불필요/소량 | 필요 | 제한적 |
| Strict-unpaired ABTR | 불필요 | 불필요 | 불필요 | 필요 | 사용 |
| Semi-paired ABTR | 불필요 | 불필요 | 소량 | 필요 | 사용 |

Threat model이 다른 결과를 단순 순위로 비교하지 않고 필요한 prior를 항상 함께 표시한다.

## 14. 데이터셋과 모델

현재 cat/dog/pug 3-class 데이터는 pipeline debugging과 qualitative example에 사용한다.
논문 main result는 최소 다음을 목표로 한다.

- CIFAR-10
- Oxford-IIIT Pet 또는 STL-10
- victim SmallCNN + ResNet 계열
- attacker PlainCNN + Residual simulator


- CIFAR-100 또는 Tiny ImageNet
- 세 번째 victim architecture
- out-of-domain auxiliary dataset
- 방어 기법 2종 이상

## 15. 평가 지표와 통계

### 이미지 복원

- MSE, MAE
- PSNR, SSIM
- LPIPS
- classifier prediction consistency
- pretrained embedding cosine similarity

### 라벨 추론

- Accuracy, Macro F1
- Purity, ARI, NMI

### Pair 영향

- exact pair fraction 대비 metric curve
- P0 대비 P2/P3 성능 하락량
- P4 대비 P5의 pair 1개당 개선량
- class consistency와 instance fidelity의 차이

### 공격 비용

- attack training time
- sample당 inference time
- peak GPU memory
- public data 및 exact pair 수

각 결과는 최소 3 seeds의 평균과 표준편차를 보고한다. Main result에는 bootstrap 95%
confidence interval을 추가한다.

## 16. 데이터 누출 방지 규칙

1. Private attacker record에는 원본과 true label이 없어야 한다.
2. Strict-unpaired public pipeline은 victim `f`, `h` 객체를 인자로 받지 않아야 한다.
3. Private holdout ID가 auxiliary train/validation manifest에 존재하면 실패해야 한다.
4. 파일 경로와 directory class name을 공격 입력으로 사용하지 않는다.
5. Oracle label은 명시적인 ablation flag가 없으면 사용할 수 없다.
6. Evaluation target loader를 attack training module에서 import하지 않는다.
7. Pair manifest는 `source_sample_id`, `target_sample_id`, `pair_type`을 기록한다.
8. P2/P3에서 동일 sample mapping이 발견되면 실패한다.
9. P4에서 sample-wise private reconstruction loss가 호출되면 실패한다.
10. Run config에 threat model, pair fraction, corruption mode, seed를 기록한다.

## 17. 구현 계획

현재 pipeline을 유지하고 pair/unpaired 실험을 별도 모듈로 추가한다.

```text
src/decoder/
  data/
    pairing_ablation.py
    transcript_dataset.py
    public_auxiliary_dataset.py
  models/
    architecture_agnostic_f_model.py
    gradient_matched_h_model.py
    bidirectional_transcript_decoder.py
  losses/
    transcript_alignment_loss.py
    gradient_matching_loss.py
  training/
    paired_unpaired_trainer.py
    transcript_alignment_trainer.py
  pipeline/
    run_pairing_ablation.py
    run_abtr_experiment.py
```

### 계획 CLI

```text
--pairing-mode exact
--pairing-mode class-shuffled
--pairing-mode global-shuffled
--pairing-mode unpaired
--pairing-mode semi-paired

--paired-data-fraction 0.00|0.01|0.05|0.10|0.25|0.50|1.00
--pair-corruption-fraction 0.00~1.00
--pairing-seed 42
```

각 실행의 `run_config.json`과 `pairing_manifest.csv`에 실제 선택 및 매핑 결과를 저장한다.

### 구현 순서

1. Pairing dataset wrapper와 manifest 구현
2. P0/P1/P2/P3 실험 CLI 및 test
3. 서버의 `dL/dz` 수집과 Decoder branch 구현
4. Signal ablation으로 backward signal의 유효성 확인
5. Architecture-agnostic `f_hat`, `h_hat` 구현
6. P4 strict-unpaired alignment 구현
7. P5 semi-paired combined loss 구현
8. External baseline과 defense 실험

## 18. Go/No-Go 기준

### Gate 1: Pair study

Go:

- P0 > P2 > P3의 경향이 instance metric에서 여러 seed에 걸쳐 반복됨
- limited pair curve에서 일관된 sample-efficiency 경향이 관측됨

No-Go 또는 재설계:

- shuffled pair와 exact pair 차이가 거의 없음
- Decoder가 class prototype만 생성하여 instance metric이 의미가 없음

### Gate 2: Backward signal

Go:

- 두 개 이상의 split depth에서 `z+dL/dz`가 `z-only`보다 SSIM, LPIPS 또는 classifier
  consistency 중 하나를 반복적으로 개선

No-Go:

- oracle label에서만 개선됨
- 여러 seed에서 개선 방향이 불안정함

### Gate 3: Strict-unpaired

Go:

- P4가 P3 global-shuffled보다 유의하게 우수함
- architecture mismatch에서도 random/public prototype baseline보다 우수함

No-Go:

- distribution alignment가 collapse함
- P4가 class 평균 이미지만 출력함
- 외부 baseline보다 모든 조건에서 열세임

## 19. 실패 위험과 대응

### Unpaired alignment가 class prototype만 복원

- pixel-perfect recovery claim을 하지 않는다.
- semantic/attribute leakage로 범위를 제한한다.
- LPIPS, embedding similarity, classifier consistency를 함께 보고한다.

### MMD가 고차원에서 불안정

- adaptive pooling과 fixed random projection 사용
- CORAL warm-up
- class-balanced batch sampler

### Pseudo label 오류

- hard label 대신 confidence-weighted soft condition
- confidence가 낮은 sample의 alignment weight 감소
- zero/inferred/oracle 결과를 모두 보고

### 기존 SDAR/MAERA와 novelty 중복

- simulator alignment 자체를 contribution으로 주장하지 않는다.
- pair-to-unpaired curve, `dL/dz`, gradient-matched tail, forward/backward joint alignment를
  핵심으로 둔다.
- 전체 literature review 전에는 `first`를 사용하지 않는다.

### 현재 victim 모델이 단순함

- ResNet 계열 victim 추가
- cross-architecture 결과를 main table에 배치
- SmallCNN은 controlled ablation으로만 사용

## 20. 예상 논문 기여

실험이 성공할 경우 기여는 다음과 같이 정리한다.

1. U-shaped Split Learning reconstruction에서 exact pair availability를 full-paired부터
   strict-unpaired까지 연속적으로 평가하는 framework를 제시한다.
2. Exact pair, same-class false pair, random false pair를 분리하여 기존 paired Decoder 결과가
   실제 sample correspondence에 얼마나 의존하는지 정량화한다.
3. Client architecture와 weights를 모르는 서버를 위한 bidirectional transcript 기반 passive
   reconstruction 방법을 제안한다.
4. 서버가 정상적으로 계산하는 `dL/dz`의 추가 leakage를 분석한다.
5. Gradient pseudo label을 이용한 class-conditional forward/backward transcript alignment를
   평가한다.

피해야 하는 표현:

- 원본을 완벽하게 복구한다.
- 아무 prior 없이 복원한다.
- 모든 Split Learning에서 동작한다.
- Shuffled pair 결과를 실제 공격 성능으로 해석한다.

권장 표현:

> We study reconstruction leakage across paired, semi-paired, and strictly unpaired threat models,
> and reconstruct visually and semantically similar inputs from passively observed U-shaped
> split-learning transcripts without access to client-side architectures or weights.

## 21. 논문 구성 초안

1. Introduction
2. Background and Related Work
3. Threat Model and Pair Knowledge Levels
4. Non-identifiability without Image Priors
5. ABTR Method
6. Pair Availability and Correspondence Experiments
7. Architecture-Unknown Reconstruction Results
8. Ablation and Defense Evaluation
9. Limitations and Ethics
10. Conclusion

## 22. 관련 연구

- UnSplit: <https://arxiv.org/abs/2108.09033>
- UnSplit code: <https://github.com/ege-erdogan/unsplit>
- FORA, CVPR 2024: <https://openaccess.thecvf.com/content/CVPR2024/html/Xu_A_Stealthy_Wrongdoer_Feature-Oriented_Reconstruction_Attack_against_Split_Learning_CVPR_2024_paper.html>
- SDAR, NDSS 2025: <https://www.ndss-symposium.org/wp-content/uploads/2025-30-paper.pdf>
- SDAR code: <https://github.com/zhxchd/SDAR_SplitNN>
- MAERA/DCRA: <https://doi.org/10.1016/j.neunet.2025.107150>

## 23. 즉시 수행할 작업

```text
Task A: exact/limited/class-shuffled/global-shuffled pair dataset wrapper를 구현한다.
Task B: 선택된 source-target mapping을 manifest로 저장하고 누출 방지 test를 작성한다.
Task C: P0/P1/P2/P3를 동일 seed와 설정으로 실행한다.
Task D: 서버가 계산한 dL/dz를 observation에 추가한다.
Task E: z-only / z+dL/du / z+dL/dz / z+dL/dz+label을 비교한다.
Task F: 두 gate를 통과하면 P4 strict-unpaired ABTR 구현으로 진행한다.
```

가장 먼저 전체 ABTR을 한 번에 구현하지 않는다. Pair study와 backward signal ablation을
통해 핵심 가설을 빠르게 검증한 뒤 strict-unpaired 공격으로 확장한다.
