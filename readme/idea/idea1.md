# Idea 1: 추론 라벨·Gradient·중간 데이터를 이용한 조건부 이미지 복원

## 1. 연구 목표

이 실험은 gradient 기반 라벨 추론이 끝났다고 가정하고 시작한다. 공격자는 각
피해 표본에 대해 추론 라벨과 서버에서 관찰 가능한 값을 결합하여 원본과
시각적·의미적으로 유사한 이미지를 근사 복원한다.

최종 공격은 다음과 같다.

```text
관찰값:
  smashed data z
  server output u
  h → g gradient dL/du

이미 완료된 단계:
  gradient → 추론 라벨 ŷ 또는 soft label q

이번 실험:
  (z, u, dL/du, ŷ) → Label-Conditioned Reconstruction Model → 복원 이미지 x̂
```

라벨 군집화와 cluster-to-label 매핑 과정은 이 문서의 범위에 포함하지 않는다.

## 2. 핵심 가설

1. Smashed data z에는 대상 이미지의 공간 구조와 표본별 특징이 남아 있다.
2. Gradient dL/du에는 라벨과 loss에 따른 정보가 남아 있다.
3. 추론 라벨 ŷ는 decoder가 올바른 클래스의 시각적 특징을 생성하도록 돕는다.
4. z, u, gradient, 추론 라벨을 함께 사용하면 z만 사용하는 복원보다 품질이
   향상될 수 있다.
5. 실제 라벨 대신 82% 정확도의 추론 라벨을 사용했을 때 발생하는 복원 품질
   감소를 측정할 수 있다.

## 3. 공격자 관점과 모델 역할

U-shaped Split Learning의 정상 통신은 다음과 같다.

```text
x → client f → z → server g → u → client h → logits/loss
                                      ↓
                              dL/du를 server g로 반환
```

서버 공격자가 관찰하거나 소유하는 값은 다음과 같다.

| 항목 | 기호 | 공격자 접근 |
|---|---|---|
| 원본 이미지 | x | 접근 불가 |
| Smashed data | z | 관찰 가능 |
| Server output | u | 관찰 가능 |
| h → g gradient | dL/du | 관찰 가능 |
| 실제 라벨 | y | 접근 불가 |
| 추론 라벨 | ŷ | 기존 공격 결과로 보유 |
| 서버 모델 | g | 서버가 원래 소유 |
| 클라이언트 모델 | f, h | 접근 불가 |

서버는 g를 원래 소유하므로 g를 다시 복제할 필요가 없다. 이번 실험에서는
클라이언트 측 기능을 모사하는 SurrogateF와 SurrogateH를 학습하고, 이미지
복원을 담당하는 별도의 decoder를 학습한다.

### 3.1 SurrogateF

SurrogateF는 공개 보조 이미지가 피해자 f에서 어떤 smashed data를 만들지
근사한다.

```text
f̂(x_aux) ≈ z_aux
```

SurrogateF는 이미지 복원 모델이 아니다. 입력 이미지를 피해자 f의 latent
공간과 정렬된 smashed data로 변환하는 역할을 한다.

### 3.2 SurrogateH

SurrogateH는 u에서 logits를 계산하고, 알려진 보조 라벨을 사용했을 때 피해자
h와 유사한 dL/du를 생성하도록 학습한다.

```text
ĝ_aux = ∂CE(ĥ(u_aux), y_aux) / ∂u_aux
ĝ_aux ≈ g_aux
```

SurrogateH도 이미지 복원 모델이 아니다. 피해자 h가 만드는 gradient 패턴을
모사하여 보조 학습 쌍을 확장하는 역할을 한다.

### 3.3 Label-Conditioned Reconstruction Model

실제 복원은 별도의 모델 R이 담당한다.

```text
x̂ = R(z, u, dL/du, label_condition)
```

label_condition은 다음 중 하나가 될 수 있다.

- 실제 라벨 one-hot: 복원 성능의 상한을 측정할 때만 사용
- 추론 라벨 one-hot: 실제 공격 조건
- Cluster 거리 기반 soft label: 라벨 추론 오류를 완화하는 조건

## 4. 필요한 데이터

Decoder를 학습하려면 이미지뿐 아니라 해당 이미지와 관찰값이 연결된 보조
학습 쌍이 필요하다.

```text
(x_aux, z_aux, u_aux, gradient_aux, y_aux)
```

단순히 cat, dog, pug 이미지만 보유하고 해당 이미지의 z와 gradient를 얻지
못하면 피해자 latent 공간에 맞는 supervised decoder를 학습할 수 없다.

### 4.1 데이터 분리

```text
Auxiliary train set
  SurrogateF, SurrogateH, decoder 학습

Auxiliary validation set
  모델 선택과 early stopping

Victim test set
  공격 시 z, u, gradient, 추론 라벨만 사용
  실제 원본과 실제 라벨은 마지막 평가에서만 사용
```

Victim test 원본을 decoder 학습에 사용하면 복원이 아니라 암기가 되므로 절대
혼합하지 않는다. 동일 이미지, 파일 해시, 촬영 연속 프레임도 split 사이에
겹치지 않도록 확인한다.

현재 클래스당 30장의 train 이미지는 파이프라인 동작을 확인하는 작은
proof-of-concept에는 사용할 수 있지만, 일반화 가능한 decoder를 학습하기에는
부족할 가능성이 높다. 본 실험에서는 클래스별 수백 장 이상의 서로 겹치지 않는
보조 이미지를 권장한다.

## 5. 학습용 Observation 생성

먼저 white-box 상한 실험에서 실제 f, g, h를 고정하고 보조 이미지에 대한
학습 쌍을 생성한다.

```text
z_aux = f(x_aux)
u_aux = g(z_aux)
logits_aux = h(u_aux)
loss_aux = CE(logits_aux, y_aux)
gradient_aux = ∂loss_aux / ∂u_aux
```

각 레코드는 다음 정보를 가진다.

```text
sample_id
image_path 또는 evaluator-only target
true_label
smashed_z
server_output_u
grad_h_to_g
cut_config
checkpoint_id
```

공격 코드가 실제 원본 경로나 true_label을 읽지 않도록 학습용 보조 레코드,
공격자 transcript, evaluator ground truth를 물리적으로 분리한다.

## 6. 단계별 모델 학습

### 6.1 Stage A: White-box 복원 상한 확인

처음부터 f와 h 복제 문제까지 동시에 풀지 않는다. 실제 모델로 생성한 정확한
z와 gradient를 사용하여 decoder 자체가 이미지를 복원할 수 있는지 먼저
확인한다.

```text
R(z_aux, u_aux, gradient_aux, y_aux) → x_aux
```

이 단계가 실패하면 SurrogateF와 SurrogateH를 추가해도 복원되지 않는다.

### 6.2 Stage B: SurrogateF 학습

```text
ẑ_aux = f̂(x_aux)
```

권장 손실:

```text
L_f =
    λ_f_mse · MSE(ẑ_aux, z_aux)
  + λ_f_cos · (1 - cosine(ẑ_aux, z_aux))
  + λ_f_stat · StatisticsLoss(ẑ_aux, z_aux)
```

StatisticsLoss는 채널별 mean과 standard deviation 차이를 측정한다. 단순히
동일한 구조의 f를 분류 목적으로 학습하는 것이 아니라, 피해자 f의 latent
좌표계와 출력값을 맞추는 distillation이 필요하다.

### 6.3 Stage C: SurrogateH 학습

```text
logits_hat = ĥ(u_aux)
gradient_hat = ∂CE(logits_hat, y_aux) / ∂u_aux
```

권장 손실:

```text
L_h =
    λ_h_grad · MSE(gradient_hat, gradient_aux)
  + λ_h_cos  · (1 - cosine(gradient_hat, gradient_aux))
  + λ_h_cls  · CE(logits_hat, y_aux)
```

Gradient matching에는 2차 미분이 필요할 수 있으므로 PyTorch에서
create_graph=True로 gradient graph를 유지해야 한다. 메모리 사용량이 커질 수
있으므로 작은 batch로 먼저 검증한다.

### 6.4 Stage D: Label-Conditioned Decoder 학습

복원 모델은 입력 종류별 encoder와 fusion decoder로 나눈다.

```text
z ───────────→ SmashedDataEncoder ─┐
                                   │
u ───────────→ ServerOutputEncoder ├→ FeatureFusion
                                   │
dL/du ───────→ GradientEncoder ────┤
                                   │
label ───────→ LabelEmbedding ─────┘
                                           ↓
                                  Upsampling Decoder
                                           ↓
                                     복원 이미지 x̂
```

Gradient와 smashed data의 shape은 cut 위치에 따라 달라질 수 있으므로 각
입력 encoder가 고정 크기의 feature map으로 변환하도록 adapter를 둔다.

## 7. Decoder 손실함수

하나의 pixel loss만 사용하면 흐릿한 이미지가 생성될 수 있으므로 여러 목적을
결합한다.

```text
L_reconstruction =
    λ_l1    · L1(x̂, x)
  + λ_ssim  · (1 - SSIM(x̂, x))
  + λ_lpips · LPIPS(x̂, x)
  + λ_cls   · CE(C(x̂), y)
  + λ_feat  · FeatureConsistency(x̂, x)
```

각 항목의 역할:

| 손실 | 역할 |
|---|---|
| L1 | 전체 색상과 pixel 차이 감소 |
| SSIM | 구조와 지역 패턴 보존 |
| LPIPS | 사람이 느끼는 시각적 유사성 향상 |
| Classification loss | cat, dog, pug 클래스 특징 유지 |
| Feature consistency | 고수준 형태와 의미 보존 |

Classification loss의 C는 decoder와 별도로 고정한 평가용 분류기를 사용한다.
실제 공격 성능 평가에서는 C 학습 데이터와 victim test를 분리한다.

## 8. 실제 공격 추론

피해 표본에 대해 서버가 다음 값을 관찰했다고 가정한다.

```text
z_target
u_target
gradient_target
```

기존 라벨 추론기가 다음을 반환한다.

```text
hard label: ŷ_target = cat

또는

soft label:
q_target = [P(cat)=0.72, P(dog)=0.23, P(pug)=0.05]
```

최종 복원:

```text
x̂_target = R(
    z_target,
    u_target,
    gradient_target,
    ŷ_target 또는 q_target
)
```

복원 단계에서는 실제 y_target이나 x_target을 읽지 않는다. 두 값은 복원이
완료된 뒤 evaluator가 지표를 계산할 때만 사용한다.

## 9. 필수 비교 실험

### 9.1 입력 정보별 ablation

| 실험 | Decoder 입력 | 확인할 내용 |
|---|---|---|
| B1 | z | Smashed data만의 복원 성능 |
| B2 | z + u | 서버 출력의 추가 효과 |
| B3 | z + gradient | Gradient의 추가 효과 |
| B4 | z + 실제 라벨 | 라벨 조건의 oracle 상한 |
| B5 | z + 추론 라벨 | 실제 공격 조건 |
| B6 | z + soft label | 라벨 오류 완화 효과 |
| B7 | z + u + gradient + 추론 라벨 | 제안하는 전체 모델 |

핵심 기여는 다음 차이로 측정한다.

```text
추론 라벨의 복원 기여 =
Quality(B5) - Quality(B1)

Gradient의 추가 기여 =
Quality(B7) - Quality(B5)

라벨 추론 오류 비용 =
Quality(B4) - Quality(B5)
```

### 9.2 모델 접근 수준별 비교

| 조건 | f/h 사용 방식 | 의미 |
|---|---|---|
| White-box upper bound | 실제 f와 h로 observation 생성 | 최대 복원 가능성 |
| Surrogate attack | f̂와 ĥ로 보조 observation 생성 | 현실적인 복제 오차 포함 |
| No h clone | 실제 관찰 gradient만 사용 | h 복제 필요성 검증 |
| No f clone | target z를 decoder에 직접 사용 | f 복제 필요성 검증 |

Target z를 서버가 직접 관찰하므로 피해 표본 복원 자체에는 f 복제가 반드시
필요하지 않다. SurrogateF는 공개 이미지에서 피해자와 정렬된 보조 latent를
생성하여 decoder 학습 데이터를 확장할 때 의미가 있다.

### 9.3 조건 변화

- Cut: early, middle, late
- h 모델 깊이
- 클래스 개수
- Batch size
- 학습 epoch
- Auxiliary 데이터 크기
- 라벨 추론 정확도
- Hard label과 soft label
- 여러 random seed

모든 결과는 독립 실행 폴더를 사용하고 최소 5개 seed의 평균과 표준편차를
보고한다.

## 10. 평가 지표

### 10.1 Pixel·구조 유사성

- PSNR: 높을수록 좋음
- SSIM: 높을수록 구조가 유사함
- L1 또는 MSE: 낮을수록 좋음

### 10.2 지각·의미 유사성

- LPIPS: 낮을수록 시각적으로 유사함
- 고정 분류기의 class consistency accuracy
- 원본과 복원 이미지의 feature cosine similarity

### 10.3 라벨 오류 전파

- 라벨 추론 성공 표본의 복원 품질
- 라벨 추론 실패 표본의 복원 품질
- 실제 라벨과 추론 라벨 decoder 간 품질 차이
- Hard label과 soft label의 품질 차이

전체 평균만 보고하면 18%의 라벨 오류가 어떤 영향을 주는지 숨겨질 수 있으므로
성공·실패 표본을 분리해 보고한다.

## 11. 성공 기준

첫 proof-of-concept는 다음 조건을 만족하면 성공으로 본다.

1. B5의 LPIPS 또는 SSIM이 B1보다 일관되게 개선된다.
2. B7이 B5보다 개선되어 gradient가 라벨 이외의 추가 정보를 제공함을 보인다.
3. 추론 라벨 조건에서도 실제 클래스 특징이 유지된다.
4. Victim test 원본을 학습하지 않고도 보조 데이터에서 학습한 decoder가
   일반화한다.
5. 여러 seed에서 동일한 경향이 재현된다.

복원 이미지가 원본과 완전히 동일해야 성공하는 것은 아니다. 이 연구의 목표는
관찰된 중간 정보와 추론 라벨이 원본의 시각적·의미적 특징을 얼마나 노출하는지
측정하는 것이다.

## 12. 권장 코드 구조

```text
src/decoder/
├─ data/                 # 관측 수집, 라벨 추론, 공격자/평가자 데이터 분리
├─ surrogate_models/     # f-hat, h-hat
├─ models/               # signal encoder, label-conditioned decoder
├─ losses/               # L1 + SSIM
├─ training/             # decoder 및 surrogate trainer
├─ evaluation/           # PSNR, SSIM, MAE, 비교 이미지
└─ pipeline/             # 전체 실험 CLI
```

각 책임을 분리하여 SurrogateF, SurrogateH, decoder, 데이터 수집, 평가지표가
서로 독립적으로 교체될 수 있도록 한다.

구현된 명령과 산출물은 [복원 실험 가이드](decoder_reconstruction_experiment.md)에 정리한다.

## 13. 결과 저장 구조

```text
workspace/results/runs/reconstruction_idea1_01/
├─ observations/
│  ├─ auxiliary/
│  └─ victim/
├─ surrogate_checkpoints/
│  ├─ surrogate_f.pt
│  └─ surrogate_h.pt
├─ decoder_checkpoints/
│  └─ best_decoder.pt
├─ reconstructions/
│  ├─ sample_xxx_original.png
│  ├─ sample_xxx_reconstructed.png
│  └─ sample_xxx_comparison.png
└─ reports/
   ├─ reconstruction_metrics.csv
   ├─ reconstruction_summary.json
   └─ reconstruction_report.md
```

## 14. 구현 순서

### Phase 1: 가장 작은 검증

1. Middle cut 하나만 사용한다.
2. 실제 f/g/h로 auxiliary observation을 만든다.
3. R(z, y_true)와 R(z)를 학습한다.
4. 라벨 조건이 복원에 도움이 되는지 확인한다.

### Phase 2: 추론 라벨 적용

1. R(z, ŷ)를 평가한다.
2. 라벨 추론 성공·실패 표본을 분리한다.
3. Hard label과 soft label을 비교한다.

### Phase 3: 모든 관찰값 결합

1. GradientEncoder와 ServerOutputEncoder를 추가한다.
2. R(z, u, gradient, ŷ)를 학습한다.
3. 입력별 ablation을 수행한다.

### Phase 4: f/h 복제

1. SurrogateF를 latent distillation으로 학습한다.
2. SurrogateH를 gradient matching으로 학습한다.
3. 실제 모델 observation과 surrogate observation으로 학습한 decoder를 비교한다.

### Phase 5: 일반화 실험

1. Early, middle, late cut을 비교한다.
2. 여러 h depth와 batch size를 비교한다.
3. 여러 seed에서 평균과 표준편차를 계산한다.

## 15. 해석 시 주의사항

- 추론 라벨만으로 생성한 이미지는 개별 원본 복원이 아니라 클래스 prototype이다.
- 원본과 유사한 개별 이미지를 만들려면 target z 같은 표본별 관찰값이 필요하다.
- Anchor 한 장은 cluster 의미 매핑에 충분할 수 있지만 decoder 학습에는 부족하다.
- SurrogateF가 피해자 latent 공간과 정렬되지 않으면 decoder가 target z를 해석할
  수 없다.
- SurrogateH의 gradient가 피해자 gradient와 맞지 않으면 보조 데이터로 학습한
  decoder가 일반화하지 못할 수 있다.
- 동일 출력 폴더를 재사용하면 transcript가 누적될 수 있으므로 매 실행마다 새
  run 디렉터리를 사용한다.
- 결과는 exact recovery가 아니라 approximate reconstruction으로 표현한다.
- 모든 실험은 승인된 공개·보조 데이터와 통제된 연구 환경에서 수행한다.

## 16. 최종 연구 주장

이 실험이 성공하면 다음과 같은 주장을 검증할 수 있다.

> U-shaped Split Learning에서 서버가 관찰하는 smashed data와 반환 gradient에는
> 원본의 표본 정보와 라벨 정보가 함께 남을 수 있다. Gradient로 추론한 라벨을
> 별도의 label-conditioned decoder에 제공하면, 라벨 없이 복원할 때보다 원본의
> 클래스와 시각적 특징을 더 잘 보존하는 근사 이미지를 생성할 수 있다.
