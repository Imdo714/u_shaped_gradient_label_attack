# Shared-server gradient transcript inversion

## 연구 개요

이 연구는 하나의 공유 서버와 여러 클라이언트가 U-shaped Split Learning을 수행하는 1:N 환경에서 클라이언트 경계에 노출되는 중간값과 gradient가 라벨 및 입력 이미지 정보를 얼마나 누출하는지 검증한다. 주 공격 조건은 서버가 클라이언트로 반환하는 `u`와 `dL/dz`를 이용해 라벨을 추론하고 입력 이미지를 복원하는 것이다.

이 연구에서 1:N 구조가 중요한 이유는 여러 클라이언트가 동일한 서버 중간 모델 `g`를 공유할 수 있기 때문이다. 하나의 클라이언트 또는 공유 통신 경계에서 학습한 공격 모델이 같은 `g`, cut layer, 전처리 및 프로토콜을 사용하는 다른 클라이언트에도 전이될 가능성이 있다. 다만 서버가 하나라는 사실만으로 공격자가 모델 구조와 가중치를 자동으로 알게 되는 것은 아니다. 모델 구조와 가중치의 획득은 서버 checkpoint, 모델 저장소, 학습 오케스트레이터 또는 배포 파이프라인이 별도로 유출된 white-box 조건으로 정의해야 한다.

## U-shaped Split Learning의 통신 과정

클라이언트는 입력 이미지 `x`를 전단 모델 `f`에 통과시켜 다음 중간 표현을 생성한다.

$$
z=f(x)
$$

클라이언트는 `z`를 서버에 전달하고, 서버는 공유 중간 모델 `g`를 사용해 다음 값을 계산하여 클라이언트에 반환한다.

$$
u=g(z)
$$

클라이언트는 후단 모델 `h`를 이용해 예측값을 만들고 자신이 보유한 라벨 `y`로 loss를 계산한다. 이어서 클라이언트는 다음 gradient를 서버에 전달한다.

$$
\delta_u=\frac{\partial L}{\partial u}
$$

서버는 `g`를 역전파하여 다음 gradient를 클라이언트에 반환한다.

$$
\delta_z=\frac{\partial L}{\partial z}
$$

각 통신값의 방향은 다음과 같다.

| 값 | 방향 | 의미 |
|---|---|---|
| `z=f(x)` | 클라이언트 → 서버 | 입력 이미지에서 생성된 smashed representation |
| `u=g(z)` | 서버 → 클라이언트 | 공유 서버 모델의 출력 특징 |
| `dL/du` | 클라이언트 → 서버 | 라벨과 후단 모델로 계산된 loss gradient |
| `dL/dz` | 서버 → 클라이언트 | 서버 모델을 역전파한 gradient |

따라서 서버에서 클라이언트로 전달되는 downlink만 관찰하는 공격자는 `u`와 `dL/dz`를 수집한다. 악성 클라이언트 또는 복호화 이후의 양방향 애플리케이션 relay는 `z`, `u`, `dL/du`, `dL/dz`를 모두 관찰할 수 있다. 암호화된 TLS 패킷만 관찰하는 외부 공격자가 tensor 값을 직접 읽을 수 있다고 가정하지 않는다.

## Gradient에 포함되는 서버의 국소 정보

서버 모델의 Jacobian을 다음과 같이 정의한다.

$$
J_g(z)=\frac{\partial g(z)}{\partial z}
$$

연쇄법칙에 따라 서버가 반환하는 gradient는 다음 관계를 만족한다.

$$
\frac{\partial L}{\partial z}
=J_g(z)^T\frac{\partial L}{\partial u}
$$

따라서 `dL/dz`는 서버 모델의 현재 입력 지점에 대한 국소적 민감도와 라벨 의존적인 loss 정보를 함께 반영한다. 이는 서버 가중치 자체가 직접 전송된다는 뜻이 아니다. 한 번의 `(dL/du, dL/dz)` 관찰은 전체 Jacobian이 아니라 `J_g(z)^T(dL/du)`라는 하나의 vector-Jacobian product만 제공한다. 여러 입력과 다양한 gradient 방향을 반복 관찰하면 서버의 국소 동작을 근사하는 데 도움이 될 수 있지만, 구조와 가중치의 정확한 복구가 보장되지는 않는다.

논문에서는 “gradient에 서버 가중치가 포함된다”라고 표현하지 않는다. 대신 “gradient가 서버 모델의 Jacobian에 의해 결정되는 국소적 1차 민감도를 노출한다”라고 표현한다.

## 위협 모델

주 실험은 공격자가 피해 모델의 checkpoint를 공격 디코더에 직접 불러오지 않는 black-box transcript 조건으로 정의한다. 공격자는 공개 이미지와 공개 라벨을 자신이 제어하는 보조 클라이언트를 통해 동일한 피해 Split Learning 서비스에 입력하고, 공개 원본과 대응하는 `(u, dL/dz)`를 수집한다. 이 대응 쌍으로 공격 디코더를 학습한 후 실제 피해 클라이언트의 원본과 라벨에는 접근하지 않고 관찰된 중간값만으로 복원을 수행한다.

확장 실험은 학습 서버 또는 모델 저장소가 유출되어 공격자가 구조와 가중치를 알고 있는 white-box 조건으로 정의한다. 공격자가 전체 `f-g-h` 구조와 가중치를 확보했다면 공개 데이터로 필요한 중간값과 gradient를 오프라인에서 생성할 수 있다. 서버의 `g`만 확보했다면 공개 이미지에서 `z=f(x)`를 생성하고 `h`와 라벨을 이용해 `dL/du`를 계산할 수 없으므로, 공격자가 제어하는 보조 클라이언트 또는 피해 서비스에 대한 질의 권한이 추가로 필요하다.

클라이언트 전체 메모리에 접근해 원본 `x`를 직접 읽을 수 있는 공격은 복원 문제가 불필요해지므로 이 연구의 범위에서 제외한다. 공격자의 배포 단계 권한은 클라이언트 통신 모듈 또는 애플리케이션 relay에서 복호화된 중간 tensor를 관찰할 수 있지만 원본 입력과 라벨을 직접 읽을 수 없는 수준으로 제한한다.

## 공격 모델

라벨 추론기는 정규화된 `dL/dz` 특징만 사용해 다음 값을 추정한다.

$$
\hat{y}=C(dL/dz)
$$

라벨 추론기를 지도학습하려면 공개 라벨이 붙은 auxiliary transcript가 필요하다. 비지도 clustering만 사용할 경우 cluster 번호와 실제 라벨 사이에 permutation ambiguity가 남으므로 소수의 label anchor 또는 공개 라벨을 이용해 cluster를 실제 클래스에 대응시켜야 한다.

주 복원 디코더는 서버 downlink에서 관찰한 신호와 추론 라벨을 이용한다.

$$
\hat{x}=D(u,dL/dz,\hat{y})
$$

확장 관찰 조건에서는 client-to-server 방향의 `z`까지 추가해 다음 복원을 수행한다.

$$
\hat{x}=D(z,u,dL/dz,\hat{y})
$$

이 두 조건은 공격자가 관찰하는 정보가 다르므로 같은 결과로 합쳐서 보고하지 않는다. `z + u + dL/dz` 결과는 `u + dL/dz`보다 강한 공격자 조건의 상한으로 해석한다.

## 연구 가설

첫 번째 가설은 `dL/dz`가 라벨 의존적인 통계적 특징을 가지므로 공개 auxiliary transcript로 학습한 분류기가 unseen 피해 transcript의 라벨을 우연 수준보다 높게 추론할 수 있다는 것이다.

두 번째 가설은 `u`가 입력의 의미적·공간적 특징을 보존하고 `dL/dz`와 추론 라벨을 결합하면 `u`만 사용한 조건보다 복원 성능이 향상된다는 것이다.

세 번째 가설은 동일한 공유 서버 `g`와 호환되는 client-side 모델을 사용하는 1:N 환경에서 한 클라이언트의 공개 auxiliary transcript로 학습한 공격 모델이 다른 클라이언트의 transcript에도 전이될 수 있다는 것이다. 클라이언트별 `f`와 `h` 가중치가 서로 다르게 업데이트되는 경우에는 중간값 분포가 달라질 수 있으므로 client 간 전이 성능을 별도로 측정한다.

네 번째 가설은 구조와 가중치를 알고 있는 white-box 조건이 checkpoint를 직접 사용하지 않는 black-box transcript 조건보다 높은 복원 성능을 보인다는 것이다. 이 조건은 주 공격의 현실성을 대신하지 않고 성능 상한을 나타내는 ablation으로 사용한다.

## 실험 조건

| 조건 | 디코더 입력 | 목적 |
|---|---|---|
| A | `u` | 서버 출력만으로 가능한 기본 복원 |
| B | `dL/dz` | gradient 단독 라벨 및 복원 정보 측정 |
| C | `u + dL/dz` | 서버 downlink만 관찰하는 주 공격 |
| D | `z + u + dL/dz` | 양방향 relay의 확장 공격 |
| E | `z + u + dL/du + dL/dz` | 악성 클라이언트가 모든 transcript를 보는 상한 |
| F | 전체 구조·가중치 사용 | white-box 모델 유출 상한 |

모든 조건은 동일한 공개 train, validation, 피해 holdout, 이미지 크기, seed 및 평가 지표를 사용한다. 모델별 tensor 채널과 크기가 다르면 signal adapter를 대상별로 학습한다. 공격 절차는 여러 Split Learning 구조에 적용할 수 있지만 학습된 디코더 하나가 서로 다른 구조, cut layer 및 가중치 버전에 그대로 적용된다고 주장하지 않는다.

주 실험 데이터는 공개 원본 450장에서 만든 10,000개 보조 학습 표본, 별도 validation 75장과 학습에 사용하지 않은 피해 holdout 100장으로 구성한다. 이 10,000개는 서로 독립적인 원본 이미지 수가 아니므로 논문에는 원본 수와 augmentation 이후의 학습 표본 수를 함께 기재한다.

## 현재 결과의 해석

`rpc_u_grad_z_128_baseline`은 공개 학습 이미지 450장과 holdout 20장을 사용한 `u + dL/dz` 조건이다. 평균 PSNR은 약 21.15 dB, SSIM은 약 0.905이며 라벨 정확도는 100%였다. 이 결과는 그림의 downlink 공격과 신호 조건이 일치하지만 평가 표본이 20장으로 작다.

`rpc_z_u_grad_z_128_aux10k_holdout100`은 10,000개 보조 학습 표본과 holdout 100장을 사용했으며 평균 PSNR은 약 25.82 dB, SSIM은 약 0.962, 라벨 정확도는 100%였다. 이 실험의 라벨 classifier는 `dL/dz` 특징만 사용하므로 현재 대상에서 gradient의 라벨 의존성을 뒷받침한다. 그러나 이미지 복원에는 `z + u + dL/dz`가 사용되었으므로 이 성능을 `u + dL/dz`만의 복원 증거로 제시하면 안 된다.

현재 가설을 직접 검증하기 위해 가장 필요한 실험은 10,000개 보조 학습 표본으로 `u + dL/dz` 디코더를 학습하고 동일한 holdout 100장에 평가하는 것이다. 이 결과를 동일 데이터의 `u only`와 `z + u + dL/dz` 조건에 비교해야 각 신호의 기여도를 분리할 수 있다.

## `u + dL/dz` 10,000표본 주 실험

다음 명령은 `z`를 수집하지 않고 공개 보조 데이터의 `u`와 `dL/dz`만으로 디코더와 라벨 head를 학습한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.run_rpc_experiment `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --client-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --data workspace\data\dataset_aux_10k `
  --output workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_aux10k `
  --aux-train-split train `
  --aux-validation-split val `
  --victim-split new_holdout `
  --holdout-count 20 `
  --holdout-start-index 0 `
  --holdout-labels cat dog `
  --image-size 128 `
  --decoder-architecture baseline_bilinear `
  --epochs 50 `
  --batch-size 8 `
  --use-label-head `
  --signal-spatial-size 16 `
  --signal-channels 64 `
  --decoder-base-channels 256 `
  --decoder-min-channels 32 `
  --refinement-blocks 1 `
  --edge-weight 0.1 `
  --perceptual-weight 0.1 `
  --laplacian-weight 0.25 `
  --early-stopping-patience 8 `
  --early-stopping-min-delta 0.0001 `
  --seed 42 `
  --process-timeout-seconds 86400 `
  --device cuda
```

학습이 완료되면 다음 명령으로 디코더를 다시 학습하지 않고 새로운 holdout 100장의 `u`와 `dL/dz`를 수집해 평가한다. 이 명령에는 `--capture-z`를 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.evaluate_rpc_holdout `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --client-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --decoder-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_aux10k\attack_training\checkpoints\client_received_decoder_best.pt `
  --data workspace\data\dataset `
  --output workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_aux10k_holdout100 `
  --victim-split new_holdout `
  --holdout-count 100 `
  --holdout-start-index 0 `
  --holdout-labels cat dog `
  --image-size 128 `
  --batch-size 8 `
  --max-grid-images 100 `
  --process-timeout-seconds 14400 `
  --device cuda
```

## 평가 지표와 성공 기준

라벨 누출은 정확도뿐 아니라 클래스별 precision, recall, F1-score와 confusion matrix로 평가한다. 현재 클래스가 균형이므로 random baseline은 cat/dog 평가에서 50%이다. 모델 구조 또는 데이터 분포가 바뀐 경우에도 정확도가 유지되는지 별도로 확인한다.

이미지 복원은 MSE, MAE, PSNR 및 SSIM을 보고한다. `u + dL/dz` 조건이 `u only`보다 평균 PSNR과 SSIM이 높고 MSE와 MAE가 낮아야 gradient가 복원에 추가로 기여했다고 판단한다. 평균값뿐 아니라 표본별 승률과 bootstrap confidence interval을 함께 보고하며, 동일 holdout 표본에 대한 paired 비교를 수행한다.

복원 이미지에서 객체 종류를 알아볼 수 있다는 사실은 privacy leakage의 정성적 증거가 될 수 있지만, 그것만으로 픽셀 수준 복원 성공을 주장하지 않는다. 객체·속성 누출과 픽셀 충실도 복원을 구분하고, 각 주장에 맞는 정량 지표를 사용한다.

## 배포 환경에 대한 제한

`dL/dz`는 loss와 역전파가 수행되는 학습 단계에서만 생성된다. 배포 후에도 지속 학습, 온라인 학습 또는 fine-tuning을 수행하는 환경에서는 `u + dL/dz` 공격이 가능하지만, 추론만 수행하는 환경에서는 일반적으로 `dL/dz`가 존재하지 않는다. 추론 전용 환경은 `u only` 또는 관찰 가능한 forward activation만 사용하는 별도 공격으로 평가한다.

모델 가중치가 학습 이후 변경되거나 cut layer와 전처리가 달라지면 공개 데이터로 학습한 디코더의 입력 분포가 변한다. 따라서 모델 버전별 성능, client 간 전이 성능과 가중치 drift에 대한 강건성 실험이 필요하다.

## 논문에서 사용할 수 있는 핵심 표현

본 연구는 U-shaped Split Learning의 서버-클라이언트 경계에서 관찰되는 `u`와 `dL/dz`가 라벨과 입력 이미지에 관한 정보를 누출하는지 분석한다. `dL/dz`는 서버 가중치를 직접 공개하지 않지만 서버 모델의 Jacobian에 의해 결정되는 vector-Jacobian product이므로 현재 입력에 대한 국소적 민감도와 라벨 의존적인 loss 정보를 반영한다. 공격자는 공개 auxiliary 데이터로 라벨 추론기와 복원 디코더를 학습하고, 피해 클라이언트의 원본과 라벨에 접근하지 않은 상태에서 관찰된 transcript만으로 라벨과 입력을 추정한다. 공유 서버가 여러 클라이언트에 공통으로 사용되는 경우 동일 공격 모델의 client 간 전이 가능성과 공격 영향 범위를 함께 평가한다.

## 주장하지 않는 내용

이 연구는 한 번의 gradient로 서버의 전체 Jacobian, 구조 또는 정확한 가중치를 복구할 수 있다고 주장하지 않는다. 또한 하나의 디코더가 재학습 없이 모든 Split Learning 구조에 적용된다고 주장하지 않는다. `z`를 포함한 확장 공격의 높은 복원 성능을 `u + dL/dz`만의 결과로 표현하지 않으며, 추론 전용 배포 환경에서 학습 gradient를 관찰할 수 있다고 가정하지 않는다.
