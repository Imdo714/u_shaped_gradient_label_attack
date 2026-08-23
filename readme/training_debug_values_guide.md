# U자형 분할 학습 중간값 및 그래디언트 로그 해설

이 문서는 `src.split_learning.train`의 상세 로그에 출력되는 실제 학습 샘플의 `x`, `z`,
`u`, logits, loss와 gradient가 무엇을 의미하는지 설명합니다.

## 실행 명령

프로젝트 루트에서 다음과 같이 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m src.split_learning.train --data ./data --epochs 1 --batch-size 1 --cut-config middle --debug-samples 1 --debug-values 10
```

- `--debug-samples 1`: 첫 epoch의 샘플 1개에 대해 상세 로그를 출력합니다.
- `--debug-values 10`: 각 텐서를 펼친 뒤 앞의 값 10개를 출력합니다.
- `--debug-samples 0`: 상세 로그를 끕니다.

## 전체 계산 흐름

```text
클라이언트 입력 x
    ↓ ClientFront f
z_local
    ↓ detach / 전송
z_wire                       서버가 받은 smashed data
    ↓ ServerMiddle g
u_server
    ↓ detach / 전송
u_wire                       ClientTail이 받은 서버 출력
    ↓ ClientTail h
logits
    ↓ CrossEntropyLoss(logits, y)
loss
    ↓ backward
grad_h_to_g = dL/du          서버가 관찰하는 공격 대상 gradient
    ↓ ServerMiddle backward
grad_g_to_f = dL/dz          서버가 ClientFront로 보내는 gradient
    ↓ ClientFront backward
f, g, h 파라미터 gradient 계산
```

## 로그 첫 줄

예시:

```text
epoch=1 batch_id=0 sample_id=sample_6456cce6839d787999e3
```

- `epoch=1`: 첫 번째 전체 데이터 학습 반복입니다.
- `batch_id=0`: 현재 epoch의 첫 번째 batch입니다.
- `sample_id`: 클래스 폴더명이 노출되지 않도록 파일 경로를 SHA-256 기반의
  불투명 ID로 바꾼 값입니다. ID만 보고 CAT/DOG를 알 수 없습니다.
- 현재 `batch_size=1`이므로 이 batch에는 이미지 한 장만 들어 있습니다.

## 모든 텐서에 공통으로 표시되는 값

```text
shape=(...)
requires_grad=True/False
min=...
max=...
mean=...
norm=...
first_10=[...]
```

| 항목 | 의미 |
|---|---|
| `shape` | batch, channel, height, width 또는 특징 차원 |
| `requires_grad` | 이 텐서에 대해 autograd가 gradient 경로를 추적하는지 여부 |
| `min`, `max` | 텐서 전체의 최솟값과 최댓값 |
| `mean` | 텐서 전체 원소의 평균 |
| `norm` | 텐서 전체의 L2 norm, 즉 `sqrt(sum(value²))` |
| `first_10` | 텐서를 1차원으로 펼친 뒤 앞의 10개 값 |

`first_10`은 텐서 전체를 대표하는 통계가 아닙니다. 실행 경로를 확인하기
위한 일부 값이며, 전체적인 크기는 `min/max/mean/norm`으로 판단합니다.

## 1. 클라이언트 입력 `x`

예시:

```text
x shape=(1, 3, 64, 64)
requires_grad=False
min=-1.861033 max=2.605142 mean=-0.311425 norm=124.699852
```

shape의 의미는 다음과 같습니다.

```text
1  : batch에 이미지 1장
3  : RGB 채널
64 : 높이
64 : 너비
```

여기서 `x`는 JPEG의 0~255 원본 픽셀을 그대로 출력한 값이 아닙니다.
이미지를 64x64로 바꾸고 `ToTensor()`와 ImageNet mean/std 정규화를 적용한
뒤의 클라이언트 입력 텐서입니다. 따라서 음수와 1보다 큰 값이 정상입니다.

입력 이미지 자체에 대한 gradient는 필요하지 않으므로
`requires_grad=False`입니다. 모델 파라미터의 gradient 계산에는 문제가 없습니다.

## 2. ClientFront 출력 `z_local`

예시:

```text
z_local shape=(1, 32, 16, 16)
min=0.000000 max=5.368872 mean=0.728527 norm=94.779488
```

`z_local = f(x)`이며 ClientFront가 계산한 첫 번째 중간 활성값입니다. 이것이
클라이언트 내부의 smashed data 원본입니다.

`middle` cut에서는 두 개의 convolution/pooling block을 통과하므로:

```text
[1, 3, 64, 64] -> [1, 16, 32, 32] -> [1, 32, 16, 16]
```

으로 변합니다. 최솟값이 0인 것은 ReLU가 음수를 0으로 바꾸기 때문입니다.

## 3. 서버가 받은 smashed data `z_wire`

```python
z_wire = z_local.detach().requires_grad_(True)
```

예시에서는 `z_local`과 `z_wire`의 숫자, min/max/mean/norm이 완전히 같습니다.
이는 정상입니다. `detach()`는 텐서 값을 바꾸지 않고 기존 autograd 그래프와의
연결만 끊습니다.

```text
z_local: 클라이언트 내부 텐서이며 f의 계산 그래프에 연결
z_wire : 서버가 수신한 동일 값의 새 leaf tensor
```

`z_wire.requires_grad=True`로 설정하는 이유는 서버가 역전파한 뒤
`z_wire.grad`에서 `dL/dz`를 얻기 위해서입니다.

## 4. ServerMiddle 출력 `u_server`

예시:

```text
u_server shape=(1, 64, 8, 8)
min=0.000000 max=4.219831 mean=0.838283 norm=71.313568
```

서버가 다음 계산을 수행한 결과입니다.

```python
u_server = g(z_wire)
```

`middle` cut의 ServerMiddle convolution/pooling block이 채널을 32에서 64로
늘리고 공간 크기를 16x16에서 8x8로 줄입니다.

## 5. ClientTail이 받은 `u_wire`

```python
u_wire = u_server.detach().requires_grad_(True)
```

`u_server`와 `u_wire`도 값은 완전히 같고 autograd 연결만 다릅니다.

```text
u_server: ServerMiddle g의 계산 그래프에 연결
u_wire  : ClientTail h가 수신한 새 leaf tensor
```

`u_wire.requires_grad=True`로 설정해야 `loss.backward()` 이후
`u_wire.grad = dL/du`를 읽을 수 있습니다.

## 6. ClientTail의 logits와 확률

예시:

```text
logits = [-0.02389283, -0.05519690]
softmax probability = [0.50782537, 0.49217463]
true label = 0 (CAT)
predicted label = 0 (CAT)
```

logits는 확률이 아니라 CAT과 DOG의 원시 점수입니다.

```text
logits[0] = CAT 점수
logits[1] = DOG 점수
```

두 점수가 모두 음수여도 문제가 없습니다. softmax는 절댓값이 아니라 두 점수의
상대적인 차이를 사용합니다. CAT logit이 더 크기 때문에 CAT 확률이 약
50.78%, DOG 확률이 약 49.22%가 되었고 예측은 CAT입니다.

훈련 모델은 logits를 그대로 `CrossEntropyLoss`에 전달합니다. 출력에 보이는
softmax 확률은 사람이 이해하기 위한 디버그 표시용이며 손실 입력에는 사용하지
않습니다.

## 7. Cross-entropy loss

예시:

```text
cross entropy loss = 0.67761761
```

batch size 1에서 실제 레이블이 CAT이면 개념적으로 다음과 같습니다.

```text
loss = -log(P(CAT))
     = -log(0.50782537)
     ≈ 0.67761761
```

정답 클래스 확률이 1에 가까워질수록 loss는 0에 가까워집니다.

## 8. 공격 대상 `grad_h_to_g = dL/du`

예시:

```text
shape=(1, 64, 8, 8)
min=-0.001690 max=0.001949 mean=0.000012 norm=0.035693
first_10=[0.00039309, 0.00039309, 0.00010989, ...]
```

ClientTail에서 다음 과정으로 계산됩니다.

```python
loss.backward()
grad_h_to_g = u_wire.grad.detach().clone()
```

수학적으로:

```text
grad_h_to_g = dL/du
```

loss가 실제 레이블을 사용하므로 이 gradient의 방향과 크기는 레이블의 영향을
받습니다. ClientTail은 ServerMiddle 역전파를 위해 이 값을 서버로 보내며,
서버는 레이블 자체 없이 이 텐서를 관찰합니다. 본 프로젝트의 K-means 공격이
평탄화하고 L2 정규화하는 핵심 특징이 바로 이 값입니다.

shape이 `u`와 동일한 이유는 `u`의 각 원소가 loss에 미치는 미분값이 하나씩
필요하기 때문입니다.

## 9. `grad_g_to_f = dL/dz`

예시:

```text
shape=(1, 32, 16, 16)
min=-0.001894 max=0.002193 mean=0.000002 norm=0.043226
```

서버가 `grad_h_to_g`를 받아 다음 계산을 수행한 결과입니다.

```python
u_server.backward(grad_h_to_g)
grad_g_to_f = z_wire.grad.detach().clone()
```

수학적으로:

```text
grad_g_to_f = dL/dz
```

ServerMiddle은 이 값을 ClientFront로 보내고, ClientFront는
`z_local.backward(grad_g_to_f)`로 자신의 파라미터 gradient를 계산합니다.
shape이 `z`와 동일한 이유 역시 `z`의 각 원소에 대한 미분값이 필요하기 때문입니다.

## 10. f, g, h 파라미터 gradient norm

예시:

```text
ClientFront f  : 1.25797153
ServerMiddle g : 1.24646115
ClientTail h   : 4.64231968
```

각 값은 해당 모델의 모든 파라미터 gradient를 하나로 모았을 때의 전체 L2
norm입니다. 세 모델 모두 0보다 크므로 `loss -> h -> g -> f` 역전파가 실제로
도달했다는 것을 확인할 수 있습니다.

이 값을 서로 직접 비교해 “h가 f보다 3.7배 더 학습된다”고 결론 내리면 안 됩니다.
모델마다 파라미터 개수와 레이어 구조가 다르고 Adam optimizer도 gradient를
그대로 빼지 않기 때문입니다. 같은 모델과 설정에서 epoch 또는 샘플에 따른
변화를 비교하는 진단 지표로 사용하는 것이 적절합니다.

## 현재 예시 값 요약

| 단계 | shape | norm 또는 값 | 의미 |
|---|---|---:|---|
| 입력 `x` | `[1,3,64,64]` | 124.699852 | 정규화된 실제 이미지 |
| `z_local` | `[1,32,16,16]` | 94.779488 | ClientFront가 만든 smashed data |
| `z_wire` | `[1,32,16,16]` | 94.779488 | 서버가 받은 동일 값의 detached tensor |
| `u_server` | `[1,64,8,8]` | 71.313568 | ServerMiddle 출력 |
| `u_wire` | `[1,64,8,8]` | 71.313568 | ClientTail이 받은 detached tensor |
| logits | `[1,2]` | `[-0.02389,-0.05520]` | CAT/DOG 원시 점수 |
| 확률 | `[1,2]` | `[50.78%,49.22%]` | 표시용 softmax 결과 |
| loss | scalar | 0.67761761 | 실제 CAT 레이블의 CE loss |
| `dL/du` | `[1,64,8,8]` | 0.035693 | 서버가 관찰하는 공격 gradient |
| `dL/dz` | `[1,32,16,16]` | 0.043226 | 서버가 ClientFront로 보내는 gradient |

## 개인정보 및 공격자 view 주의사항

이 상세 터미널은 코드 흐름을 검증하는 연구자 디버그 view이므로 실제 레이블,
예측 레이블과 확률까지 표시합니다. 정직하지만 호기심 많은 서버 공격자에게
허용되는 정보와는 다릅니다.

서버 공격자 transcript에는 다음 항목만 저장됩니다.

```text
opaque sample_id
epoch, batch_id
z
u
dL/du
dL/dz
```

실제 레이블은 별도의 `evaluator_ground_truth`에 저장되며 K-means 학습에는
사용되지 않습니다.
