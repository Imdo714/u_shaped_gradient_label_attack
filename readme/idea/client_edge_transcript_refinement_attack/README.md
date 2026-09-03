# Client-Received Transcript Reconstruction and Refinement Attack

## 아이디어 요약

이 문서는 U-shaped Split Learning에 참여하는 하나의 클라이언트가 서버로부터 받는 `u`와 `dL/dz`를 외부 관찰자가 수집하고, 두 값을 이용하여 해당 클라이언트의 원본 이미지를 근사 복원하는 공격 가설을 설명한다. 공격자는 특정 함수 `f`의 내부 실행을 직접 관찰하는 것이 아니라, 하나의 클라이언트와 서버 사이의 통신 경계에서 클라이언트로 들어오는 중간 표현과 gradient를 관찰한다.

공격자는 먼저 같은 학습 단계에서 생성된 `u`와 `dL/dz`를 결합하여 coarse decoder에 입력한다. Coarse decoder는 원본 이미지의 객체 종류, 위치, 색, 배경과 대략적인 형태를 복원한다. 이후 공개 auxiliary 데이터만으로 학습한 refiner가 coarse reconstruction을 보정하여 구조적 충실도와 시각적 품질을 높인다. 이 과정은 원본 픽셀을 완벽하게 되찾는다고 주장하는 것이 아니라, 클라이언트가 수신하는 학습 메시지에 원본 이미지와 라벨에 관한 정보가 어느 정도 남아 있는지를 평가하는 것을 목적으로 한다.

## U-shaped Split Learning에서 클라이언트가 받는 정보

U-shaped Split Learning에서는 원본 이미지 `x`가 클라이언트 앞부분 모델인 `f`를 지나 smashed data `z`로 변환되고, `z`가 서버의 중간 모델 `g`로 전달된다. 서버는 `z`를 처리하여 중간 표현 `u`를 만든 뒤 클라이언트 뒷부분 모델에 보낸다. 클라이언트 뒷부분 모델은 `u`로 예측과 loss를 계산하고, 역전파 과정에서 `dL/du`를 서버에 반환한다. 서버는 `dL/du`를 이용해 자신의 모델을 역전파한 뒤 `dL/dz`를 클라이언트 앞부분 모델에 돌려준다.

클라이언트의 앞부분 모델과 뒷부분 모델이 동일한 물리적 장치 또는 동일한 관리 영역에 존재한다면, 그 클라이언트가 서버로부터 수신하는 대표적인 학습 메시지는 forward 단계의 `u`와 backward 단계의 `dL/dz`이다. 따라서 본 가설은 클라이언트가 서버에 보내는 `z`나 `dL/du`를 수집하는 공격이 아니라, 동일한 하나의 클라이언트가 서버로부터 받는 `u`와 `dL/dz`를 수집하는 공격으로 정의한다.

`u`는 서버 모델을 통과한 중간 표현이므로 입력 이미지의 고수준 의미와 형태 정보를 포함할 가능성이 있다. `dL/dz`는 loss와 실제 라벨의 영향을 받은 역전파 정보가 서버 모델을 거쳐 첫 번째 cut으로 전달된 값이므로, 입력의 특징과 클래스에 관한 정보가 남아 있을 가능성이 있다. 두 신호는 서로 다른 정보를 제공할 수 있으므로 함께 사용할 때 각 신호를 단독으로 사용할 때보다 복원 성능이 높아지는지를 검증한다.

## 공격자 가정

공격자는 임의의 인터넷 사용자가 서버에 접속하거나 서버로 가장하는 능동 공격자로 가정하지 않는다. 대신 하나의 클라이언트와 서버 사이에서 서버가 클라이언트로 보내는 통신을 수동적으로 관찰할 수 있는 외부 관찰자를 가정한다. 이 관찰자는 클라이언트 측의 침해된 gateway나 relay, 복호화 이후 평문 텐서가 전달되는 통신 계층, 또는 허가된 연구 환경의 passive proxy에 위치할 수 있다.

공격자는 `u`와 `dL/dz`를 관찰할 수 있지만 피해 클라이언트의 원본 이미지, 실제 라벨, 로컬 저장소 및 모델 내부 상태에는 직접 접근하지 않는다. 또한 통신 메시지를 변경하거나 악성 텐서를 삽입하지 않고, 서버 또는 클라이언트로 가장하지 않으며, 정상 학습 과정에 영향을 주지 않는 수동 관찰만 수행한다.

클라이언트와 서버가 다대일 관계라는 사실만으로 공격자의 신원이 자동으로 숨겨지는 것은 아니다. 네트워크 로그, 인증 정보, gateway 기록 및 endpoint telemetry에 따라 관찰 행위가 추적될 수 있다. 따라서 본 가설은 공격자가 반드시 익명이라는 전제를 두지 않고, 관찰 위치와 운영 환경에 따라 직접적인 행위자 식별이 어려울 수 있다는 정도로만 설명한다. 탐지 회피나 신원 은폐 방법은 연구 범위에 포함하지 않는다.

이 가설이 성립하려면 공격자가 `u`와 `dL/dz`의 실제 텐서 값을 볼 수 있어야 한다. TLS나 mTLS가 올바르게 적용된 외부 네트워크에서는 일반적인 패킷 캡처만으로 암호화된 메시지의 평문 텐서를 읽을 수 없다. 따라서 실험에서는 암호화되지 않은 연구용 통신, 클라이언트 측 복호화 이후의 통신 인터페이스, TLS를 적법하게 종료하는 연구용 proxy 중 하나를 명시적으로 가정해야 한다. 네트워크에서 텐서를 관찰할 수 있는 조건과 관찰된 텐서가 일으키는 privacy leakage는 서로 구분하여 평가해야 한다.

## 동일 클라이언트와 학습 단계의 연결

`u`와 `dL/dz`는 모두 서버에서 동일 클라이언트로 전달되지만 동시에 도착하지는 않는다. `u`는 forward 단계에서 먼저 전달되고, `dL/dz`는 클라이언트가 loss를 계산해 `dL/du`를 반환한 이후 backward 단계에서 전달된다. 따라서 복원 입력을 만들 때에는 같은 클라이언트, 같은 학습 round, 같은 batch 및 같은 sample에서 생성된 두 값을 정확히 연결해야 한다.

연구용 collector는 client session, training round, batch, sample offset 및 request를 구분할 수 있는 불투명한 식별자를 이용하여 두 메시지를 연결해야 한다. 실제 사용자 ID나 원본 파일 경로는 저장하지 않고, 재식별이 어려운 transcript 식별자를 사용한다. 배치 크기가 1보다 큰 경우에는 `u`와 `dL/dz`의 첫 번째 차원을 동일한 sample 순서로 분리하거나, 배치 내부 sample offset을 함께 기록해야 한다.

정확한 연결이 중요한 이유는 서로 다른 클라이언트나 서로 다른 학습 단계의 값을 결합하면 하나의 원본 이미지에 대응하지 않는 입력이 만들어지기 때문이다. 정확하게 연결한 결과와 동일 클래스 안에서 섞은 결과, 전체 표본 사이에서 무작위로 섞은 결과를 비교하면 decoder가 개별 transcript를 복원하는지 아니면 클래스 평균에 가까운 이미지를 생성하는지 확인할 수 있다.

## 복원 모델

공격 모델은 `u`와 `dL/dz`를 서로 다른 encoder로 처리한 뒤 두 feature를 결합하는 구조를 사용한다. 두 텐서는 서로 다른 cut에서 생성되므로 shape과 의미가 다를 수 있으며, 원본 텐서를 직접 더하거나 단순히 같은 feature로 취급해서는 안 된다. `u` encoder는 서버가 만든 고수준 표현을 처리하고, gradient encoder는 정규화한 `dL/dz`의 방향과 분포를 처리한다. 두 encoder의 출력과 공격자가 추론한 soft label condition을 결합한 뒤 image decoder가 coarse reconstruction을 생성한다.

라벨 추론도 공격자가 실제로 관찰할 수 있는 `dL/dz` 또는 `u`와 `dL/dz`의 결합 정보만 사용해야 한다. 현재 구현의 label predictor가 `dL/du`에서 만든 label condition을 사용한다면, 그 결과는 본 위협 모델의 순수한 실험으로 볼 수 없다. 본 가설을 엄격하게 검증하려면 `dL/dz` 전용 라벨 추론기나 `u`와 `dL/dz`를 함께 사용하는 라벨 추론기를 별도로 구성해야 하며, 피해 클라이언트의 실제 라벨은 평가 이외의 공격 과정에 사용하면 안 된다.

Coarse reconstruction 이후에는 refiner를 적용한다. 첫 번째 refiner는 coarse image만 입력받아 공개 데이터에서 학습한 일반적인 이미지 prior로 노이즈와 경계를 보정한다. 두 번째 refiner는 coarse image와 함께 인코딩된 `u`, 인코딩된 `dL/dz` 및 추론된 soft label을 조건으로 사용한다. 두 방식을 비교하면 단순한 이미지 후처리 효과와 관찰 transcript가 세부 복원에 추가로 기여한 효과를 구분할 수 있다.

Refiner가 만든 이미지가 더 선명하게 보인다고 해서 새로 생성된 털, 눈, 무늬 또는 배경 세부가 실제 원본에 존재했다고 단정할 수는 없다. Refiner는 공개 데이터에서 학습한 시각적 prior를 이용해 그럴듯한 세부를 만들어낼 수 있으므로, 시각적 품질 향상과 원본 정보 복구를 구분해야 한다. 원본과의 픽셀 오차, 구조적 유사도, feature 유사도 및 error map이 함께 개선될 때에만 원본 충실도가 높아졌다고 해석한다.

## 데이터 수집과 학습 분리

공격자의 decoder와 refiner는 공개 auxiliary 이미지로 학습한다. 공개 이미지를 피해 모델과 동일하거나 호환되는 Split Learning 구조에 통과시켜 `u`와 `dL/dz`를 생성하고, 공개 원본과 transcript의 대응 관계를 이용해 공격 모델을 학습한다. 이 auxiliary 학습 과정은 피해 클라이언트의 holdout 원본이나 실제 라벨을 사용하지 않는다.

피해 holdout은 decoder 학습, label predictor 구성 및 refiner 학습에서 모두 제외하고 마지막 평가에서만 사용한다. 공격자 record에는 동일 transcript 식별자로 연결된 `u`와 `dL/dz` 및 관찰 가능한 최소 메타데이터만 저장한다. `z`, `dL/du`, 피해 원본 및 피해 실제 라벨이 공격자 record에 포함되지 않았는지는 자동 검사와 manifest로 확인해야 한다. 평가용 원본과 실제 라벨은 공격자 입력과 분리된 evaluator 전용 경로에서만 읽어야 한다.

## 실험 설계

핵심 비교에서는 `u`만 사용한 조건, `dL/dz`만 사용한 조건 및 `u`와 `dL/dz`를 함께 사용한 조건을 같은 holdout에서 평가한다. 결합 조건에서는 label condition을 제공하지 않는 경우, 관찰 가능한 `dL/dz`에서 추론한 label을 제공하는 경우 및 실제 label을 제공하는 oracle 상한선을 구분한다. Oracle 조건은 실제 공격 성능이 아니라 label 추론 오류가 없을 때 가능한 성능의 상한선으로만 해석한다.

복원 단계에서는 coarse decoder만 사용한 결과와 image-only refiner를 적용한 결과, transcript-conditioned refiner를 적용한 결과를 비교한다. Pairing 실험에서는 동일 client와 동일 step의 정확한 `u`와 `dL/dz`를 연결한 결과를 shuffled pair와 비교한다. 가능하다면 cut 위치도 변경하여 서버에서 전달되는 `u`의 깊이와 클라이언트로 반환되는 `dL/dz`의 위치가 정보 누출량에 미치는 영향을 측정한다.

라벨 추론은 accuracy, macro F1, clustering purity와 calibration을 이용해 평가한다. 이미지 복원은 MSE, MAE, PSNR, SSIM, edge similarity, feature similarity 및 피해 모델의 재분류 결과를 이용해 평가한다. Coarse와 refined 결과는 동일한 holdout 표본에서 비교하고, label 추론에 성공한 표본과 실패한 표본을 분리하여 보고한다. 최소 여러 개의 random seed를 사용하여 평균과 표준편차를 제시하면 특정 초기화나 holdout 선택에 의존하는 결과를 줄일 수 있다.

## 현재 코드와의 관계

현재 저장소의 `strong_u_grad_z` 조건은 decoder 입력으로 `u`와 `grad_z`를 함께 사용하는 방향과 일치한다. 코드에서 `grad_z` 또는 `grad_g_to_f`로 표현되는 값은 본 문서의 `dL/dz`에 해당한다. 따라서 coarse decoder의 입력 신호 자체는 현재 가설과 연결할 수 있다.

다만 기존 실험이 `dL/du`에서 얻은 inferred label을 condition으로 사용했다면 해당 결과는 `u`와 `dL/dz`만 관찰하는 순수한 위협 모델을 완전히 검증한 것이 아니다. 기존 80 epoch 결과를 해석할 때에는 decoder가 받은 이미지 복원 신호와 label predictor가 사용한 신호를 분리해서 확인해야 한다. 후속 구현에서는 `dL/dz` 전용 label predictor를 추가하고, 공격자 record에서 `z`와 `dL/du`가 실제로 제거되었는지 검증해야 한다.

현재 저장소는 ClientFront, ServerMiddle 및 ClientTail을 실제 네트워크의 별도 프로세스로 실행하지 않고 하나의 Python 프로세스 안에서 논리적 cut과 `detach()`를 이용해 Split Learning 통신을 모사한다. 따라서 현재 observation은 실제 외부 packet sniffer가 수집한 결과가 아니라, 허가된 passive observer가 복호화 이후 수집할 텐서를 모사한 결과다. 실제 네트워크 가설을 검증하려면 각 구간을 별도 프로세스로 분리하고 request 식별자가 있는 RPC 또는 TCP transport를 구성한 뒤, 허가된 observer 환경에서 평문 조건과 TLS 조건을 각각 평가해야 한다.

## 연구 가설

첫 번째 가설은 하나의 클라이언트가 수신하는 `dL/dz`에 실제 라벨과 입력 특징의 영향이 남아 있어, 실제 라벨을 직접 관찰하지 않고도 클래스 추론이 가능하다는 것이다. 두 번째 가설은 동일 학습 단계의 `u`와 `dL/dz`를 결합하면 어느 한 신호만 사용할 때보다 holdout 이미지의 구조와 의미를 더 잘 복원할 수 있다는 것이다. 세 번째 가설은 관찰 가능한 신호로 추론한 soft label을 제공하면 label condition이 없는 경우보다 복원 품질이 향상된다는 것이다.

네 번째 가설은 정확한 동일 client 및 동일 step pairing이 shuffled pairing보다 개별 이미지 복원에 유리하다는 것이다. 다섯 번째 가설은 공개 auxiliary 데이터만으로 학습한 refiner가 피해 holdout을 학습하지 않고도 coarse reconstruction의 구조적 충실도나 지각 품질을 개선할 수 있다는 것이다. 마지막 가설은 refiner의 시각적 개선 중 일부가 transcript에서 복구된 정보가 아니라 공개 데이터의 image prior로 생성된 hallucination일 수 있다는 것이다.

## 해석과 연구 범위

복원 결과는 원본 이미지의 완전한 복구가 아니라 근사 reconstruction으로 표현한다. 시각적으로 보기 좋은 결과만으로 개인 정보가 정확하게 복구되었다고 주장하지 않으며, 정량 지표와 표본별 error map을 함께 제시한다. `dL/du`에서 추론한 label을 사용한 결과는 본 가설의 순수한 공격 결과에서 제외하고 별도의 참고 조건으로 표시한다.

본 연구는 소유하거나 명시적으로 허가받은 Split Learning 환경에서 통신 transcript의 privacy leakage를 측정하는 것을 목적으로 한다. 제삼자 네트워크에 대한 무단 감청, 인증 우회, 악성 메시지 삽입, 탐지 회피 및 공격자 신원 은폐는 연구 범위에 포함하지 않는다.

## 연구 정의

본 연구는 U-shaped Split Learning에서 하나의 클라이언트가 서버로부터 수신하는 동일 학습 단계의 `u`와 `dL/dz`를 수동적으로 관찰했을 때 숨겨진 라벨과 원본 이미지를 어느 수준까지 근사 복원할 수 있는지를 검증하고, 공개 데이터로 학습한 refiner가 엄격한 holdout 조건에서 복원 충실도와 지각 품질에 미치는 영향을 평가한다.

## 구현된 이미지 전용 정제기

정제기 코드는 `src/client_received_transcript_refiner`에 구현되어 있다. 정제기는 `comparison_grid.png`를 입력으로 사용하지 않고, 고정된 coarse decoder가 각 transcript의 `u`와 `dL/dz`로 생성한 개별 64×64 복원 텐서를 입력으로 사용한다. Residual U-Net은 coarse 영상에 더할 제한된 크기의 잔차만 학습하며, 마지막 잔차 출력층을 0으로 초기화하므로 학습 시작 시점에는 coarse 영상과 같은 결과를 출력한다. 기본 설정에서는 각 픽셀의 변화량을 최대 0.1로 제한한다.

정제기 학습 과정에서는 기존 decoder checkpoint만 불러오고 그 파라미터를 완전히 고정한다. 공개 auxiliary train과 validation transcript에서 coarse 영상을 생성하고 evaluator 전용 공개 원본과 비교하여 정제기만 학습한다. 체크포인트에는 사용한 decoder 파일의 SHA-256과 학습 및 검증 transcript ID가 저장된다. 최종 평가 명령은 평가 transcript가 이 ID들과 겹치면 기본적으로 오류를 발생시키므로 피해 holdout이 정제기 학습에 섞이는 것을 방지한다. Split Learning victim checkpoint는 정제기 학습이나 평가 프로세스에서 직접 불러오지 않는다.

저장소 루트의 PowerShell에서 다음 명령으로 현재 100 epoch coarse decoder를 고정하고 정제기를 40 epoch 학습할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_refiner.pipeline.train_refiner `
  --decoder-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_label_head_100epoch\checkpoints\client_received_decoder_best.pt `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_validation\evaluator_manifest.csv `
  --output workspace\results\client_received_transcript_refiner\rpc_u_grad_z_label_head_100epoch `
  --epochs 40 `
  --batch-size 8 `
  --learning-rate 0.0001 `
  --device cuda
```

학습이 완료되면 다음 명령으로 정제기 학습에 사용하지 않은 피해 holdout 20장을 평가한다. Decoder 경로는 refiner checkpoint에 기록되어 있으므로 평가 명령에서 다시 지정하지 않아도 된다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_refiner.pipeline.evaluate_refiner `
  --refiner-checkpoint workspace\results\client_received_transcript_refiner\rpc_u_grad_z_label_head_100epoch\checkpoints\residual_unet_refiner_best.pt `
  --attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\victim_holdout\attacker_manifest.csv `
  --evaluator-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_refiner\rpc_u_grad_z_label_head_100epoch\evaluation `
  --batch-size 8 `
  --max-grid-images 20 `
  --device cuda
```

평가 폴더에는 개별 `originals`, `coarse_reconstructions`, `refined_reconstructions`와 표본별 `comparisons`가 분리되어 저장된다. 전체 비교 이미지는 `refinement_comparison_grid.png`, 표본별 지표는 `refinement_metrics.csv`, 평균 지표는 `refinement_summary.json`에 저장된다. 요약 파일의 `delta_refined_minus_coarse`는 refined 지표에서 coarse 지표를 뺀 값이므로 PSNR과 SSIM은 양수일 때 개선이고 MSE와 MAE는 음수일 때 개선이다. 시각적으로만 선명해지고 이 정량 지표가 악화되면 정보 복구가 아니라 공개 데이터 prior에 의한 시각적 보정 또는 hallucination으로 해석해야 한다.

## OOF와 transcript-conditioned 정제기 실험

기존 image-only 정제기는 coarse decoder가 이미 학습한 공개 train 이미지를 다시 입력으로 사용하여 train 오류에는 과적합하고 unseen 오류에는 일반화하지 못할 수 있다. 이를 줄이기 위해 공개 train을 여러 fold로 나누고, 각 샘플을 해당 샘플을 학습하지 않은 fold decoder로 복원한 out-of-fold coarse 데이터를 생성한다. 이 과정은 공개 auxiliary train과 validation만 사용하며 피해 holdout은 읽지 않는다.

다음 명령은 5개의 fold decoder를 각각 100 epoch 학습하고 OOF coarse train 450장과 기준 decoder의 validation coarse 75장을 생성한다. 계산 시간을 먼저 확인하려면 `--decoder-epochs 100`을 `10`으로 낮춰 파이프라인만 검증할 수 있지만, 최종 비교에서는 기준 decoder와 가까운 품질을 얻기 위해 100 epoch를 사용한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_refiner.pipeline.prepare_oof_data `
  --reference-decoder-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_label_head_100epoch\checkpoints\client_received_decoder_best.pt `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_validation\evaluator_manifest.csv `
  --output workspace\results\client_received_transcript_refiner\oof_u_grad_z_5fold `
  --folds 5 `
  --decoder-epochs 100 `
  --batch-size 8 `
  --device cuda
```

OOF 데이터 생성이 끝나면 coarse 이미지와 raw `u`, `dL/dz`를 함께 입력받는 작은 조건부 정제기를 학습한다. 현재 공개 표본 수가 크지 않으므로 base와 condition channel을 16으로 제한하고 최대 픽셀 변화량도 0.05로 제한한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_refiner.pipeline.train_conditioned_refiner `
  --inference-decoder-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_label_head_100epoch\checkpoints\client_received_decoder_best.pt `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_train\evaluator_manifest.csv `
  --train-coarse-manifest workspace\results\client_received_transcript_refiner\oof_u_grad_z_5fold\oof_train\coarse_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\aux_validation\evaluator_manifest.csv `
  --validation-coarse-manifest workspace\results\client_received_transcript_refiner\oof_u_grad_z_5fold\reference_validation\coarse_manifest.csv `
  --output workspace\results\client_received_transcript_refiner\oof_conditioned_u_grad_z `
  --epochs 40 `
  --batch-size 8 `
  --learning-rate 0.00005 `
  --base-channels 16 `
  --condition-channels 16 `
  --bottleneck-blocks 1 `
  --max-residual 0.05 `
  --residual-weight 0.1 `
  --low-frequency-weight 0.1 `
  --device cuda
```

정제기 학습은 학습 전 identity 상태를 `epoch 0` 검증 기준선으로 먼저 측정한다. 이후 epoch가 이 기준선보다 낮은 validation loss를 만들 때만 해당 모델을 최종 checkpoint로 채택한다. 따라서 `best_epoch`가 0이면 현재 공개 데이터에서는 일반화 가능한 정제 개선을 찾지 못했다는 의미이며, 그 경우 최종 정제 영상은 coarse와 동일하게 유지된다.

다음 명령은 정제기 학습에 사용하지 않은 피해 holdout 20장을 평가하고 각 표본을 `Original | Coarse | Refined` 순서로 한 번에 저장한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_refiner.pipeline.evaluate_refiner `
  --refiner-checkpoint workspace\results\client_received_transcript_refiner\oof_conditioned_u_grad_z\checkpoints\transcript_conditioned_refiner_best.pt `
  --attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\victim_holdout\attacker_manifest.csv `
  --evaluator-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_100epoch_new_holdout20\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_refiner\oof_conditioned_u_grad_z\evaluation `
  --batch-size 8 `
  --max-grid-images 20 `
  --device cuda
```

전체 3열 비교 결과는 `workspace/results/client_received_transcript_refiner/oof_conditioned_u_grad_z/evaluation/refinement_comparison_grid.png`에 저장된다. 같은 폴더의 `refinement_summary.json`에서 coarse와 refined 평균 지표 및 변화량을 확인하고, `checkpoints/training_history.json`과 checkpoint의 `best_epoch`를 함께 확인한다.

## 128×128 multi-scale decoder v2와 보수적 후처리

정제 단계에서 coarse 이미지에 없는 정보를 다시 만드는 데 한계가 있으므로 복원 decoder 자체를 개선하는 실험을 별도로 수행한다. Decoder v2는 기존 bilinear upsampling 대신 ICNR로 초기화한 PixelShuffle을 사용하고, `u`와 `dL/dz` 특징을 16×16 시작 지점뿐 아니라 32×32, 64×64, 128×128의 각 복원 단계에 FiLM 방식으로 다시 주입한다. Laplacian pyramid loss는 여러 해상도의 고주파 경계 오차를 함께 줄인다.

기존 64×64 observation의 평가 원본은 이미 64×64로 저장되어 있으므로 128×128 복원을 공정하게 평가하려면 transcript와 evaluator target을 128×128로 다시 수집해야 한다. 다음 RPC 명령은 동일한 128×128 observation을 만들면서 비교 기준인 bilinear decoder를 100 epoch 학습한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.run_rpc_experiment `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --client-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --data workspace\data\dataset `
  --output workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline `
  --aux-train-split train `
  --aux-validation-split val `
  --victim-split new_holdout `
  --holdout-count 20 `
  --holdout-start-index 0 `
  --holdout-labels cat dog `
  --image-size 128 `
  --decoder-architecture baseline_bilinear `
  --epochs 100 `
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
  --device cuda
```

수집된 동일 observation을 이용해 decoder v2만 별도로 학습한다. 이 프로세스는 Split Learning victim checkpoint를 불러오지 않고 공개 train과 validation manifest만 사용한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.train_from_transcripts `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_validation\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_v2 `
  --image-size 128 `
  --decoder-architecture multiscale_pixelshuffle `
  --epochs 100 `
  --batch-size 8 `
  --use-label-head `
  --signal-spatial-size 16 `
  --signal-channels 64 `
  --decoder-base-channels 256 `
  --decoder-min-channels 32 `
  --refinement-blocks 1 `
  --film-strength 0.1 `
  --edge-weight 0.1 `
  --perceptual-weight 0.1 `
  --laplacian-weight 0.25 `
  --device cuda
```

두 decoder의 학습이 끝나면 다음 명령으로 같은 unseen holdout 20장을 `Original | Decoder v1 | Decoder v2 | V2 + visual` 순서로 비교한다. 마지막 열은 생성 모델이 아니라 고정된 약한 chroma denoise, sharpen, contrast 및 saturation 보정이며 복원 결과가 아닌 시각화 후처리 조건으로 구분한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.evaluate_v2_comparison `
  --baseline-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\attack_training\checkpoints\client_received_decoder_best.pt `
  --v2-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_v2\checkpoints\client_received_decoder_best.pt `
  --attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\victim_holdout\attacker_manifest.csv `
  --evaluator-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_attack\decoder_v2_128_comparison `
  --batch-size 8 `
  --max-grid-images 20 `
  --chroma-denoise 0.15 `
  --sharpen-amount 0.2 `
  --contrast 1.03 `
  --saturation 1.03 `
  --device cuda
```

전체 비교 이미지는 `workspace/results/client_received_transcript_attack/decoder_v2_128_comparison/decoder_v2_comparison_grid.png`에 저장되고, 평균 지표는 `decoder_v2_summary.json`, 표본별 지표는 `decoder_v2_metrics.csv`에 저장된다. Decoder v2의 PSNR과 SSIM이 baseline보다 함께 개선되어야 실제 decoder 복원 향상으로 해석한다. 마지막 후처리 열은 정량 지표가 악화될 수 있으며, 이 경우 논문에서는 원본 정보 복구가 아니라 가독성을 위한 시각화로만 표시한다.

## 128×128 residual detail 미세조정 실험

Multi-scale PixelShuffle decoder가 공개 학습 데이터에 과적합한 결과를 반영하여, 다음 실험은 기존 bilinear decoder 구조를 그대로 유지하고 출력단에 작은 residual detail branch만 추가한다. 이 branch는 bilinear 복원 이미지와 `u`, `dL/dz`의 저차원 특징을 함께 받아 RGB logit에 제한된 크기의 보정을 더한다. 마지막 convolution은 0으로 초기화되므로 학습 시작 시 출력은 기존 baseline과 정확히 같다.

초기화에 사용하는 checkpoint는 피해 Split Learning 모델이 아니라 공개 보조 transcript로 이미 학습한 공격자 baseline decoder이다. 따라서 이 과정에서도 피해 모델의 계층이나 가중치를 공격자 프로세스가 직접 불러오지 않는다. 미세조정 전 상태를 검증 기준으로 보존하고, 낮은 학습률과 early stopping을 사용하여 검증 데이터에서 baseline보다 나빠진 상태가 최종 checkpoint로 선택되는 것을 막는다.

기존 128×128 observation을 재사용하여 residual detail decoder를 학습한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.train_from_transcripts `
  --train-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_train\attacker_manifest.csv `
  --train-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_train\evaluator_manifest.csv `
  --validation-attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_validation\attacker_manifest.csv `
  --validation-target-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\aux_validation\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_residual_detail `
  --image-size 128 `
  --decoder-architecture residual_detail `
  --initialize-from-decoder-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\attack_training\checkpoints\client_received_decoder_best.pt `
  --epochs 80 `
  --batch-size 8 `
  --learning-rate 0.0001 `
  --weight-decay 0.0001 `
  --use-label-head `
  --signal-spatial-size 16 `
  --signal-channels 64 `
  --decoder-base-channels 256 `
  --decoder-min-channels 32 `
  --refinement-blocks 1 `
  --detail-condition-channels 8 `
  --detail-channels 16 `
  --detail-scale 0.25 `
  --edge-weight 0.1 `
  --perceptual-weight 0.1 `
  --laplacian-weight 0.25 `
  --early-stopping-patience 12 `
  --early-stopping-min-delta 0.0001 `
  --preserve-initial-state `
  --device cuda
```

학습이 끝나면 같은 unseen holdout 20장을 `Original | Decoder v1 | Residual detail` 순서로 비교한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.evaluate_residual_detail_comparison `
  --baseline-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\attack_training\checkpoints\client_received_decoder_best.pt `
  --residual-detail-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_residual_detail\checkpoints\client_received_decoder_best.pt `
  --attacker-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\victim_holdout\attacker_manifest.csv `
  --evaluator-manifest workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_attack\residual_detail_128_comparison `
  --batch-size 8 `
  --max-grid-images 20 `
  --device cuda
```

전체 비교 이미지는 `residual_detail_128_comparison/residual_detail_comparison_grid.png`, 평균 지표는 `residual_detail_summary.json`, 표본별 지표는 `residual_detail_metrics.csv`에 저장된다. 성공 기준은 residual detail의 평균 PSNR과 SSIM이 baseline보다 모두 높고, 표본별 향상 비율도 0.5를 넘는 것이다. 한 지표만 소폭 좋아지거나 시각적으로만 날카로워진 경우에는 세부 정보 복구가 입증된 것으로 해석하지 않는다.

## Proxy에서 z를 추가 수집하는 확장 실험

기본 RPC 공격은 서버에서 클라이언트로 전달되는 `u`와 `dL/dz`만 저장한다. 확장 조건에서는 같은 protocol-aware passive relay가 클라이언트에서 서버로 전달되는 `z`도 같은 request ID로 저장한다. Proxy는 피해 모델 checkpoint, 원본 및 라벨을 불러오지 않으며, 공격 decoder는 공개 auxiliary transcript와 공개 target만 이용해 학습한다. 이 조건은 단순한 TLS 외부 패킷 관찰자가 아니라 애플리케이션 계층의 평문 tensor payload를 볼 수 있는 양방향 relay를 가정한다.

다음 명령은 기존과 같은 128×128 설정으로 `z`, `u`, `dL/dz`를 새로 수집하고 `z_u_grad_z_bilinear` decoder를 100 epoch 학습한다. 기존 observation에는 `z`가 저장되어 있지 않으므로 반드시 다시 수집해야 한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.run_rpc_experiment `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --client-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --data workspace\data\dataset `
  --output workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_100epoch `
  --aux-train-split train `
  --aux-validation-split val `
  --victim-split new_holdout `
  --holdout-count 20 `
  --holdout-start-index 0 `
  --holdout-labels cat dog `
  --image-size 128 `
  --capture-z `
  --decoder-architecture z_u_grad_z_bilinear `
  --epochs 100 `
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
  --max-grid-images 20 `
  --device cuda
```

실행 후 각 observation의 `proxy_capture_summary.json`에는 `attacker_visible_signals`가 `z`, `u`, `dL/dz`로 기록되고, `process_boundary_audit.json`에는 proxy가 저장한 신호와 공격 trainer가 victim role checkpoint를 불러오지 않았다는 사실이 기록된다. 개별 attacker NPZ에는 `smashed_z`, `server_output_u`, `grad_g_to_f`만 저장된다.

다음 명령은 새로 수집한 동일 victim holdout에서 기존 `u + dL/dz` baseline과 `z + u + dL/dz` 결과를 `Original | u + dL/dz | z + u + dL/dz` 순서로 비교한다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.evaluate_z_signal_comparison `
  --u-grad-z-checkpoint workspace\results\client_received_transcript_attack\rpc_u_grad_z_128_baseline\attack_training\checkpoints\client_received_decoder_best.pt `
  --z-u-grad-z-checkpoint workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_100epoch\attack_training\checkpoints\client_received_decoder_best.pt `
  --attacker-manifest workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_100epoch\observations\victim_holdout\attacker_manifest.csv `
  --evaluator-manifest workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_100epoch\observations\victim_holdout\evaluator_manifest.csv `
  --class-names cat dog pug `
  --output workspace\results\client_received_transcript_attack\z_signal_128_comparison `
  --batch-size 8 `
  --max-grid-images 20 `
  --device cuda
```

전체 비교 이미지는 `z_signal_128_comparison/z_signal_comparison_grid.png`, 평균 지표는 `z_signal_summary.json`, 표본별 지표는 `z_signal_metrics.csv`에 저장된다. `delta_z_u_grad_z_minus_u_grad_z`에서 PSNR과 SSIM은 양수, MSE와 MAE는 음수일 때 `z` 추가 조건이 개선된 것이다. 이 결과는 제한된 downlink 관찰과 `z`까지 보이는 확장 관찰 조건의 위험 차이로 해석한다.

## 1만 개 보조 학습 표본 실험

기존 RPC 실험의 실제 auxiliary train transcript는 450개이고 validation은 75개이다. 다음 준비 과정은 train의 원본 450개를 모두 한 번씩 포함하고, train 원본에만 재현 가능한 crop, 좌우 반전, 밝기·대비·채도 변형을 적용해 총 10,000개 표본을 만든다. 클래스별 수는 cat 3,334개, dog 3,333개, pug 3,333개이다. validation, test, new_holdout은 변형의 입력으로 사용하지 않고 그대로 연결하거나 복사한다.

```powershell
.\.venv\Scripts\python.exe -m src.shared.data.prepare_augmented_aux_dataset `
  --source workspace\data\dataset `
  --output workspace\data\dataset_aux_10k `
  --target-train-count 10000 `
  --image-size 128 `
  --seed 42 `
  --workers 8
```

이 데이터는 서로 독립적인 원본 사진 10,000장이 아니라 원본 450장에서 만든 학습 표본 10,000장이다. 논문에는 원본 수와 변형 후 학습 표본 수를 함께 보고해야 하며, 단순히 “공개 이미지 10,000장”이라고 표현하면 안 된다. 이 실험은 데이터 변형에 따른 과적합 완화 여부를 확인할 수 있지만, 새로운 원본 사진을 추가했을 때의 효과를 직접 입증하지는 않는다.

현재 `z + u + dL/dz` 조건을 1만 개 표본으로 실행하는 권장 명령은 다음과 같다. 50 epoch를 상한으로 두고 validation이 8 epoch 동안 개선되지 않으면 조기 종료한다. 1 epoch의 표본 수가 기존보다 약 22배 많으므로 기존의 100 epoch를 그대로 적용할 필요는 없다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.run_rpc_experiment `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --client-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --data workspace\data\dataset_aux_10k `
  --output workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k `
  --aux-train-split train `
  --aux-validation-split val `
  --victim-split new_holdout `
  --holdout-count 20 `
  --holdout-start-index 0 `
  --holdout-labels cat dog `
  --image-size 128 `
  --capture-z `
  --decoder-architecture z_u_grad_z_bilinear `
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
  --max-grid-images 20 `
  --seed 42 `
  --process-timeout-seconds 86400 `
  --device cuda
```

실험이 끝나면 `evaluation/comparison_grid.png`에서 원본과 복원 결과를 확인하고, `evaluation/reconstruction_summary.json`을 기존 450개 학습 결과와 비교한다. 동일한 victim holdout 20장, seed, 구조와 손실 가중치를 유지해야 데이터 규모 외 조건이 섞이지 않는다.

## 학습 없이 holdout 100장으로 확대 평가

`new_holdout`은 기존 20장을 포함하면서 cat 50장과 dog 50장, 총 100장으로 확장한다. 다음 데이터 준비 명령은 공개 train, validation, test 및 anchor에 이미 사용된 원본을 제외하며 같은 seed를 사용하므로 기존 20장이 100장 집합의 부분집합으로 유지된다.

```powershell
.\.venv\Scripts\python.exe -m src.shared.data.prepare_new_holdouts `
  --count 100 `
  --labels cat dog `
  --seed 142 `
  --workers 8
```

다음 명령은 1만 표본으로 이미 학습한 decoder checkpoint를 그대로 유지하고 새 holdout 100장의 RPC transcript만 수집해 평가한다. Auxiliary transcript 수집과 decoder 재학습은 수행하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.client_received_transcript_attack.pipeline.evaluate_rpc_holdout `
  --server-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\server_role.pt `
  --client-role-checkpoint workspace\results\client_received_transcript_attack\rpc_roles\client_role.pt `
  --decoder-checkpoint workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k\attack_training\checkpoints\client_received_decoder_best.pt `
  --data workspace\data\dataset `
  --output workspace\results\client_received_transcript_attack\rpc_z_u_grad_z_128_aux10k_holdout100 `
  --victim-split new_holdout `
  --holdout-count 100 `
  --holdout-start-index 0 `
  --holdout-labels cat dog `
  --image-size 128 `
  --capture-z `
  --batch-size 8 `
  --max-grid-images 100 `
  --process-timeout-seconds 14400 `
  --device cuda
```

전체 비교 이미지는 `evaluation/comparison_grid.png`, 평균 지표는 `evaluation/reconstruction_summary.json`, 표본별 지표는 `evaluation/reconstruction_metrics.csv`에 저장된다. `holdout_evaluation_audit.json`에는 holdout 수, 관찰 신호와 decoder를 재학습하지 않았다는 사실이 기록된다.
