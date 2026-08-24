# Label-conditioned 이미지 복원 실험

이 실험은 U-shaped Split Learning에서 관찰되는 z(smashed data), u(g 모델 출력),
dL/du(h에서 g로 반환되는 gradient), gradient로 추론한 라벨을 결합해 원본과 유사한
이미지를 생성하는 decoder를 학습한다. 정확한 원본 복구를 보장하는 실험이 아니라
근사 복원 위험을 PSNR, SSIM, MAE와 분류 일관성으로 측정하는 실험이다.

## 코드 책임 구조

    src/decoder/
      data/                 관측값 수집, 원본/공격자 데이터 분리, gradient 라벨 추론
      surrogate_models/     공격자 소유 f-hat, h-hat 모델
      models/               관측 신호 encoder와 label-conditioned decoder
      losses/               L1 + SSIM 복원 손실
      training/             decoder 및 f-hat/h-hat 학습
      evaluation/           PSNR, SSIM, MAE, 라벨별 결과와 비교 이미지
      pipeline/             전체 실험 순서를 조립하는 CLI

attacker_records에는 z, u, gradient, 추론 라벨만 저장된다. 원본과 실제 라벨은
evaluator_targets에 따로 저장되어 보조 데이터 학습 또는 최종 평가에서만 사용된다.
이 분리는 실제 라벨을 공격자 입력으로 실수로 노출하는 것을 막기 위한 것이다.

## 기본 실험

프로젝트 루트에서 실행한다.

    ..venvScriptspython.exe -m src.experiments.reconstruct_images --decoder-epochs 50 --batch-size 8

기본값은 cut_middle의 10 epoch 체크포인트와 3개 클러스터 centroid/mapping을 사용한다.
보조 train/val 데이터의 실제 라벨로 decoder를 학습하고, test 데이터에는 gradient로
추론한 soft label만 조건으로 전달한다.

## f-hat/h-hat 복제 결과로 학습

    ..venvScriptspython.exe -m src.experiments.reconstruct_images --decoder-observation-source surrogate --surrogate-epochs 20 --decoder-epochs 50 --batch-size 8

이 모드는 먼저 f-hat(x)에서 z를 latent matching으로 학습하고, h-hat은 분류 손실과
dL/du gradient matching으로 학습한다. 이후 f-hat -> 실제 g -> h-hat 경로에서
다시 만든 보조 관측값으로 decoder를 학습한다. victim test 관측값은 실제 f/g/h에서
수집되므로 surrogate와 실제 모델 사이의 오차까지 결과에 포함된다.

## 비교 실험

라벨 추론 오차가 없을 때의 상한선:

    ..venvScriptspython.exe -m src.experiments.reconstruct_images --victim-label-mode oracle

입력 ablation:

    ..venvScriptspython.exe -m src.experiments.reconstruct_images --signals z
    ..venvScriptspython.exe -m src.experiments.reconstruct_images --signals z,u
    ..venvScriptspython.exe -m src.experiments.reconstruct_images --signals z,u,gradient

빠른 동작 확인:

    ..venvScriptspython.exe -m src.experiments.reconstruct_images --decoder-epochs 1 --max-train-samples 3 --max-val-samples 3 --max-test-samples 3 --device cpu

## 결과

각 실행은 workspace/results/decoder/실행시각 아래에 저장된다.

    observations/                         분리된 train/val/test 관측값
    surrogates/surrogate_f.pt             선택 실행 시 f-hat
    surrogates/surrogate_h.pt             선택 실행 시 h-hat
    checkpoints/decoder_best.pt            최적 decoder
    checkpoints/training_history.json      epoch별 손실
    evaluation/reconstruction_metrics.csv  샘플별 PSNR/SSIM/MAE
    evaluation/reconstruction_summary.json 전체 평균 및 라벨 성공/실패별 품질
    evaluation/comparisons/*.png           라벨이 표시된 좌우 비교 이미지
    evaluation/comparison_grid.png         대표 샘플을 모은 전체 비교표
    run_config.json                        재현에 필요한 전체 실행 인자

correct_label_psnr/ssim과 wrong_label_psnr/ssim을 따로 비교해야 라벨 추론 성공이
복원 품질에 미치는 영향을 볼 수 있다. 한 번의 값으로 결론 내리지 말고 여러 seed에서
평균과 표준편차를 보고한다.
