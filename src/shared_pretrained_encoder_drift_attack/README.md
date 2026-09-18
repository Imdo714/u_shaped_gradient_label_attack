# Shared pretrained encoder drift attack

동일한 Autoencoder Encoder에서 시작한 여러 Split Learning client를 독립적으로
fine-tuning하고, 악성 client가 초기 상태의 public auxiliary transcript로 학습한
고정 공격기를 victim epoch별 `z`와 `dL/dz`에 적용한다.

- `pipeline/pretrain_autoencoder.py`: Autoencoder 사전학습과 `E⁽⁰⁾` 저장
- `pipeline/run_drift_attack.py`: classifier warmup, 다중 client 학습, 공격 및 drift 평가
- `pipeline/run_online_z_attack.py`: 실제 `z`만 사용하는 online 라벨 추론 비교
- `pipeline/run_online_z_gradient_attack.py`: 실제 `z + dL/dz`를 결합한 online 라벨 추론 비교
- `pipeline/run_online_fair_z_comparison.py`: 한 Split 실행을 공유하는 `dL/dz` / `u + dL/dz` / `z` / `z + dL/dz` 공정 비교
- `pipeline/run_paper_label_condition.py`: ResNet-20 한 조건에서 C_psi/Cosine/PCAT-label/SDAR-label paired 비교
- `pipeline/run_paper_label_sweep.py`: CIFAR-10과 Animal5의 split·auxiliary·관측량 전체 재개 가능 스위프
- `pipeline/run_unsplit_comparison.py`: MNIST/Fashion-MNIST/CIFAR-10 동일 샘플 비교
- `models.py`: 독립적인 representation/gradient branch와 fusion decoder
- `unsplit_benchmark.py`: 공개 UnSplit 구조, 교대 최적화 baseline, native-size Decoder

실행 명령은
`readme/idea/shared_pretrained_encoder_drift_attack/README.md`에 정리되어 있다.

출력의 `summary.csv`는 client/epoch/조건별 MSE, MAE, PSNR, SSIM을 기록하고,
`drift_metrics.csv`는 초기 Encoder 대비 weight L2와 representation cosine/L2를
기록한다. 피해 원본은 평가에만 사용되며 공격기 학습에는 사용하지 않는다.

UnSplit 비교 pipeline은 공식 demo의 클래스별 첫 test sample 선택 규칙을
`target_manifest.csv`로 고정한다. `summary.csv`와 `final/comparison_grid.png`에서
우리 고정 Decoder와 UnSplit을 같은 victim smashed data 기준으로 비교한다.
