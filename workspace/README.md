# Workspace

애플리케이션 소스가 아닌 입력 데이터, 실행 결과, 테스트를 역할별로 관리합니다.

```text
workspace/
├─ data/
│  ├─ dataset/          학습·검증·테스트 이미지와 manifest
│  ├─ anchors/          클래스 의미를 식별하는 기준 이미지
│  └─ dataset_classes.json
├─ results/
│  ├─ checkpoints/      f/g/h 모델 체크포인트
│  ├─ transcripts/      서버 관찰값과 evaluator 정답
│  ├─ reports/          군집 결과, 평가 지표, 표와 그림
│  └─ runs/             cut 설정별 독립 실행 결과
└─ tests/               소스 패키지의 자동화 테스트
```

코드에서 사용하는 기본 경로는
src/shared/configuration/workspace_paths.py에서 중앙 관리합니다.
