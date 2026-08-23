# 데이터 출처

이 프로젝트의 JPEG는 공개 학술 데이터셋인 **Oxford-IIIT Pet Dataset**에서 선택한 소규모 부분집합입니다.

- 공식 소개: https://www.robots.ox.ac.uk/~vgg/data/pets/
- 사용한 공개 미러: https://github.com/ml4py/dataset-iiit-pet
- 선택 seed: `42`
- 클래스: `cat`, `dog`(pug 제외), `pug`
- 클래스별 분할: train 30장, val 10장, test 10장
- 별도 앵커: 클래스별 1장(학습·검증·테스트 이미지와 중복되지 않음)

`pug`를 별도 클래스로 분리했기 때문에 `dog` 클래스의 품종 목록에서는 pug를 제외했습니다. 클래스와 원본 품종의 대응은 프로젝트 루트의 `workspace/data/dataset_classes.json`에 정의되어 있습니다.

각 파일의 원본 URL, split, 크기, 바이트 수, SHA-256은 `public_subset_manifest.csv`에 기록됩니다.

전체 데이터셋을 다시 준비하려면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m src.shared.data.prepare_public_dataset
```

설정에 새로 추가한 클래스만 준비하고 기존 manifest의 다른 클래스를 보존하려면 다음처럼 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m src.shared.data.prepare_public_dataset --labels pug
```
