# Sentiment-Ops - 빠른 시작 가이드

## 5분 안에 시작하기

### 1. 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd sentiment-Ops

# 가상환경 활성화 (이미 생성되어 있다면)
source venv/bin/activate

# 패키지 설치 (필요시)
pip install transformers datasets torch scikit-learn pandas mlflow fastapi uvicorn accelerate
```

### 2. 전체 파이프라인 실행

```bash
# Step 1: 데이터 전처리
python scripts/preprocess.py

# Step 2: 모델 학습 (약 1-2분 소요)
python scripts/train.py

# Step 3: 모델 평가
python scripts/evaluate.py

# Step 4: API 서비스 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. API 테스트

**터미널에서**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This is amazing!"
```

**브라우저에서**:
- `http://localhost:8000/docs` 접속
- `/predict` 엔드포인트 테스트

## 주요 명령어

| 작업 | 명령어 |
|------|--------|
| 데이터 전처리 | `python scripts/preprocess.py` |
| 모델 학습 | `python scripts/train.py` |
| 모델 평가 | `python scripts/evaluate.py` |
| API 시작 | `uvicorn app.main:app --port 8000` |
| MLflow UI | `mlflow ui --backend-store-uri file:./mlruns` |

## 문제 발생 시

### Accelerate 에러
```bash
pip install accelerate -U
```

### 메모리 부족
`train.py`에서 배치 크기 줄이기:
```python
per_device_train_batch_size=2
```

## 다음 단계

자세한 사용법은 [README.md](README.md)를 참고하세요.
