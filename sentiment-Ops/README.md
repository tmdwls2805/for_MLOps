# Sentiment-Ops - 감성 분석 MLOps 시스템

## 프로젝트 개요

Sentiment-Ops는 텍스트 감성 분석 모델을 학습, 평가, 배포하는 완전한 MLOps 파이프라인입니다. DistilBERT 기반의 경량화된 모델을 사용하여 텍스트의 긍정/부정 감정을 분류하며, MLflow를 활용한 실험 추적 및 FastAPI 기반의 REST API 서비스를 제공합니다.

## 주요 기능

- **감성 분석 모델**: DistilBERT 기반 이진 분류 (긍정/부정)
- **MLOps 파이프라인**: 데이터 전처리 → 모델 학습 → 평가 → 배포
- **실험 추적**: MLflow를 통한 모델 버전 관리 및 성능 추적
- **REST API**: FastAPI 기반의 실시간 감성 분석 서비스
- **모델 평가**: 정확도, 정밀도, 재현율, F1 스코어 등 상세 평가

## 기술 스택

### ML/DL
- **Transformers**: Hugging Face 트랜스포머 라이브러리
- **DistilBERT**: 경량화된 BERT 모델
- **PyTorch**: 딥러닝 프레임워크
- **scikit-learn**: 평가 지표 계산

### MLOps
- **MLflow**: 실험 추적 및 모델 관리
- **FastAPI**: 모델 서빙 API
- **Uvicorn**: ASGI 서버

### 데이터 처리
- **Pandas**: 데이터 전처리
- **Datasets**: Hugging Face 데이터셋 라이브러리

## 프로젝트 구조

```
sentiment-Ops/
├── app/
│   └── main.py              # FastAPI 서비스 (모델 배포)
├── data/
│   ├── train.csv            # 학습 데이터
│   └── test.csv             # 테스트 데이터
├── scripts/
│   ├── preprocess.py        # 데이터 전처리
│   ├── train.py             # 모델 학습
│   ├── evaluate.py          # 모델 평가
│   └── analyze_emotion_distribution.py  # 감정 레이블 분포 분석
├── models/
│   ├── checkpoint-{N}/      # 학습 중 체크포인트
│   └── final/               # 최종 학습 모델
├── mlruns/                  # MLflow 실험 추적 데이터
└── README.md               # 프로젝트 문서
```

## 시스템 요구사항

### 소프트웨어
- **Python**: 3.11 이상
- **pip**: 최신 버전

### 하드웨어
- **RAM**: 최소 4GB 이상 권장
- **디스크**: 약 2GB 여유 공간
- **CPU**: 멀티코어 권장 (GPU 선택사항)

## 설치 방법

### 1. Python 가상환경 생성

```bash
# Python 3.11 설치 (Homebrew 사용)
brew install python@3.11

# 가상환경 생성
/opt/homebrew/bin/python3.11 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

### 2. 의존성 패키지 설치

```bash
pip install --upgrade pip
pip install transformers datasets torch scikit-learn pandas mlflow fastapi uvicorn accelerate
```

## 사용 방법

### 전체 파이프라인 실행 순서

```
1. 데이터 전처리 → 2. 모델 학습 → 3. 모델 평가 → 4. API 서비스 배포
```

### 1. 데이터 전처리

```bash
cd sentiment-Ops
python scripts/preprocess.py
```

**기능**:
- 원본 데이터 로드 및 정제
- 학습/테스트 데이터 분할
- `data/train.csv` 및 `data/test.csv` 생성

**데이터 형식**:
```csv
text,label
"This is great!",1
"I hate this.",0
```

### 2. 모델 학습

```bash
python scripts/train.py
```

**기능**:
- DistilBERT 모델 로드 및 파인튜닝
- MLflow를 통한 실험 추적
- 학습 진행 상황 모니터링
- 모델 체크포인트 및 최종 모델 저장

**학습 설정** (train.py에서 수정 가능):
- `num_train_epochs`: 에폭 수 (기본값: 20)
- `per_device_train_batch_size`: 배치 크기 (기본값: 4)
- `evaluation_strategy`: 평가 전략 (기본값: "epoch")

**출력**:
- `models/checkpoint-{N}/`: 각 에폭의 체크포인트
- `models/final/`: 최종 학습된 모델
- `mlruns/`: MLflow 실험 데이터

**MLflow UI 확인**:
```bash
mlflow ui --backend-store-uri file:./mlruns
```
브라우저에서 `http://localhost:5000` 접속하여 실험 결과 확인

### 3. 모델 평가

```bash
python scripts/evaluate.py
```

**기능**:
- 학습된 모델 로드
- 평가 데이터로 성능 측정
- 상세 평가 리포트 생성
- 새로운 텍스트로 실시간 테스트

**출력 예시**:
```
============================================================
📊 감성 분석 모델 평가 리포트
============================================================

평가 데이터: 1개 샘플

✅ 정확도 (Accuracy): 100.00%
✅ 정밀도 (Precision): 100.00%
✅ 재현율 (Recall): 100.00%
✅ F1 스코어: 1.0000

혼동 행렬 (Confusion Matrix):
[[0 0]
 [0 1]]

============================================================
📝 샘플별 예측 결과
============================================================

✅ 샘플 1:
   텍스트: I love this!
   실제: 긍정 (label=1)
   예측: 긍정 (label=1)

============================================================
🧪 새로운 텍스트 테스트
============================================================

텍스트: This movie is amazing!
예측: 긍정 😊 (신뢰도: 92.34%)

텍스트: I hate this product.
예측: 부정 😞 (신뢰도: 88.76%)
```

### 4. API 서비스 배포

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**API 엔드포인트**:

#### POST /predict
텍스트의 감성을 분석합니다.

**요청**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This is amazing!"
```

**응답**:
```json
{
  "label": "LABEL_1",
  "score": 0.9876
}
```

- `LABEL_0`: 부정 (Negative)
- `LABEL_1`: 긍정 (Positive)
- `score`: 신뢰도 (0.0 ~ 1.0)

**Swagger UI**:
`http://localhost:8000/docs`에서 API 문서 및 테스트 가능

## 고급 기능

### 감정 레이블 분포 분석

감성 대화 말뭉치의 감정 레이블(E10, E20 등) 분포를 분석합니다.

```bash
python scripts/analyze_emotion_distribution.py
```

**전제 조건**:
- 프로젝트 루트에 `감성대화말뭉치(최종데이터)_Training.json` 파일 필요

**출력**:
- `data/emotion_distribution.csv`: 전체 감정 레이블 통계
- `data/emotion_distribution.png`: 시각화 그래프
- 콘솔에 상위 10개 감정 레이블 분포 출력

### MLflow 실험 관리

**MLflow UI 실행**:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

**기능**:
- 모든 학습 실험 비교
- 하이퍼파라미터 추적
- 성능 메트릭 시각화
- 모델 아티팩트 관리

## 데이터 형식

### 학습 데이터 (train.csv)
```csv
text,label
"I love this movie!",1
"This is terrible.",0
"Great product!",1
```

### 감성 레이블
- `0`: 부정 (Negative)
- `1`: 긍정 (Positive)

## 성능 최적화

### 학습 최적화

1. **배치 크기 조정**:
```python
# train.py에서 수정
per_device_train_batch_size=8  # 메모리에 맞게 조정
```

2. **에폭 수 증가**:
```python
num_train_epochs=5  # 더 많은 에폭으로 학습
```

3. **Learning Rate 조정**:
```python
learning_rate=2e-5  # 기본값은 5e-5
```

### GPU 사용

GPU가 있는 경우 자동으로 감지하여 사용합니다. CPU만 있어도 정상 작동합니다.

## 문제 해결

### 1. 메모리 부족 에러

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결 방법**:
- 배치 크기 줄이기: `per_device_train_batch_size=2`
- Gradient accumulation 사용

### 2. MLflow 경고

**증상**:
```
FutureWarning: Filesystem tracking backend is deprecated
```

**해결 방법**:
경고일 뿐 정상 작동합니다. 향후 SQLite 백엔드로 전환 권장.

### 3. Accelerate 버전 에러

**증상**:
```
ImportError: Using the Trainer with PyTorch requires accelerate>=0.21.0
```

**해결 방법**:
```bash
pip install accelerate -U
```

## 모델 정보

### DistilBERT

- **기반 모델**: `distilbert-base-uncased`
- **파라미터**: 약 66M (BERT의 60%)
- **속도**: BERT 대비 60% 빠름
- **성능**: BERT의 97% 성능 유지

### 모델 구조

```
DistilBERT (Encoder)
    ↓
Pooling Layer
    ↓
Pre-classifier (Linear 768 → 768)
    ↓
Classifier (Linear 768 → 2)
    ↓
Softmax (긍정/부정)
```

## API 사용 예시

### Python 코드

```python
import requests

# 감성 분석 요청
response = requests.post(
    'http://localhost:8000/predict',
    data={'text': 'This is an excellent product!'}
)

result = response.json()
print(f"Label: {result['label']}")
print(f"Score: {result['score']:.2%}")
```

### JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append('text', 'I love this!');

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 확장 계획

- [ ] 다중 클래스 감성 분석 (긍정/중립/부정)
- [ ] 감정 세분화 (기쁨, 슬픔, 분노 등)
- [ ] 배치 예측 API 추가
- [ ] Docker 컨테이너화
- [ ] 모델 경량화 (ONNX, TensorRT)
- [ ] 실시간 모니터링 대시보드

## 참고 자료

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트 및 기능 제안은 이슈로 등록해주세요.
