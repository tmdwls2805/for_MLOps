# AI Projects Collection

다양한 AI/ML 프로젝트들의 모음입니다. 자연어 처리(NLP), 감성 분석, 문서 질의응답 등 여러 AI 기술을 활용한 프로젝트들이 포함되어 있습니다.

## 📁 프로젝트 목록

### 1. ChatDoc RAG - 문서 기반 질의응답 시스템
> 📂 [`chatdoc-rag/`](./chatdoc-rag/)

로컬 LLM을 활용한 문서 기반 질의응답 시스템입니다. RAG(Retrieval-Augmented Generation) 기술을 사용하여 업로드된 문서를 분석하고 질문에 답변합니다.

**주요 기술**:
- Microsoft Phi-2 LLM
- LangChain RAG
- ChromaDB 벡터 검색
- FastAPI REST API

**주요 기능**:
- PDF, TXT 문서 업로드
- 문서 내용 기반 질의응답
- Swagger UI 제공

**바로 시작하기**:
```bash
cd chatdoc-rag
docker-compose up --build
# http://localhost:8000/docs
```

📖 [자세한 문서 보기](./chatdoc-rag/README.md)

---

### 2. Sentiment-Ops - 감성 분석 MLOps 시스템
> 📂 [`sentiment-Ops/`](./sentiment-Ops/)

완전한 MLOps 파이프라인을 갖춘 텍스트 감성 분석 시스템입니다. DistilBERT를 활용하여 텍스트의 긍정/부정을 분류합니다.

**주요 기술**:
- DistilBERT
- MLflow 실험 추적
- FastAPI 모델 서빙
- Hugging Face Transformers

**주요 기능**:
- 데이터 전처리 파이프라인
- 모델 학습 및 평가
- MLflow 실험 관리
- REST API 감성 분석 서비스
- 감정 레이블 분포 분석

**바로 시작하기**:
```bash
cd sentiment-Ops
python scripts/preprocess.py
python scripts/train.py
python scripts/evaluate.py
uvicorn app.main:app --port 8000
```

📖 [자세한 문서 보기](./sentiment-Ops/README.md)

---

### 3. Store Review Sentiment Classification - 스토어 리뷰 감성 분류
> 📂 [`store_review_sentiment_classification/`](./store_review_sentiment_classification/)

온라인 쇼핑몰 리뷰의 감성을 분류하는 시스템입니다. 여성 의류 리뷰 데이터를 기반으로 Aspect-based Sentiment Analysis를 수행합니다.

**주요 기술**:
- Aspect-based Sentiment Analysis
- 리뷰 데이터 전처리
- 감성 분류 모델

**주요 기능**:
- 리뷰 데이터 전처리
- Aspect별 감성 분석
- 도메인별 감성 분류

**바로 시작하기**:
```bash
cd store_review_sentiment_classification
python preprocessing.py
```

---

## 🛠️ 전체 환경 설정

### 공통 요구사항

**소프트웨어**:
- Python 3.11
- Docker (선택사항)
- Git

**하드웨어**:
- RAM: 8GB 이상 권장
- 디스크: 15GB 이상 여유 공간

### 가상환경 설정

```bash
# Python 3.11 설치
brew install python@3.11

# 가상환경 생성
/opt/homebrew/bin/python3.11 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 공통 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 프로젝트 비교

| 프로젝트 | 주요 기술 | 사용 모델 | API | Docker |
|---------|----------|----------|-----|--------|
| ChatDoc RAG | RAG, Vector DB | Phi-2 (2.7B) | ✅ | ✅ |
| Sentiment-Ops | MLOps, 실험 추적 | DistilBERT (66M) | ✅ | ❌ |
| Store Review | ABSA | TBD | ❌ | ❌ |

## 🚀 빠른 시작 가이드

### ChatDoc RAG
```bash
cd chatdoc-rag && docker-compose up
# http://localhost:8000/docs
```

### Sentiment-Ops
```bash
cd sentiment-Ops
python scripts/train.py && uvicorn app.main:app --port 8001
# http://localhost:8001/docs
```

### Store Review
```bash
cd store_review_sentiment_classification
python preprocessing.py
```

## 📚 기술 스택

### AI/ML
- **Transformers**: Hugging Face 트랜스포머
- **LangChain**: LLM 애플리케이션 프레임워크
- **PyTorch**: 딥러닝 프레임워크
- **Sentence Transformers**: 문장 임베딩

### MLOps
- **MLflow**: 실험 추적 및 모델 관리
- **Docker**: 컨테이너화
- **FastAPI**: API 서빙

### 데이터
- **ChromaDB**: 벡터 데이터베이스
- **Pandas**: 데이터 처리
- **Datasets**: Hugging Face 데이터셋

## 📖 문서

각 프로젝트별 상세 문서:
- [ChatDoc RAG README](./chatdoc-rag/README.md)
- [Sentiment-Ops README](./sentiment-Ops/README.md)

## 🔧 개발 환경

### VSCode 추천 확장

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "redhat.vscode-yaml",
    "ms-azuretools.vscode-docker"
  ]
}
```

### 코드 스타일

```bash
# Black formatter 설치
pip install black

# 코드 포맷팅
black .
```

## 🤝 기여

각 프로젝트에 기여하고 싶으시다면:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트들은 교육 및 연구 목적으로 제작되었습니다.

## 📧 연락처

버그 리포트 및 기능 제안은 GitHub Issues를 이용해주세요.

---

**Last Updated**: 2025-11-14
