# ChatDoc RAG - 문서 기반 질의응답 시스템

## 프로젝트 개요

ChatDoc RAG는 로컬 LLM(Large Language Model)을 활용한 문서 기반 질의응답 시스템입니다. 사용자가 업로드한 문서를 분석하고, 문서 내용에 기반하여 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- **문서 업로드**: PDF, TXT 등 다양한 형식의 문서 업로드 지원
- **벡터 검색**: ChromaDB를 활용한 효율적인 문서 검색
- **로컬 LLM**: Microsoft Phi-2 모델을 사용한 오프라인 질의응답
- **RESTful API**: FastAPI 기반의 간단하고 직관적인 API 제공
- **Swagger UI**: 내장된 API 문서화 및 테스트 인터페이스

## 기술 스택

### 백엔드 프레임워크
- **FastAPI**: 고성능 웹 프레임워크
- **Uvicorn**: ASGI 서버

### AI/ML
- **Transformers**: Hugging Face 트랜스포머 라이브러리
- **Microsoft Phi-2**: 로컬 LLM 모델
- **Sentence Transformers**: 문장 임베딩 생성
- **LangChain**: LLM 체인 및 RAG 구현

### 벡터 데이터베이스
- **ChromaDB**: 벡터 검색 및 저장

### 기타
- **PyTorch**: 딥러닝 프레임워크
- **Accelerate**: 모델 로딩 최적화

## 프로젝트 구조

```
chatdoc-rag/
├── app/
│   ├── main.py           # FastAPI 애플리케이션 진입점
│   ├── model.py          # LLM 모델 로딩 및 관리
│   ├── chains.py         # LangChain RAG 체인 구성
│   ├── vectorstore.py    # ChromaDB 벡터 저장소 관리
│   ├── loaders.py        # 문서 로더 (PDF, TXT 등)
│   └── requirements.txt  # Python 의존성 패키지
├── Dockerfile            # Docker 이미지 빌드 파일
└── README.md            # 프로젝트 문서
```

## 시스템 요구사항

### 하드웨어
- **RAM**: 최소 8GB 이상 권장
- **디스크**: 약 10GB 여유 공간 (모델 캐시 포함)
- **CPU/GPU**: CPU만으로 실행 가능 (GPU 사용 시 성능 향상)

### 소프트웨어
- **Docker**: 20.10 이상
- **Python**: 3.11 (로컬 실행 시)

## 설치 및 실행 방법

### 방법 1: Docker 사용 (권장)

#### 1. Docker 이미지 빌드

```bash
cd chatdoc-rag
docker build -t chatdoc .
```

#### 2. Docker 컨테이너 실행

```bash
docker run -p 8000:8000 \
  --memory=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  chatdoc
```

**옵션 설명:**
- `-p 8000:8000`: 포트 매핑 (호스트:컨테이너)
- `--memory=8g`: 메모리 할당 (최소 8GB 권장)
- `-v ~/.cache/huggingface:/root/.cache/huggingface`: 모델 캐시 볼륨 마운트 (재시작 시 다시 다운로드 방지)

### 방법 2: 로컬 환경 실행

#### 1. Python 가상환경 생성 (Python 3.11 사용)

```bash
# Python 3.11 설치 (Homebrew 사용)
brew install python@3.11

# 가상환경 생성
/opt/homebrew/bin/python3.11 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

#### 2. 의존성 패키지 설치

```bash
pip install --upgrade pip
pip install -r app/requirements.txt
```

#### 3. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API 사용 방법

### Swagger UI 사용 (권장)

1. 브라우저에서 `http://localhost:8000/docs` 접속
2. API 문서를 확인하고 직접 테스트 가능

### API 엔드포인트

#### 1. 루트 경로 - 서비스 정보

```bash
GET http://localhost:8000/
```

**응답 예시:**
```json
{
  "message": "Welcome to ChatDoc API",
  "endpoints": {
    "/docs": "API documentation (Swagger UI)",
    "/upload": "POST - Upload a document",
    "/ask": "POST - Ask a question about uploaded documents"
  }
}
```

#### 2. 문서 업로드

```bash
POST http://localhost:8000/upload
```

**요청 (multipart/form-data):**
- `file`: 업로드할 문서 파일 (PDF, TXT 등)

**curl 예시:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/document.pdf"
```

**응답 예시:**
```json
{
  "status": "upload",
  "file": "document.pdf"
}
```

#### 3. 질문하기

```bash
POST http://localhost:8000/ask
```

**요청 (application/x-www-form-urlencoded):**
- `q`: 질문 내용

**curl 예시:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -F "q=문서의 주요 내용은 무엇인가요?"
```

**응답 예시:**
```json
{
  "answer": "문서의 주요 내용은..."
}
```

## 사용 예시

### 1. Swagger UI를 통한 테스트

1. **문서 업로드:**
   - `http://localhost:8000/docs` 접속
   - `/upload` 섹션 클릭
   - "Try it out" 버튼 클릭
   - 파일 선택 후 "Execute" 클릭

2. **질문하기:**
   - `/ask` 섹션 클릭
   - "Try it out" 버튼 클릭
   - `q` 필드에 질문 입력
   - "Execute" 클릭

### 2. Python 코드 예시

```python
import requests

# 문서 업로드
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f}
    )
    print(response.json())

# 질문하기
response = requests.post(
    'http://localhost:8000/ask',
    data={'q': '문서의 주요 내용은?'}
)
print(response.json())
```

## 작동 원리

### RAG (Retrieval-Augmented Generation) 파이프라인

1. **문서 업로드 단계:**
   - 사용자가 문서를 업로드
   - 문서를 청크(chunk) 단위로 분할
   - Sentence Transformers를 사용하여 각 청크를 벡터로 변환
   - ChromaDB에 벡터 저장

2. **질문 처리 단계:**
   - 사용자 질문을 벡터로 변환
   - ChromaDB에서 관련성 높은 문서 청크 검색 (Retrieval)
   - 검색된 문서와 질문을 LLM에 전달
   - LLM이 문서 기반으로 답변 생성 (Generation)

### 사용 모델

- **LLM**: Microsoft Phi-2 (2.7B 파라미터)
  - 경량화된 모델로 CPU에서도 실행 가능
  - float16 정밀도로 메모리 사용량 최적화

- **Embedding**: Sentence Transformers 기본 모델
  - 문서와 질문을 벡터 공간으로 매핑
  - 의미적 유사도 계산

## 문제 해결

### Docker 메모리 부족

**증상:**
```
cannot allocate memory
```

**해결 방법:**
1. Docker Desktop → Settings → Resources
2. Memory를 8GB 이상으로 설정
3. 또는 실행 시 메모리 옵션 추가:
   ```bash
   docker run -p 8000:8000 --memory=8g chatdoc
   ```

### Python 버전 호환성 문제

**증상:**
```
ERROR: No matching distribution found for onnxruntime
```

**해결 방법:**
Python 3.11 사용 (Python 3.14는 일부 패키지와 호환되지 않음)

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
```

### 모델 다운로드 실패

**증상:**
```
Repository Not Found
```

**해결 방법:**
- 인터넷 연결 확인
- Hugging Face 접근 확인
- 모델 이름 철자 확인 (microsoft/phi-2)

## 향후 개선 계획

- [ ] GPU 지원 추가
- [ ] 더 많은 문서 형식 지원 (DOCX, PPTX 등)
- [ ] 채팅 히스토리 관리
- [ ] 멀티 문서 검색 최적화
- [ ] 스트리밍 응답 지원
- [ ] 사용자 인증 및 세션 관리

## 라이선스

이 프로젝트는 교육 및 개인 사용 목적으로 제작되었습니다.

## 기여

버그 리포트 및 기능 제안은 이슈로 등록해주세요.

## 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Microsoft Phi-2 Model](https://huggingface.co/microsoft/phi-2)
- [ChromaDB Documentation](https://docs.trychroma.com/)
