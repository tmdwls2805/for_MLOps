# ChatDoc RAG - 빠른 시작 가이드

## 1분 안에 시작하기

### Docker Compose 사용 (가장 간단)

```bash
# 1. 프로젝트 디렉토리로 이동
cd chatdoc-rag

# 2. Docker Compose로 빌드 및 실행
docker-compose up --build

# 3. 브라우저에서 접속
# http://localhost:8000/docs
```

### Docker 직접 사용

```bash
# 1. 이미지 빌드
docker build -t chatdoc .

# 2. 컨테이너 실행
docker run -p 8000:8000 --memory=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  chatdoc

# 3. 브라우저에서 접속
# http://localhost:8000/docs
```

## API 테스트하기

### Swagger UI 사용

1. `http://localhost:8000/docs` 접속
2. `/upload` 클릭 → "Try it out" → 파일 업로드
3. `/ask` 클릭 → "Try it out" → 질문 입력

### curl 사용

```bash
# 문서 업로드
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"

# 질문하기
curl -X POST "http://localhost:8000/ask" \
  -F "q=이 문서의 주요 내용은?"
```

## 문제 발생 시

### 메모리 부족
- Docker Desktop → Settings → Resources → Memory를 8GB 이상으로 설정

### 포트 충돌
- 다른 포트 사용: `docker run -p 9000:8000 ...`

### 모델 다운로드 느림
- 첫 실행 시 약 5.5GB 모델 다운로드 (시간 소요)
- 이후 실행은 캐시 사용으로 빠름

## 다음 단계

자세한 사용법은 [README.md](README.md)를 참고하세요.
