FROM python:3.10-slim
WORKDIR /app

COPY app/ ./app/
RUN pip install --no-cache-dir -r app/requirements.txt

# 모델은 첫 실행 시 자동으로 다운로드됩니다
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]