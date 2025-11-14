import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import mlflow

# 1. 데이터 로드
df = pd.read_csv("data/fashion_segments.csv")

# 입력 텍스트 생성: 문장 + [SEP] + 속성명
df["input_text"] = df.apply(lambda x: f"{x['text']} [SEP] {x['aspect']}", axis=1)

# 2. 데이터셋 변환
dataset = Dataset.from_pandas(df)
dataset = dataset.rename_column("polarity", "labels")

# 3. 토크나이저/모델 로드
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text", "aspect", "input_text"])
dataset.set_format("torch")

# 4. 학습/검증 분리
train_test = dataset.train_test_split(test_size=0.2, seed=42)

# 5. 모델 준비
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Trainer 설정
training_args = TrainingArguments(
    output_dir="./models/fashion",
    evaluation_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
)

# 7. 학습 실행
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fashion-sentiment")

with mlflow.start_run():
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_samples", len(df))
    trainer.train()
    model.save_pretrained("models/fashion_final")
    tokenizer.save_pretrained("models/fashion_final")

print("✅ 모델 학습 완료: models/fashion_final 저장됨")
