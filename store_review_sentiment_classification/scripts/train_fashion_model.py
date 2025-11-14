import pandas as pd
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import mlflow
from sklearn.preprocessing import LabelEncoder

# GPU/CPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("GPU를 사용할 수 없습니다. CPU로 학습합니다.")

# 1. 데이터 로드
train_df = pd.read_csv("data_result/train_data.csv")
test_df = pd.read_csv("data_result/test_data.csv")

print(f"학습 데이터: {len(train_df)}개")
print(f"테스트 데이터: {len(test_df)}개")
print(f"카테고리 종류: {train_df['main_category'].nunique()}개")
print(f"카테고리 목록: {train_df['main_category'].unique().tolist()}")

# 2. MainCategory 레이블 인코딩
label_encoder = LabelEncoder()
train_df["category_label"] = label_encoder.fit_transform(train_df["main_category"])
test_df["category_label"] = label_encoder.transform(test_df["main_category"])

# 레이블 매핑 저장
label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
with open("models/category_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)
print(f"\n레이블 매핑: {label_mapping}")

# 3. 입력 텍스트 생성: 문장 + [SEP] + 속성명
train_df["input_text"] = train_df.apply(lambda x: f"{x['text']} [SEP] {x['aspect']}", axis=1)
test_df["input_text"] = test_df.apply(lambda x: f"{x['text']} [SEP] {x['aspect']}", axis=1)

# 4. 토크나이저/모델 로드
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 5-1. 감정 분류 모델 학습 (polarity)
print("\n=== 감정 분류 모델 학습 중 ===")
sentiment_train = Dataset.from_pandas(train_df[["input_text", "polarity"]])
sentiment_test = Dataset.from_pandas(test_df[["input_text", "polarity"]])

sentiment_train = sentiment_train.rename_column("polarity", "labels")
sentiment_test = sentiment_test.rename_column("polarity", "labels")

sentiment_train = sentiment_train.map(tokenize, batched=True)
sentiment_test = sentiment_test.map(tokenize, batched=True)

sentiment_train = sentiment_train.remove_columns(["input_text"])
sentiment_test = sentiment_test.remove_columns(["input_text"])

sentiment_train.set_format("torch")
sentiment_test.set_format("torch")

sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

sentiment_args = TrainingArguments(
    output_dir="./models/sentiment",
    evaluation_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=100,
    num_train_epochs=100,
    logging_dir="./logs/sentiment",
    learning_rate=5e-5,
    no_cuda=not torch.cuda.is_available(),  # GPU 없으면 CPU 사용
    use_cpu=not torch.cuda.is_available(),  # 명시적 CPU 사용 설정
)

sentiment_trainer = Trainer(
    model=sentiment_model,
    args=sentiment_args,
    train_dataset=sentiment_train,
    eval_dataset=sentiment_test,
)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fashion-sentiment")

with mlflow.start_run(run_name="sentiment_classification"):
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_train_samples", len(train_df))
    mlflow.log_param("task", "sentiment")
    sentiment_trainer.train()
    sentiment_model.save_pretrained("models/sentiment_final")
    tokenizer.save_pretrained("models/sentiment_final")

print("✅ 감정 분류 모델 학습 완료: models/sentiment_final")

# 5-2. 카테고리 분류 모델 학습 (main_category)
print("\n=== 카테고리 분류 모델 학습 중 ===")
num_categories = len(label_encoder.classes_)
print(f"카테고리 수: {num_categories}")

category_train = Dataset.from_pandas(train_df[["input_text", "category_label"]])
category_test = Dataset.from_pandas(test_df[["input_text", "category_label"]])

category_train = category_train.rename_column("category_label", "labels")
category_test = category_test.rename_column("category_label", "labels")

category_train = category_train.map(tokenize, batched=True)
category_test = category_test.map(tokenize, batched=True)

category_train = category_train.remove_columns(["input_text"])
category_test = category_test.remove_columns(["input_text"])

category_train.set_format("torch")
category_test.set_format("torch")

category_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_categories
)

category_args = TrainingArguments(
    output_dir="./models/category",
    evaluation_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs/category",
    learning_rate=5e-5,
    no_cuda=not torch.cuda.is_available(),  # GPU 없으면 CPU 사용
    use_cpu=not torch.cuda.is_available(),  # 명시적 CPU 사용 설정
)

category_trainer = Trainer(
    model=category_model,
    args=category_args,
    train_dataset=category_train,
    eval_dataset=category_test,
)

with mlflow.start_run(run_name="category_classification"):
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_train_samples", len(train_df))
    mlflow.log_param("num_categories", num_categories)
    mlflow.log_param("task", "category")
    category_trainer.train()
    category_model.save_pretrained("models/category_final")
    tokenizer.save_pretrained("models/category_final")

print("✅ 카테고리 분류 모델 학습 완료: models/category_final")
print("\n✅ 모든 모델 학습 완료!")
