import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import mlflow
import torch

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("sentiment-analysis")

train_df = pd.read_csv("data/train.csv")

# Split dataset into train and eval
train_size = int(0.8 * len(train_df))
train_data = train_df[:train_size]
eval_data = train_df[train_size:]

train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

trainging_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=trainging_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

with mlflow.start_run():
    mlflow.log_param("model_name", model_name)
    trainer.train()
    mlflow.log_artifacts("models")
    model.save_pretrained("models/final")
    tokenizer.save_pretrained("models/final")
