import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# 모델 및 토크나이저 로드
model_path = "models/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 테스트 데이터 로드
train_df = pd.read_csv("data/train.csv")
# 평가용 데이터 (학습에 사용한 eval set과 동일)
eval_data = train_df[int(0.8 * len(train_df)):]

print("=" * 60)
print("📊 감성 분석 모델 평가 리포트")
print("=" * 60)
print(f"\n평가 데이터: {len(eval_data)}개 샘플\n")

# 예측
predictions = []
labels = eval_data["label"].tolist()

for text in eval_data["text"]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

# 성능 지표 계산
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
conf_matrix = confusion_matrix(labels, predictions)

print(f"✅ 정확도 (Accuracy): {accuracy:.2%}")
print(f"✅ 정밀도 (Precision): {precision:.2%}")
print(f"✅ 재현율 (Recall): {recall:.2%}")
print(f"✅ F1 스코어: {f1:.4f}")
print(f"\n혼동 행렬 (Confusion Matrix):")
print(conf_matrix)
print("\n[행: 실제 레이블, 열: 예측 레이블]")
print("[[TN, FP],")
print(" [FN, TP]]")

# 샘플별 예측 결과
print("\n" + "=" * 60)
print("📝 샘플별 예측 결과")
print("=" * 60)

for idx, (text, true_label, pred_label) in enumerate(zip(eval_data["text"], labels, predictions)):
    status = "✅" if true_label == pred_label else "❌"
    sentiment_true = "긍정" if true_label == 1 else "부정"
    sentiment_pred = "긍정" if pred_label == 1 else "부정"
    
    print(f"\n{status} 샘플 {idx + 1}:")
    print(f"   텍스트: {text}")
    print(f"   실제: {sentiment_true} (label={true_label})")
    print(f"   예측: {sentiment_pred} (label={pred_label})")

# 새로운 텍스트로 테스트
print("\n" + "=" * 60)
print("🧪 새로운 텍스트 테스트")
print("=" * 60)

test_texts = [
    "This movie is amazing!",
    "I hate this product.",
    "It's okay, nothing special.",
    "Best experience ever!",
    "Terrible service, very disappointed."
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred].item()
    
    sentiment = "긍정 😊" if pred == 1 else "부정 😞"
    print(f"\n텍스트: {text}")
    print(f"예측: {sentiment} (신뢰도: {confidence:.2%})")

print("\n" + "=" * 60)
print("평가 완료!")
print("=" * 60)
