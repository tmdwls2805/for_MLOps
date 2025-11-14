from transformers import pipeline

pipe = pipeline("text-classification", model="models/fashion_final", tokenizer="models/fashion_final")

# 각 속성별로 감정 확인
print(pipe("가격이 착하고 [SEP] 가격"))
print(pipe("디자인이 예쁩니다 [SEP] 디자인"))
print(pipe("가격이 착하고 디자인이 예쁩니다 [SEP] 전체문장"))

print(pipe("가격이 비싸고 [SEP] 가격"))
print(pipe("디자인이 별로입니다 [SEP] 디자인"))
print(pipe("털빠짐때매 옷이아니네여"))