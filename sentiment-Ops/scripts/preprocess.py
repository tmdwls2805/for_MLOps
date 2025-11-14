import pandas as pd
from sklearn.model_selection import train_test_split


data = {
    "text": [
        "오늘 날씨가 정말 좋네요!",
        "기분이 너무 우울하다...",
        "맛있는 점심 먹고 행복했어요.",
        "회의가 너무 길어서 피곤해요.",
        "좋은 하루 보내세요!",
    ],
    "label": [1, 0, 1, 0, 1]  # 1=긍정, 0=부정
}

df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print("데이터 전처리 완료")