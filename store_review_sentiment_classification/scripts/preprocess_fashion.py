import json
import pandas as pd

# 1. 원본 JSON 로드
with open("data/1-1.여성의류(1).json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

normalized = []

# 2. Aspect별 세그먼트 + 전체 문장 포함
for d in raw_data:
    entry = {"RawText": d["RawText"], "Segments": []}
    
    # Aspect별 감정 정보
    for asp in d.get("Aspects", []):
        entry["Segments"].append({
            "Text": asp["SentimentText"],
            "Aspect": asp["Aspect"],
            "Polarity": int(asp["SentimentPolarity"])
        })

    # 전체 문장 감정 추가
    general = int(d.get("GeneralPolarity", 0))
    entry["Segments"].append({
        "Text": d["RawText"],
        "Aspect": "전체문장",
        "Polarity": general
    })

    normalized.append(entry)

# 3. JSON 저장
with open("data/preprocessed_fashion.json", "w", encoding="utf-8") as f:
    json.dump(normalized, f, ensure_ascii=False, indent=2)

# 4. CSV 변환
rows = []
for d in normalized:
    for seg in d["Segments"]:
        rows.append({
            "text": seg["Text"],
            "aspect": seg["Aspect"],
            "polarity": seg["Polarity"]
        })

df = pd.DataFrame(rows)
df.to_csv("data/fashion_segments.csv", index=False)

print("✅ 전처리 완료: data/fashion_segments.csv 생성됨")
