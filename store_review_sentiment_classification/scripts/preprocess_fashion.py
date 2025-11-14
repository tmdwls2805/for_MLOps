import json
import pandas as pd
from pathlib import Path

# data_result 폴더 생성 (없으면)
output_dir = Path("data_result")
output_dir.mkdir(exist_ok=True)

def process_json_files(data_dir, dataset_type):
    """재귀적으로 모든 JSON 파일을 찾아서 처리"""
    json_files = list(data_dir.rglob("*.json"))

    if not json_files:
        print(f"경고: {data_dir} 폴더에 JSON 파일이 없습니다.")
        return []

    print(f"\n{dataset_type} - 발견된 JSON 파일: {len(json_files)}개")

    all_rows = []

    for json_file in json_files:
        print(f"  처리 중: {json_file.relative_to(data_dir)}")

        try:
            # JSON 파일 로드
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # 각 리뷰 처리
            for d in raw_data:
                main_category = d.get("MainCategory", "Unknown")
                domain = d.get("Domain", "Unknown")
                raw_text = d.get("RawText", "")
                general_polarity = int(d.get("GeneralPolarity", 0))

                # Aspect별 감정 정보
                for asp in d.get("Aspects", []):
                    all_rows.append({
                        "text": asp.get("SentimentText", ""),
                        "aspect": asp.get("Aspect", ""),
                        "polarity": int(asp.get("SentimentPolarity", 0)),
                        "main_category": main_category,
                        "domain": domain,
                        "full_text": raw_text
                    })

                # 전체 문장 감정 추가
                all_rows.append({
                    "text": raw_text,
                    "aspect": "전체문장",
                    "polarity": general_polarity,
                    "main_category": main_category,
                    "domain": domain,
                    "full_text": raw_text
                })

        except Exception as e:
            print(f"    ⚠️ 오류 발생: {json_file.name} - {str(e)}")
            continue

    print(f"  ✅ 총 {len(all_rows)}개 세그먼트 처리 완료")
    return all_rows

# train_data 처리
train_dir = Path("data/train_data")
if train_dir.exists():
    train_rows = process_json_files(train_dir, "train_data")
    if train_rows:
        train_df = pd.DataFrame(train_rows)
        train_csv = output_dir / "train_data.csv"
        train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ {train_csv} 저장 완료 (총 {len(train_df)}개 행)")
else:
    print(f"경고: {train_dir} 폴더가 존재하지 않습니다.")

# test_data 처리
test_dir = Path("data/test_data")
if test_dir.exists():
    test_rows = process_json_files(test_dir, "test_data")
    if test_rows:
        test_df = pd.DataFrame(test_rows)
        test_csv = output_dir / "test_data.csv"
        test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
        print(f"✅ {test_csv} 저장 완료 (총 {len(test_df)}개 행)")
else:
    print(f"경고: {test_dir} 폴더가 존재하지 않습니다.")

print("\n✅ 전처리 완료: train_data.csv 및 test_data.csv가 data_result 폴더에 저장됨")
