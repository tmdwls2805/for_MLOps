import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from pathlib import Path


# data_result 폴더 생성 (없으면)
output_dir = Path("data_result")
output_dir.mkdir(exist_ok=True)

# data 폴더의 모든 JSON 파일 찾기
data_dir = Path("data")
json_files = list(data_dir.glob("*.json"))

if not json_files:
    print("경고: data 폴더에 JSON 파일이 없습니다.")
else:
    print(f"발견된 JSON 파일: {len(json_files)}개")

    for json_file in json_files:
        print(f"\n처리 중: {json_file.name}")

        # JSON 파일 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # DataFrame으로 변환
        df = pd.DataFrame(data)

        # 학습/테스트 분리
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # 파일명 생성 (원본 파일명 기반)
        base_name = json_file.stem
        train_csv = output_dir / f"{base_name}_train.csv"
        test_csv = output_dir / f"{base_name}_test.csv"

        # CSV 저장
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print(f"  ✅ {train_csv.name} 저장 완료 ({len(train_df)}개 샘플)")
        print(f"  ✅ {test_csv.name} 저장 완료 ({len(test_df)}개 샘플)")

print("\n데이터 전처리 완료")