import pandas as pd
import json

def json_to_csv(json_data, output_csv_file):
    # 將 JSON 資料轉換為 pandas DataFrame
    df = pd.json_normalize(json_data)
    
    # 將 DataFrame 輸出為 CSV 檔案
    df.to_csv(output_csv_file, index=False)
    print(f"CSV 檔案已儲存為 {output_csv_file}")

# 讀取 JSON 檔案
with open('output.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 呼叫函數進行轉換
json_to_csv(json_data['result'], 'output_file.csv')
