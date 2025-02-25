from shutuba_table_main import shutuba_table_main
from preprocessing1 import RaceDataPreprocessor1
from predict_time import predict_time
from preprocessing2 import RaceDataPreprocessor2
from predict_ranking import predict_ranking_proba
from modules.constants._race_ground_from_name_to_id import convert_ground_to_id
import pandas as pd
import json
import os
import glob

def predict_main(input_date, input_race_number, input_ground):
    # input_date = '20241020'
    # input_race_number = 11
    # input_ground = '京都'

    '''
    # 予測用データの読み込み
    input_data = shutuba_table_main(input_date, input_race_number, input_ground)
    print("出馬表読み込み完了")
    '''
    ground_id = convert_ground_to_id(input_ground)
    print(ground_id)
    folder_path = os.path.join("出馬表データ",input_date + input_ground)
    print(folder_path)
    # ファイルパターンを指定
    file_pattern = os.path.join(folder_path, f"{input_date}{ground_id}{input_race_number}*.json")
    print(file_pattern)
    
    # 該当するファイルを検索
    json_files = glob.glob(file_pattern)
    

    if not json_files:
        print("No matching file found.")
        return None
    
    file_path = json_files[0]

    # JSONファイルを読み込む
    with open(file_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    input_data = pd.DataFrame(input_data)

    # データ前処理
    preprocessor1 = RaceDataPreprocessor1(is_train=False)
    processed_data = preprocessor1.transform(input_data)
    # 不要な列を削除
    processed_data = processed_data.drop(columns=['race_id'])

    # 走破時間の予測（データには追加しない）
    predicted_time = predict_time(processed_data)
    print("走破時間予測完了")
    '''
    # 馬名と予測した走破時間を並べて表示
    for horse, time in zip(input_data["馬"], predicted_time):
        print(f"馬: {horse}, 予測走破時間: {time}")
    '''
    
    
    # 予測データを統合
    input_data["走破時間"] = predicted_time
    print("走破時間のデータ結合完了")

    preprocessor2 = RaceDataPreprocessor2(is_train=False)
    processed_data2 = preprocessor2.transform(input_data)

    
    # 着順の予測（走破時間のデータを含めて実施）
    predicted_ranking = predict_ranking_proba(input_data, processed_data2)
    print("予測完了")
    
    # 結果を出力
    processed_data2["複勝確率"] = predicted_ranking["normalized_pred_proba"]
    # 「馬の名前」と「複勝確率」だけを抽出して表示
    results = pd.DataFrame({
    # "race_id": input_data["race_id"] ,  
    "馬": input_data["馬"],        # 馬の名前
    "馬番": input_data["馬番"],     # 馬番
    "人気": input_data["人気"],     # 人気
    "予想複勝確率": processed_data2["複勝確率"],  # 複勝確率
    "レース名":input_data["レース名"]
    })
    
    # 複勝確率の高い順に並べ替え
    results = results.sort_values(by="複勝確率", ascending=False)
    
    # result_race_name = input_data["レース名"].iloc[0]
    print(results)
    
    return results
    
    



