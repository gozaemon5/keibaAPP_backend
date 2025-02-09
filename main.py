from shutuba_table_main import shutuba_table_main
from preprocessing1 import RaceDataPreprocessor1
from predict_final_sectional_time import predict_final_sectional_time
from predict_time import predict_time
from predict_ranking import predict_ranking_proba
import pandas as pd

def main(input_date, input_race_number,input_ground ):
    try:        
        # 予測用データの読み込み
        input_data = shutuba_table_main(input_date, input_race_number, input_ground)
        print("出馬表読み込み完了")
    
        # データ前処理
        preprocessor = RaceDataPreprocessor1(is_train=False)
        processed_data = preprocessor.transform(input_data)
        # 不要な列を削除
        processed_data = processed_data.drop(columns=['race_id'])

        # 走破時間の予測（データには追加しない）
        predicted_time = predict_time(processed_data)
        print("走破時間予測完了")
    
        # 上がりの予測（走破時間なしのデータで実施）
        predicted_final_time = predict_final_sectional_time(processed_data)
        print("上がり予測完了")
    
        # 予測データを統合
        processed_data["走破時間"] = predicted_time
        processed_data["上がり"] = predicted_final_time
        print("走破時間・上がりのデータ結合完了")
    
        # 着順の予測（走破時間・上がりのデータを含めて実施）
        predicted_ranking = predict_ranking_proba(processed_data)
        print("予測完了")
    
        # 結果を出力
        processed_data["複勝確率"] = predicted_ranking
        # 「馬の名前」と「複勝確率」だけを抽出して表示
        results = pd.DataFrame({
        "馬": input_data["馬"],        # 馬の名前
        "馬番": input_data["馬番"],     # 馬番
        "人気": input_data["人気"],     # 人気
        "複勝確率": processed_data["複勝確率"]  # 複勝確率
        })

        # 複勝確率の高い順に並べ替え
        results = results.sort_values(by="複勝確率", ascending=False)

        # result_race_name = input_data["レース名"].iloc[0]
    
    
        return  results
    except Exception as e:
        print("mainメソッド内のエラー")
        print(f"Error in main function: {str(e)}")
        raise e
    

if __name__ == "__main__":
    main()

