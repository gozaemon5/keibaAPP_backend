import joblib
import xgboost as xgb
import pandas as pd

def predict_ranking_proba(input_data, processed_data):
    # 学習済みモデルの読み込み
    model = joblib.load("model/horse_race_model.pkl")

    # CSVファイルからカラム順を読み込む
    order_file=r"model/column_order.csv"
    order_df = pd.read_csv(order_file)  # このファイルにカラム順が書かれていると仮定
    print('並び替え順のcsvファイル読み込み完了')
    order = order_df['column_name'].tolist()  # 必要なカラム順のリスト

    # データの並べ替え
    processed_data = processed_data[order]  # カラム順を合わせる
    
    # DMatrixに変換
    dtest = xgb.DMatrix(processed_data)

    # 予測確率（3着以内に入る確率）
    y_pred_proba = model.predict(dtest, output_margin=False)

    # 予測結果を processed_data に追加
    processed_data["pred_proba"] = y_pred_proba
    
    # `input_data` から `race_id` を取得し、`processed_data` に追加
    if "race_id" not in input_data.columns:
        raise ValueError("input_data に 'race_id' が存在しません")

    processed_data["race_id"] = input_data["race_id"].values  # 同じ順序で race_id を割り当てる

    '''
    # 各レースIDごとに確率を正規化
    processed_data["normalized_pred_proba"] = processed_data.groupby("race_id")["pred_proba"].transform(
    lambda x: x / x.sum()
    )
    '''  
    return processed_data
    
