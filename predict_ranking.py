import joblib
import xgboost as xgb
import pandas as pd

def predict_ranking_proba(processed_data, column_order_file="model/column_order.csv"):
    # 学習済みモデルの読み込み
    model = joblib.load("model/horse_race_model.pkl")

    # CSVファイルからカラム順を読み込む
    column_order_df = pd.read_csv(column_order_file)  # このファイルにカラム順が書かれていると仮定
    columns_order = column_order_df['column_name'].tolist()  # 必要なカラム順のリスト

    # データの並べ替え
    processed_data = processed_data[columns_order]  # カラム順を合わせる

    # DMatrixに変換
    dtest = xgb.DMatrix(processed_data)

    # 予測確率（3着以内に入る確率）
    y_pred_proba = model.predict(dtest, output_margin=False)

    return y_pred_proba

