import joblib

def predict_final_sectional_time(processed_data):
    # 学習済みモデルの読み込み
    model = joblib.load("model/horse_race_model_上がり.pkl")

    # モデルにそのままデータを入力
    predicted_final_sectional_time = model.predict(processed_data)

    return predicted_final_sectional_time

