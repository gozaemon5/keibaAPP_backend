import joblib

def predict_time(processed_data):
    # 学習済みモデルの読み込み
    model = joblib.load("model/horse_race_model_走破時間.pkl")

    # モデルにそのままデータを入力
    predicted_time = model.predict(processed_data)

    return predicted_time

