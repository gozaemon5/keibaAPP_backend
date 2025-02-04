from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from preprocessing import RaceDataPreprocessor

app = Flask(__name__)
CORS(app)

# 学習済みモデルとスケーラー、エンコーダーの読み込み
model = joblib.load("horse_race_model.pkl")
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
onehot_encoders = joblib.load('onehot_encoder.pkl')

# 前処理の準備
preprocessor = RaceDataPreprocessor()
preprocessor.scaler = scaler  # preprocessorのスケーラーを更新
preprocessor.label_encoders = label_encoders  # preprocessorのエンコーダーを更新
preprocessor.onehot_encoder = onehot_encoders  # preprocessorのワンホットエンコーダーを更新

@app.route('/process', methods=['POST'])
def predict():
    try:
        # JSON形式のデータを受け取る
        data = request.get_json()
        print("Received data:", data)  # 受け取ったデータをログに表示
        
        # JSONデータをDataFrameに変換
        df_predict = pd.DataFrame(data)
        
        # 前処理（予測時はtransform）
        df_predict_processed = preprocessor.transform(df_predict)
        
        # 予測
        X_predict = df_predict_processed.drop(columns=["race_id", "馬", "着順"])  # 必要な列だけを残す
        y_pred_prob = model.predict_proba(X_predict)[:, 1]  # 3着以内になる確率
        
        # 予測結果をデータフレームに追加
        df_predict["複勝確率"] = y_pred_prob
        
        # 確率が高い順にソート
        df_predict_sorted = df_predict.sort_values("複勝確率", ascending=False)
        
        # 結果をJSON形式で返す 
        result = df_predict_sorted[["馬", "複勝確率"]].to_dict(orient="records")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
