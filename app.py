from flask import Flask, request, jsonify
from flask_cors import CORS
from main import predict_main
app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def predict():
    try:
        # JSON形式のデータを受け取る
        data = request.get_json()
        print("Received data:", data)  # 受け取ったデータをログに表示

        # 必要なデータを抽出
        input_date = data.get('input_date')
        input_race_number = data.get('input_race_number')
        input_ground = data.get('input_ground')

        # main関数を直接呼び出す
        results = predict_main(input_date, input_race_number, input_ground)
        print("main処理完了")

        # DataFrameを辞書に変換
        result = results.to_dict(orient='records')
        # 結果をJSONとして返す
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)