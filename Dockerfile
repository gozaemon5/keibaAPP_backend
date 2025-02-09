# 1. Python 3.10 の公式イメージを使用
FROM python:3.10

# 2. 作業ディレクトリを設定
WORKDIR /app

# 3. 依存関係をコピー
COPY requirements.txt .

# 4. 必要なパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 5. 残りのアプリケーションファイルをコピー
COPY . .

# 6. Flask アプリを起動（ポート5000で起動）
EXPOSE 5000
CMD ["python", "app.py"]
