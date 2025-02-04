import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import json
import os

class RaceDataPreprocessor:
    def __init__(self, stats_file="horse_stats.json", scaler=None):
        self.scaler = scaler if scaler else StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.stats_file = stats_file
        self.horse_stats = self._load_horse_stats()  # 過去の馬のデータをロード

    def fit_transform(self, df):
        df = self._common_preprocessing(df)  # 共通の前処理
        df = self._process_ranking(df)  # 着順の処理を追加
        self._compute_horse_stats(df)  # 馬の過去成績を計算
        df = self._add_horse_features(df)  # 馬の過去成績を特徴量として追加
        df = self._fit_encoders(df)
        df = self._fit_onehot_encoder(df)
        df = self._scale_numeric(df)  # 標準化
        return df

    def transform(self, df):
        # 予測時の前処理
        df = self._common_preprocessing(df)
        df = self._add_horse_features(df)  # 馬の過去成績を特徴量として追加
        df = self._apply_encoders(df)
        df = self._apply_onehot_encoder(df)
        df = self._scale_numeric(df, fit=False)  # 数値データの標準化
        return df

    def _compute_horse_stats(self, df):
        """ 各馬の過去成績を計算して辞書に保存（JSONファイルに保存） """
        horse_groups = df.groupby("馬")
        for horse, group in horse_groups:
            self.horse_stats[horse] = {
                "平均人気": group["人気"].mean(),
                "平均着順": group["着順"].mean(),
                "勝率": (group["着順"] <= 3).mean(),  # 3着以内の割合
                "出走回数": len(group)
            }
        self._save_horse_stats()  # JSONファイルに保存
    
    def _save_horse_stats(self):
        """ 馬の成績情報をJSONファイルに保存 """
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.horse_stats, f, ensure_ascii=False, indent=4)

    def _load_horse_stats(self):
        """ JSONファイルから馬の成績情報を読み込む """
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}    

    def _add_horse_features(self, df):
        """ 馬の過去成績を特徴量として追加（JSONファイルから読み込んだデータを使用） """
        df["馬の平均人気"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("平均人気", 0))
        df["馬の平均着順"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("平均着順", 10))
        df["馬の勝率"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("勝率", 0))
        df["馬の出走回数"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("出走回数", 0))
        return df

    @staticmethod
    def convert_time_to_seconds(time_str):
        # 時間形式が"HHMM"であれば、'1415'を'14:15'に変換して処理
        if isinstance(time_str, str) and len(time_str) == 4 and time_str.isdigit():
            try:
                hours = int(time_str[:2])
                minutes = int(time_str[2:])
                return hours * 60 + minutes
            except ValueError:
                return None
        elif isinstance(time_str, str):
            try:
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            except ValueError:
                return None
        return None

    def _common_preprocessing(self, df):
        df = df.copy()

        # 不要な列を削除
        df = df.drop(columns=["レース名", "場名", "通過順", "開催"], errors="ignore")
        # 日本語の日付フォーマットを指定して変換
        df["日付"] = pd.to_datetime(df["日付"], format='%Y-%m-%d', errors='coerce')
        df["月"] = df["日付"].dt.month
        df["曜日"] = df["日付"].dt.weekday

        # 走破時間の変換
        df['走破時間'] = df['走破時間'].apply(self.convert_time_to_seconds)

        # 数値列を数値に変換（非数値はNaNに）
        numeric_cols = ["体重", "体重変化", "斤量", "距離", "人気", "オッズ"]
        for col in numeric_cols:
            df[col] = df[col].replace(r'[^\d.-]', '', regex=True)  # 数字以外を取り除く
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # "上がり" 列が存在する場合のみ処理
        if "上がり" in df.columns:
            df["上がり"] = df["上がり"].replace(r'[^\d.-]', '', regex=True)  # 数字以外を取り除く
            df["上がり"] = pd.to_numeric(df["上がり"], errors='coerce')

        df = df.drop(columns=["日付"], errors="ignore")

        return df


    def _fit_encoders(self, df):
        # ラベルエンコーディング（馬、騎手）
        categorical_cols = ["馬", "騎手"]
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def _apply_encoders(self, df):
        # 予測時にラベルエンコーディング
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        return df

    def _fit_onehot_encoder(self, df):
        # ワンホットエンコーディング
        categorical_cols = ["クラス", "天気", "馬場", "場id", "性", "芝・ダート", "回り"]
        df_onehot = self.onehot_encoder.fit_transform(df[categorical_cols])
        df_onehot = pd.DataFrame(df_onehot, columns=self.onehot_encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df, df_onehot], axis=1)
        df = df.drop(columns=categorical_cols)
        return df

    def _apply_onehot_encoder(self, df):
        # 予測時にワンホットエンコーディング
        categorical_cols = ["クラス", "天気", "馬場", "場id", "性", "芝・ダート", "回り"]
        df_onehot = self.onehot_encoder.transform(df[categorical_cols])
        df_onehot = pd.DataFrame(df_onehot, columns=self.onehot_encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df, df_onehot], axis=1)
        df = df.drop(columns=categorical_cols)
        return df

    def _scale_numeric(self, df, fit=True):
        numeric_cols = ["体重", "体重変化", "斤量", "距離", "人気", "オッズ",
                        "馬の平均人気", "馬の平均着順", "馬の勝率", "馬の出走回数"]

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def _process_ranking(self, df):
        df["着順"] = df["着順"].replace({"除": pd.NA, "中": pd.NA, "取": pd.NA, "失": pd.NA})
        df["着順"] = df["着順"].str.replace(r'\(降\)', '', regex=True)
        df["着順"] = pd.to_numeric(df["着順"], errors='coerce')
        # 着順に基づいて3着以内の列を作成
        df["3着以内"] = df["着順"].apply(lambda x: 1 if pd.notna(x) and x <= 3 else 0)
        df = df.dropna(subset=["着順"])
        return df
