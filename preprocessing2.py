import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import json
import os
import joblib
import numpy as np

class RaceDataPreprocessor2:
    def __init__(self, is_train=True,stats_file="model/horse_stats.json", jockey_stats_file="model/jockey_stats.json",
                 scaler_file="model/scaler.pkl", horse_encoder_file="model/horse_encoder.pkl", 
                 jockey_encoder_file="model/jockey_encoder.pkl", onehot_encoder_file="model/onehot_encoder.pkl"):
        self.is_train = is_train  # 学習時と予測時を区別するフラグ
        self.stats_file = stats_file
        self.jockey_stats_file = jockey_stats_file
        self.scaler_file = scaler_file
        self.horse_encoder_file = horse_encoder_file
        self.jockey_encoder_file = jockey_encoder_file
        self.onehot_encoder_file = onehot_encoder_file
        
        self.skip_horse_stats = os.path.exists(self.stats_file)
        self.skip_jockey_stats = os.path.exists(self.jockey_stats_file)
        
        self.horse_stats = self._load_horse_stats()
        self.jockey_stats = self._load_jockey_stats()
        
        if os.path.exists(self.scaler_file):
            self.scaler = joblib.load(self.scaler_file)
        else:
            self.scaler = StandardScaler()
        
        if os.path.exists(self.horse_encoder_file):
            self.horse_label_encoder = joblib.load(self.horse_encoder_file)
        else:
            self.horse_label_encoder = LabelEncoder()
        
        if os.path.exists(self.jockey_encoder_file):
            self.jockey_label_encoder = joblib.load(self.jockey_encoder_file)
        else:
            self.jockey_label_encoder = LabelEncoder()

        if os.path.exists(self.onehot_encoder_file):
            self.onehot_encoder = joblib.load(self.onehot_encoder_file)
        else:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    
    def transform(self, df):
        df = self._add_horse_features(df)
        df = self._add_jockey_features(df)
        df = self._common_preprocessing(df)
        df = self._transform_label_encoders(df)
        df = self._fit_onehot_encoder(df)
        df = self._scale_numeric(df)
        return df    


    def _compute_horse_stats(self, df):
        """各馬の過去成績を計算して辞書に保存（JSONファイルに保存）"""
        if self.skip_horse_stats:
            print("過去の馬の成績データが存在するため、スキップします。")
            return

        horse_groups = df.groupby("馬")
        for horse, group in horse_groups:
            if horse in self.horse_stats:
                # すでにデータが存在する場合はスキップ
                continue

            # 馬ごとの過去成績の計算
            avg_speed = (group["走破時間"] / group["距離"]).mean()  # 平均速度計算（走破時間 / 距離）

            self.horse_stats[horse] = {
                "平均着順": group["着順"].mean(),
                "勝率": (group["着順"] <= 3).mean(),  # 3着以内の割合
                "出走回数": len(group),
                "平均速度": avg_speed  # 馬ごとの平均速度を追加
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
        """馬ごとに過去レースの平均速度（走破時間 / 距離）を計算する"""
        # df["走破距離速度"] = df["走破時間"] / df["距離"]  # 1mあたりの走破時間

        """馬の過去成績を特徴量として追加（JSONファイルから読み込んだデータを使用）"""
        df["馬の平均着順"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("平均着順", 10))
        df["馬の勝率"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("勝率", 0))
        df["馬の出走回数"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("出走回数", 0))
        df["馬の平均速度"] = df["馬"].map(lambda x: self.horse_stats.get(x, {}).get("平均速度", 0))
        # df["馬の平均速度"] = df.groupby("馬")["走破距離速度"].transform(lambda x: x.expanding().mean().shift(1))

        return df
    
    def _compute_jockey_stats(self, df):
        """ 各騎手の過去成績を計算して辞書に保存（JSONファイルに保存） """
        if self.skip_jockey_stats:
            print("過去の騎手の成績データが存在するため、スキップします。")
            return

        jockey_groups = df.groupby("騎手")
        for jockey, group in jockey_groups:
            self.jockey_stats[jockey] = {
                "平均着順": group["着順"].mean(),
                "勝率": (group["着順"] <= 3).mean(),  # 3着以内の割合
                "出走回数": len(group)
            }
        self._save_jockey_stats()  # JSONファイルに保存

    def _save_jockey_stats(self):
        """ 騎手の成績情報をJSONファイルに保存 """
        with open(self.jockey_stats_file, "w", encoding="utf-8") as f:
            json.dump(self.jockey_stats, f, ensure_ascii=False, indent=4)

    def _load_jockey_stats(self):
        """ JSONファイルから騎手の成績情報を読み込む """
        if os.path.exists(self.jockey_stats_file):
            with open(self.jockey_stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def _add_jockey_features(self, df):
        """ 騎手の過去成績を特徴量として追加（JSONファイルから読み込んだデータを使用） """
        df["騎手の平均着順"] = df["騎手"].map(lambda x: self.jockey_stats.get(x, {}).get("平均着順", 10))
        df["騎手の勝率"] = df["騎手"].map(lambda x: self.jockey_stats.get(x, {}).get("勝率", 0))
        df["騎手の出走回数"] = df["騎手"].map(lambda x: self.jockey_stats.get(x, {}).get("出走回数", 0))
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
        

        # 不要な列を削除
        df = df.drop(columns=["レース名", "場id", "通過順", "開催"], errors="ignore")
        # 日本語の日付フォーマットを指定して変換
        df["日付"] = pd.to_datetime(df["日付"], format='%Y-%m-%d', errors='coerce')
        df["月"] = df["日付"].dt.month
        #df["曜日"] = df["日付"].dt.weekday

        # 走破時間の補正
        # df["走破時間"] = df["走破時間"].apply(self.convert_time_to_seconds)
        

        # 数値列を数値に変換（非数値はNaNに）
        numeric_cols = ["体重", "体重変化", "斤量", "距離", "人気", "オッズ"]
        for col in numeric_cols:
            df[col] = df[col].replace(r'[^\d.-]', '', regex=True)  # 数字以外を取り除く
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # "上がり" 列が存在する場合のみ処理
        if "上がり" in df.columns:
            df["上がり"] = df["上がり"].replace(r'[^\d.-]', '', regex=True)  # 数字以外を取り除く
            df["上がり"] = pd.to_numeric(df["上がり"], errors='coerce')
        
        # 距離区分を追加
        df["距離区分"] = pd.cut(df["距離"], 
                            bins=[0, 1200, 1600, 2000, float('inf')], 
                            labels=["短距離", "マイル", "中距離", "長距離"])
        
        df["距離"] = df["距離"]/1000

        df["人気差"] = df["人気"] - df.groupby("race_id")["人気"].transform("min")
        df = df.drop(columns=["人気"], errors="ignore")

        df = df.drop(columns=["日付"], errors="ignore")
        df["馬番"] = df["馬番"].astype(int)

        return df

    # 馬と騎手のラベルエンコーディングを行うメソッドを追加
    # 馬と騎手のラベルエンコーディングを行うメソッドを追加
    def _transform_label_encoders(self, df):
        if self.is_train:  # 学習時
            df["馬"] = self.horse_label_encoder.fit_transform(df["馬"])
            df["騎手"] = self.jockey_label_encoder.fit_transform(df["騎手"])
            # fit された classes_ 属性を表示して確認
            print("馬エンコーダーのクラス: ", self.horse_label_encoder.classes_)
            print("騎手エンコーダーのクラス: ", self.jockey_label_encoder.classes_)
        else:  # 予測時
            df["馬"] = df["馬"].map(lambda x: self.horse_label_encoder.transform([x])[0] 
                                    if x in self.horse_label_encoder.classes_ else -1)
            df["騎手"] = df["騎手"].map(lambda x: self.jockey_label_encoder.transform([x])[0] 
                                    if x in self.jockey_label_encoder.classes_ else -1)

        return df




    def _fit_onehot_encoder(self, df):
        # ワンホットエンコーディング
        categorical_cols = ["クラス", "天気", "馬場", "場名", "性", "芝・ダート", "回り","距離区分"]
        df_onehot = self.onehot_encoder.transform(df[categorical_cols])
        df_onehot = pd.DataFrame(df_onehot, columns=self.onehot_encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df, df_onehot], axis=1)
        df = df.drop(columns=categorical_cols)
        return df



    def _scale_numeric(self, df):
        df["騎手の出走回数"] = np.log1p(df["騎手の出走回数"])

        numeric_cols = ["体重", "体重変化", "斤量",  "オッズ", 
                    "馬の平均着順", "馬の出走回数","馬の勝率","馬の平均速度",
                    "騎手の平均着順","騎手の勝率","騎手の出走回数",  "走破時間", "距離"]

        # race_id ごとに標準化
        def standardize(group):
            return (group - group.mean()) / group.std(ddof=0)

        df[numeric_cols] = df.groupby("race_id")[numeric_cols].transform(standardize)

        return df


