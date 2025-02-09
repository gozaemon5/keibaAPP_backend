from modules.preparing._scrape_race_id_list import scrape_race_id_list
from modules.constants._race_ground_from_name_to_id import convert_ground_to_id
from modules.preparing._scrape_shutuba_table import scrape_shutuba_table
import pandas as pd
import os

def shutuba_table_main(input_date, input_race_number, input_ground):

    input_date_year = input_date[:4]
    ground_id = convert_ground_to_id(input_ground)

    race_id_list = scrape_race_id_list([input_date]) #レースidを取得

    # 条件に合致する race_id を取得
    race_id = next((r for r in race_id_list if r.startswith(input_date_year + ground_id) and r.endswith(str(input_race_number))), None)

    scraping_data = scrape_shutuba_table(race_id,input_ground,input_date)

    # CSVファイルからカラムの順序を読み込む
    csv_file_path = os.path.join(os.path.dirname(__file__), 'model/column_order.csv')
    column_order_df = pd.read_csv(csv_file_path, header=0)
    column_order = column_order_df['column_name_input'].tolist()  # 必要なカラム順のリスト  # CSVの1列目をリスト化
    # nan を削除してカラムを並べ替え
    column_order = [col for col in column_order if pd.notna(col)]

    # 並び替えを実施（scraping_data に含まれるカラムのみを対象にする）
    scraping_data = scraping_data[column_order]

    return scraping_data

