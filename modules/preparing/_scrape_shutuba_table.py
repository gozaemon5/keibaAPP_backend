import re
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ._prepare_chrome_driver import prepare_chrome_driver

# 仕様変更対策
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]

def scrape_shutuba_table(race_id: str, input_ground: str, input_date: str):
    """
    当日の出馬表をスクレイピング。
    dateはyyyy/mm/ddの形式。
    """
    driver = prepare_chrome_driver()
    # 取得し終わらないうちに先に進んでしまうのを防ぐため、暗黙的な待機（デフォルト10秒）
    driver.implicitly_wait(10)
    query = '?race_id=' + race_id
    url = 'https://race.netkeiba.com/race/shutuba.html' + query
    df = pd.DataFrame()
    
    
    driver.get(url)
    # 現在のウィンドウハンドル（元のタブ）を保存
    original_handle = driver.current_window_handle

    rows = []  # 全体の行データを格納するリスト

    # メインのテーブルの取得
    # メインのテーブルの取得
    for tr in driver.find_elements(By.CLASS_NAME, 'HorseList'):

        row = [None] * 10  # 10列分の空のリストを事前に作成
        horse_name = None
        umaban = None
        sex = None
        age = None
        weight = None
        horse_weight = None
        horse_weight_change = None
        jockey_full_name = None
        odds = None
        popularity = None

        for td in tr.find_elements(By.TAG_NAME, 'td'):
            # HorseInfoクラスのセルから馬の名前を取得
            if td.get_attribute('class') == 'HorseInfo':
                horse_name = td.find_element(By.CLASS_NAME, 'HorseName').find_element(By.TAG_NAME, 'a').text
                row[0] = horse_name  # 馬の名前

            # 馬番の情報がある<td>を取得
            elif re.search(r'Umaban\d+ Txt_C', td.get_attribute('class')):  # 直接クラス名を取得して比較
                umaban = td.text  # 馬番を取得
                row[1] = umaban  # 馬番
                        
            # 性別・年齢の情報がある<td>を取得
            elif td.get_attribute('class') == 'Barei Txt_C':
                sex_age = td.text  # 例えば「牡3」
                sex, age = sex_age[0], int(sex_age[1:])  # 性別と年齢を分割
                row[2] = sex  # 性別
                row[3] = age  # 年齢
        
            # 斤量の情報がある<td>を取得
            elif td.get_attribute('class') == 'Txt_C' and td.text.replace('.', '').isdigit():
                weight = float(td.text)  # 斤量を数値化
                row[4] = weight  # 斤量
        
            # 体重と体重変化の情報がある<td>を取得
            elif td.get_attribute('class') == 'Weight':
                weight_text = td.text.split('(')[0].strip()  # 体重（数字部分）を取得
                weight_change_text = td.find_element(By.TAG_NAME, 'small').text.strip('()')  # 体重変化を取得
            
                horse_weight = int(weight_text)  # 体重（数字部分）
                if weight_change_text == '前計不':
                    horse_weight_change = 0  # 0 にする場合
                else:
                    horse_weight_change = int(weight_change_text)
            
            
                row[5] = horse_weight  # 体重
                row[6] = horse_weight_change  # 体重変化
            # オッズの情報がある<td>を取得
            elif td.get_attribute('class') == 'Txt_R Popular':
                odds = td.find_element(By.TAG_NAME, 'span').text  # オッズを取得
                row[8] = odds  # オッズ
            
            # 人気の情報がある<td>を取得
            elif td.get_attribute('class') == 'Popular Popular_Ninki Txt_C':
                
                popularity = td.find_element(By.TAG_NAME, 'span').text  # 人気を取得
                row[9] = popularity  # 人気
            
            # 騎手名の情報がある<td>を取得
            elif td.get_attribute('class') == 'Jockey':
                # 騎手のリンクを取得
                jockey_link = td.find_element(By.TAG_NAME, 'a').get_attribute('href')
            
                # 新しいタブを開く
                driver.execute_script(f'window.open("{jockey_link}");')

                # 新しいタブに切り替え
                new_handle = [handle for handle in driver.window_handles if handle != original_handle][0]
                driver.switch_to.window(new_handle)

                # ページがロードされるまで待機
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'title')))

                # フルネームを取得
                title = driver.title
                jockey_full_name = title.split("の近走成績")[0]  # 名前を抽出

                # フルネームを行に追加
                row[7] = jockey_full_name  # 騎手名

                # 新しいタブを閉じる
                driver.close()

                # 元のタブに戻る
                driver.switch_to.window(original_handle)
                WebDriverWait(driver, 10).until(
                lambda driver: driver.execute_script('return document.readyState') == 'complete'
                )
            

        # 各行のデータが完成したら、リストに追加
        rows.append(row)
        
    # ループ終了後、一度だけDataFrameに変換
    df_rows = pd.DataFrame(rows, columns=['馬','馬番', '性', '齢', '斤量', '体重', '体重変化', '騎手', 'オッズ', '人気'])
    

    # レース情報の取得
    texts = driver.find_element(By.CLASS_NAME, 'RaceData01').text

    # 正規表現を使って必要な情報を抽出
    track_type = re.search(r'(芝|ダート)', texts)
    track_type = track_type.group(0) if track_type else None

    distance = re.search(r'(\d+)(m)', texts)
    distance = distance.group(1) if distance else None

    direction = re.search(r'(右|左|外)', texts)
    direction = direction.group(0) if direction else None

    weather = re.search(r'天候:(\w+)', texts)
    weather = weather.group(1) if weather else None

    ground_condition = re.search(r'馬場:(\w+)', texts)
    ground_condition = ground_condition.group(1) if ground_condition else None

    # RaceData02クラス内の情報を取得
    class_info = driver.find_element(By.CLASS_NAME, 'RaceData02')

    # <span>タグの内容をすべて取得
    span_elements = class_info.find_elements(By.TAG_NAME, 'span')

    # 「サラ系」が含まれていればそれを取り除く
    race_class = span_elements[3].text  # サラ系３歳がある部分
    if 'サラ系' in race_class:
        race_class = race_class.replace('サラ系', '').strip()  # サラ系を取り除く

    # 「オープン」の部分を取得
    race_class += span_elements[4].text  # オープンを追加

    # レース名を取得
    racelist_item = driver.find_element(By.CLASS_NAME, 'RaceList_Item02')
    if racelist_item:
        race_name = racelist_item.find_element(By.CLASS_NAME, 'RaceName').text.strip()
        
    else:
        print("RaceList_Item02 が見つかりません")

    # 日付の形式を変換
    input_date = input_date[:4] + "-" + input_date[4:6] + "-" + input_date[6:8]

    # レース情報をリストにまとめる
    race_info = [race_id, track_type, distance, direction, weather, ground_condition,race_class, input_ground, race_name, input_date]
    
    # 出走馬の数（dfの行数）に合わせて繰り返す
    race_info_repeated = [race_info] * len(df_rows)

    # 繰り返した情報をdfに新しい列として追加
    race_info_df = pd.DataFrame(race_info_repeated, columns=["race_id","芝・ダート", "距離", "回り", "天気", "馬場", "クラス","場名","レース名","日付"])
    df = pd.concat([df_rows, race_info_df], axis=1)
    
    df = df.dropna(subset=['馬'])
    df = df.reset_index(drop=True)  # インデックスをリセット
    print(df)
   
    return df

