def convert_ground_to_id(input_ground):
    ground_dict = {
        "札幌": "01",
        "函館": "02",
        "福島": "03",
        "新潟": "04",
        "東京": "05",
        "中山": "06",
        "中京": "07",
        "京都": "08",
        "阪神": "09",
        "小倉": "10"
    }
    
    return ground_dict.get(input_ground, "Unknown")  # 該当しない場合 "Unknown" を返す