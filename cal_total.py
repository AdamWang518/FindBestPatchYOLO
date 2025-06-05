import os
import re
from collections import defaultdict

# 設定根目錄
root_dir = r"D:\\Dockers\\split_COCO"
datasets = ['train', 'val']

# 儲存總圖片數與所有日期集合
total_images = {}
all_dates = set()

# 遍歷 train 和 val 資料夾
for ds in datasets:
    dataset_dir = os.path.join(root_dir, ds)
    image_count = 0

    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
                # 抓取開頭的4位數日期碼（例如 0811_ 開頭）
                match = re.match(r'(\d{4})_', file)
                if match:
                    all_dates.add(match.group(1))

    total_images[ds] = image_count

# 輸出結果
print("====== 統計結果 ======")
print(f"Train 圖片總數：{total_images.get('train', 0)} 張")
print(f"Val 圖片總數：{total_images.get('val', 0)} 張")
print(f"出現過的不同日期總數：{len(all_dates)}")
print(f"所有日期代碼：{sorted(all_dates)}")
