#数据预处理

import os
import shutil
import cv2
import json
from collections import Counter
from tqdm import tqdm

# 设置数据路径
RAW_DATA_PATH = "../data/raw"         # 原始图片目录
PROCESSED_DATA_PATH = "../data/processed"  # 处理后的数据目录

# 确保处理后的数据目录存在
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

#获取指定目录下的图片
def get_image_files(directory):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    return [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

#检查并清理损坏的图片
def check_and_clean_images(directory):
    image_files = get_image_files(directory)
    valid_images = []
    for img_file in tqdm(image_files, desc="Checking images"):
        img_path = os.path.join(directory, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"删除损坏的图片: {img_path}")
                os.remove(img_path)  # 删除损坏的图片
            else:
                valid_images.append(img_file)
        except Exception as e:
            print(f"读取错误: {img_path}, 错误: {e}")
            os.remove(img_path)
    return valid_images

#按照类别整理图片
def organize_images():
    print("统计类别")
    categories = Counter()

    # 遍历所有图片
    for img_file in tqdm(os.listdir(RAW_DATA_PATH), desc="Processing images"):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue  # 跳过非图片文件
        
        # 通过文件名判断类别 (文件名格式: 类别_编号.jpg)
        category = img_file.split("_")[0]
        categories[category] += 1

        # 目标文件夹
        category_folder = os.path.join(PROCESSED_DATA_PATH, category)
        os.makedirs(category_folder, exist_ok=True)

        # 复制文件
        src_path = os.path.join(RAW_DATA_PATH, img_file)
        dst_path = os.path.join(category_folder, img_file)
        shutil.copy2(src_path, dst_path)

    # 保存类别统计结果
    stats_file = os.path.join(PROCESSED_DATA_PATH, "labels.json")
    with open(stats_file, "w") as f:
        json.dump(categories, f, indent=4)

    print(f"处理完成, 总类别: {len(categories)}，数据已整理到 {PROCESSED_DATA_PATH}")
    print(f"类别统计已保存: {stats_file}")

if __name__ == "__main__":
    print("开始数据预处理")
    valid_images = check_and_clean_images(RAW_DATA_PATH)
    organize_images()
    print("预处理完成")
