import os
import shutil
import cv2
import math
from pathlib import Path
from tqdm import tqdm

def process_image_and_label(img_path, label_path, output_dir, crop_w, crop_h):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠ 無法讀取圖像：{img_path}")
        return
    img_h, img_w = img.shape[:2]

    with open(label_path, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]

    rows = math.ceil(img_h / crop_h)
    cols = math.ceil(img_w / crop_w)

    for i in range(rows):
        for j in range(cols):
            x0, y0 = j * crop_w, i * crop_h
            x1 = min(x0 + crop_w, img_w)
            y1 = min(y0 + crop_h, img_h)

            crop = img[y0:y1, x0:x1]
            crop_h_actual = y1 - y0
            crop_w_actual = x1 - x0

            crop_name = f"{img_path.stem}_r{i}_c{j}.jpg"
            crop_path = output_dir / "images" / crop_name
            cv2.imwrite(str(crop_path), crop)

            crop_label_path = output_dir / "labels" / f"{img_path.stem}_r{i}_c{j}.txt"
            with open(crop_label_path, 'w') as out_lbl:
                for ann in annotations:
                    cls_id, x, y, w, h = int(ann[0]), *map(float, ann[1:])
                    x_abs, y_abs = x * img_w, y * img_h
                    w_abs, h_abs = w * img_w, h * img_h

                    xmin = x_abs - w_abs / 2
                    ymin = y_abs - h_abs / 2
                    xmax = x_abs + w_abs / 2
                    ymax = y_abs + h_abs / 2

                    if (xmin < x1 and xmax > x0 and
                        ymin < y1 and ymax > y0):

                        nxmin = max(xmin, x0) - x0
                        nymin = max(ymin, y0) - y0
                        nxmax = min(xmax, x1) - x0
                        nymax = min(ymax, y1) - y0

                        bw = nxmax - nxmin
                        bh = nymax - nymin

                        if bw <= 1e-3 or bh <= 1e-3:
                            continue

                        cx = (nxmin + nxmax) / 2 / crop_w_actual
                        cy = (nymin + nymax) / 2 / crop_h_actual
                        bw /= crop_w_actual
                        bh /= crop_h_actual

                        out_lbl.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def generate_patch_datasets(full_dataset_root, patch_sizes, output_root):
    full_dataset_root = Path(full_dataset_root)
    output_root = Path(output_root)
    assert (full_dataset_root / "train/images").exists(), "train/images 不存在"

    for size in patch_sizes:
        print(f"\n🔧 建立 Patch Size: {size}x{size}")
        for split in ["train", "val"]:
            img_dir = full_dataset_root / split / "images"
            lbl_dir = full_dataset_root / split / "labels"
            out_dir = output_root / f"patch_{size}" / split

            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / "labels").mkdir(parents=True, exist_ok=True)

            img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png']]
            for img_path in tqdm(img_files, desc=f"{split} patch {size}", ncols=100):
                label_path = lbl_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    print(f"❌ 缺少標註檔：{label_path}")
                    continue
                process_image_and_label(img_path, label_path, out_dir, size, size)

    print("\n✅ 所有 patch 資料集處理完畢。")
def copy_original_val(full_dataset_root, output_root):
    """保留一份未切割的 val 資料集，供 full model 比對使用"""
    src_img_dir = Path(full_dataset_root) / "val/images"
    src_lbl_dir = Path(full_dataset_root) / "val/labels"
    dst_img_dir = Path(output_root) / "original_val/images"
    dst_lbl_dir = Path(output_root) / "original_val/labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_file in src_img_dir.glob("*.[jp][pn]g"):
        label_file = src_lbl_dir / (img_file.stem + ".txt")
        dst_img = dst_img_dir / img_file.name
        dst_lbl = dst_lbl_dir / label_file.name
        shutil.copy(img_file, dst_img)
        if label_file.exists():
            shutil.copy(label_file, dst_lbl)

    print(f"📦 已複製原始 val 至 {dst_img_dir.parent}")

# 使用範例
if __name__ == "__main__":
    full_dataset_root = r"../RandomPick_v6_6_Full"
    patch_sizes = [320, 480, 640, 800, 960]
    output_root = r"../RandomPick_v6_Train_Patched"

    generate_patch_datasets(full_dataset_root, patch_sizes, output_root)
    copy_original_val(full_dataset_root, output_root)
