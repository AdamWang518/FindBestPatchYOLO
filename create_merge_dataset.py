# ────────────────────────────────────────────────────────────────
# Patch 拼接 + 僅左右跨 patch 的貼邊物件合併
# 2025-04-26  © 你的專案
# ────────────────────────────────────────────────────────────────
# 用來重建破碎的圖片還原原本的資料
import os
import cv2
import numpy as np
from tqdm.auto import tqdm

# ──────────────────────── 讀取 patch 檔案 ──────────────────────────
def collect_patches(img_root: str):
    patch_map = {}
    for root, _, files in os.walk(img_root):
        for fn in files:
            if '_cropped_' not in fn or not fn.lower().endswith(('.jpg', '.png')):
                continue
            basename = os.path.splitext(fn)[0]
            orig, rc = basename.split('_cropped_', 1)
            try:
                r, c = map(int, rc.split('_'))
            except ValueError:
                continue
            patch_map.setdefault(orig, []).append((r, c, os.path.join(root, fn)))
    return patch_map

# ──────────────────────── 僅左右跨 patch 的框合併 ──────────────────────────
def merge_boxes_across_patches(
        boxes,
        crop_w: int,
        crop_h: int,
        edge_thr: int = 5,
        min_vertical_overlap: float = 0.3):
    """
    僅合併左右相鄰 patch 的貼邊物件，防止小物件誤併成超大物件。
    """

    if not boxes:
        return []

    def patch_col(b):
        return int(b[1] // crop_w)

    def patch_row(b):
        return int(b[2] // crop_h)

    def horizontally_adjacent(b1, b2):
        r1, c1 = patch_row(b1), patch_col(b1)
        r2, c2 = patch_row(b2), patch_col(b2)
        if r1 != r2 or abs(c1 - c2) != 1:
            return False

        x1_min, x1_max = b1[1] - b1[3]/2, b1[1] + b1[3]/2
        y1_min, y1_max = b1[2] - b1[4]/2, b1[2] + b1[4]/2
        x2_min, x2_max = b2[1] - b2[3]/2, b2[1] + b2[3]/2
        y2_min, y2_max = b2[2] - b2[4]/2, b2[2] + b2[4]/2

        dx = max(x2_min - x1_max, x1_min - x2_max, 0)
        if dx > edge_thr:
            return False

        vertical_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        min_h = max(min(b1[4], b2[4]), 1e-6)
        return (vertical_overlap / min_h) >= min_vertical_overlap

    def merge_boxes(b1, b2):
        x_min = min(b1[1] - b1[3]/2, b2[1] - b2[3]/2)
        x_max = max(b1[1] + b1[3]/2, b2[1] + b2[3]/2)
        y_min = min(b1[2] - b1[4]/2, b2[2] - b2[4]/2)
        y_max = max(b1[2] + b1[4]/2, b2[2] + b2[4]/2)
        return [
            b1[0],
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
            max(b1[5], b2[5])
        ]

    merged = boxes.copy()
    changed = True
    while changed:
        changed = False
        result, skip = [], set()
        for i in range(len(merged)):
            if i in skip:
                continue
            base = merged[i]
            for j in range(i + 1, len(merged)):
                if j in skip or merged[j][0] != base[0]:
                    continue
                if horizontally_adjacent(base, merged[j]):
                    base = merge_boxes(base, merged[j])
                    skip.add(j)
                    changed = True
            result.append(base)
        merged = result

    return merged

# ──────────────────────── 主流程：拼接 + 合併 ──────────────────────────
def merge_patches_all(
        image_root: str,
        output_dir: str,
        img_w: int, img_h: int,
        crop_w: int, crop_h: int,
        edge_thr: int = 5,
        min_vertical_overlap: float = 0.3):

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    patch_map = collect_patches(image_root)
    rows, cols = img_h // crop_h, img_w // crop_w

    for orig in tqdm(sorted(patch_map.keys()),
                     desc="Merging patches",
                     unit="img",
                     ascii=True,
                     dynamic_ncols=True):
        canvas = 255 * np.ones((img_h, img_w, 3), dtype=np.uint8)
        all_boxes = []

        patch_dict = {(r, c): p for r, c, p in patch_map[orig]}

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in patch_dict:
                    continue
                patch_path = patch_dict[(r, c)]
                x_off, y_off = c * crop_w,c*crop_h
                y_off = r * crop_h

                img = cv2.imread(patch_path)
                if img is not None:
                    canvas[y_off:y_off + crop_h, x_off:x_off + crop_w] = img

                label_path = (patch_path
                              .replace("images", "labels")
                              .rsplit('.', 1)[0] + ".txt")
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        ps = line.strip().split()
                        if len(ps) < 5:
                            continue
                        cls, xc, yc, w, h = map(float, ps[:5])
                        x1 = (xc - w/2) * crop_w + x_off
                        y1 = (yc - h/2) * crop_h + y_off
                        x2 = (xc + w/2) * crop_w + x_off
                        y2 = (yc + h/2) * crop_h + y_off
                        all_boxes.append([
                            cls,
                            (x1 + x2) / 2,
                            (y1 + y2) / 2,
                            x2 - x1,
                            y2 - y1,
                            1.0
                        ])

        merged_boxes = merge_boxes_across_patches(
            all_boxes, crop_w, crop_h,
            edge_thr=edge_thr,
            min_vertical_overlap=min_vertical_overlap
        )

        cv2.imwrite(os.path.join(output_dir, "images", f"{orig}.jpg"), canvas)

        with open(os.path.join(output_dir, "labels", f"{orig}.txt"),
                  'w', encoding='utf-8') as f:
            for b in merged_boxes:
                cls, xc, yc, w, h, _ = b
                f.write(f"{int(cls)} {xc / img_w:.6f} {yc / img_h:.6f} "
                        f"{w / img_w:.6f} {h / img_h:.6f}\n")

# ──────────────────────── 執行設定 ──────────────────────────
if __name__ == "__main__":
    image_root = r"D:\Github\RandomPick_v6"            # patch 根目錄（含 images/labels）
    output_dir = r"D:\Github\RandomPick_v6_New_Combined"   # 輸出合併結果目錄

    merge_patches_all(
        image_root=image_root,
        output_dir=output_dir,
        img_w=2560, img_h=1920,       # 大圖尺寸
        crop_w=640, crop_h=640,       # patch尺寸
        edge_thr=5,                  # 左右貼邊最大距離(px)
        min_vertical_overlap=0.3     # 垂直方向最小重疊比例
    )
