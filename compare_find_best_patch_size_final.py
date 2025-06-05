import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from torchvision.ops import nms
import torch
import cv2
import matplotlib.pyplot as plt
import gc

# ───────────── 設定區 ─────────────
PATCH_SIZES = [320, 480, 640, 800]  # 不再使用 960
VAL_IMG_DIR = Path("../RandomPick_v6_Train_Patched/original_val/images")
VAL_LBL_DIR = Path("../RandomPick_v6_Train_Patched/original_val/labels")
WORK_DIR    = Path("./CompareResultsFinal")
CLASS_NAMES = ['ship', 'aquaculture cage', 'buoy']
CONF_THRES, IOU_THRES = 0.5, 0.5
# ────────────────────────────────

def xywhn_to_xyxy(box):
    x, y, w, h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]

def calc_iou(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0: return 0.0
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter)

def is_near_patch_edge(box, img_w, img_h, crop_w, crop_h, thr=5):
    """box: [cls, x, y, w, h, conf] (pixels)"""
    _, x_c, y_c, w, h, _ = box
    x_min, y_min = x_c - w/2, y_c - h/2
    x_max, y_max = x_c + w/2, y_c + h/2

    col = int(x_c // crop_w)
    row = int(y_c // crop_h)
    patch_x_min, patch_y_min = col * crop_w, row * crop_h
    patch_x_max, patch_y_max = patch_x_min + crop_w, patch_y_min + crop_h

    return (
        (x_min - patch_x_min) < thr or (patch_x_max - x_max) < thr or
        (y_min - patch_y_min) < thr or (patch_y_max - y_max) < thr
    )

def boxes_are_adjacent(b1, b2, max_dist=5):
    """b1/b2: [cls, x, y, w, h, conf] (pixels)"""
    x1_min, y1_min = b1[1] - b1[3]/2, b1[2] - b1[4]/2
    x1_max, y1_max = b1[1] + b1[3]/2, b1[2] + b1[4]/2
    x2_min, y2_min = b2[1] - b2[3]/2, b2[2] - b2[4]/2
    x2_max, y2_max = b2[1] + b2[3]/2, b2[2] + b2[4]/2

    horiz_touch = abs(x1_min - x2_max) < max_dist or abs(x1_max - x2_min) < max_dist
    vert_overlap = not (y1_max < y2_min or y2_max < y1_min)

    vert_touch = abs(y1_min - y2_max) < max_dist or abs(y1_max - y2_min) < max_dist
    horiz_overlap = not (x1_max < x2_min or x2_max < x1_min)

    return (horiz_touch and vert_overlap) or (vert_touch and horiz_overlap)

def merge_two_boxes(b1, b2):
    cls = b1[0]
    x_min = min(b1[1] - b1[3]/2, b2[1] - b2[3]/2)
    y_min = min(b1[2] - b1[4]/2, b2[2] - b2[4]/2)
    x_max = max(b1[1] + b1[3]/2, b2[1] + b2[3]/2)
    y_max = max(b1[2] + b1[4]/2, b2[2] + b2[4]/2)
    x_c, y_c = (x_min + x_max)/2, (y_min + y_max)/2
    w, h     = x_max - x_min, y_max - y_min
    conf     = min(b1[5], b2[5])
    return [cls, x_c, y_c, w, h, conf]

def merge_boxes_across_patches(boxes, crop_w, crop_h, img_w, img_h, thr=5):
    """boxes: list[[cls,x,y,w,h,conf]] (pixels)"""
    used, merged = set(), []
    patch_map = {}
    for idx, b in enumerate(boxes):
        col, row = int(b[1] // crop_w), int(b[2] // crop_h)
        patch_map.setdefault((row, col), []).append(idx)

    for idx, b in enumerate(boxes):
        if idx in used: continue
        cur = b.copy()
        if is_near_patch_edge(b, img_w, img_h, crop_w, crop_h, thr):
            col, row = int(b[1] // crop_w), int(b[2] // crop_h)
            neighbors = [(row+1,col),(row-1,col),(row,col+1),(row,col-1)]
            for nb in neighbors:
                for n_idx in patch_map.get(nb, []):
                    if n_idx in used: continue
                    b2 = boxes[n_idx]
                    if b2[0] != b[0]: continue
                    if boxes_are_adjacent(cur, b2, thr):
                        cur = merge_two_boxes(cur, b2)
                        used.add(n_idx)
        used.add(idx)
        merged.append(cur)
    return merged


def evaluate(pred_dir, gt_dir, file_list, class_num):
    TP = [0] * class_num
    GT = [0] * class_num
    PRED = [0] * class_num

    for fn in file_list:
        stem = Path(fn).stem
        gt_file = gt_dir / f"{stem}.txt"
        pr_file = pred_dir / f"{stem}.txt"

        gt = {i: [] for i in range(class_num)}
        pr = {i: [] for i in range(class_num)}

        if gt_file.exists():
            for ln in gt_file.read_text().splitlines():
                c, x, y, w, h = map(float, ln.strip().split()[:5])
                gt[int(c)].append(xywhn_to_xyxy([x, y, w, h]))

        if pr_file.exists():
            for ln in pr_file.read_text().splitlines():
                c, x, y, w, h = map(float, ln.strip().split()[:5])
                pr[int(c)].append(xywhn_to_xyxy([x, y, w, h]))

        for cls in range(class_num):
            GT[cls] += len(gt[cls])
            PRED[cls] += len(pr[cls])
            matched = [False] * len(pr[cls])
            for g in gt[cls]:
                for i, p in enumerate(pr[cls]):
                    if matched[i]: continue
                    if calc_iou(g, p) >= IOU_THRES:
                        TP[cls] += 1
                        matched[i] = True
                        break

    recall = [TP[c] / GT[c] if GT[c] else 0 for c in range(class_num)]
    prec   = [TP[c] / PRED[c] if PRED[c] else 0 for c in range(class_num)]
    return recall, prec

def infer_patch(model, img_dir, out_dir, crop_size):
    out_lbl = out_dir / "labels"
    out_lbl.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png']])
    total_time, max_mem = 0.0, 0

    for fname in tqdm(files, desc=f"Infer Patch {crop_size}", ncols=100):
        img = cv2.imread(str(img_dir / fname))
        if img is None: continue
        h, w = img.shape[:2]
        all_boxes = []
        start = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ✅ 無 overlap + ceil 切法（如 generate）
        rows = (h + crop_size - 1) // crop_size
        cols = (w + crop_size - 1) // crop_size

        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * crop_size, r * crop_size
                x1, y1 = min(x0 + crop_size, w), min(y0 + crop_size, h)
                patch = img[y0:y1, x0:x1]

                # ✅ 指定輸入尺寸
                for res in model(patch, verbose=False, imgsz=(crop_size, crop_size)):
                    for b in res.boxes:
                        if b.conf.item() < CONF_THRES: continue
                        cls = int(b.cls)
                        x, y, bw, bh = b.xywh[0].tolist()
                        all_boxes.append([cls, x0 + x, y0 + y, bw, bh, b.conf.item()])

        total_time += time.time() - start
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**2
            max_mem = max(max_mem, mem)

        # ── 每類別 NMS ──
        nms_boxes = []
        for cls in set(b[0] for b in all_boxes):
            cls_boxes = [b for b in all_boxes if b[0] == cls]
            xyxy  = torch.tensor([[b[1]-b[3]/2, b[2]-b[4]/2,
                                   b[1]+b[3]/2, b[2]+b[4]/2] for b in cls_boxes])
            confs = torch.tensor([b[5] for b in cls_boxes])
            keep  = nms(xyxy, confs, IOU_THRES)
            nms_boxes.extend([cls_boxes[i] for i in keep])

        # ✅ 跨 patch 邊界合併
        merged = merge_boxes_across_patches(nms_boxes, crop_size, crop_size, w, h)
        final  = [[b[0], b[1]/w, b[2]/h, b[3]/w, b[4]/h] for b in merged]

        (out_lbl / f"{Path(fname).stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"
                      for c, x, y, bw, bh in final)
        )

    return total_time, max_mem, files




# ──────── 主程序 ────────
if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    all_recall, all_prec, all_f1, all_time, all_mem = {}, {}, {}, {}, {}

    for size in PATCH_SIZES:
        model_path = Path(f"./runs_patch_{size}/exp_patch/weights/best.pt")
        assert model_path.exists(), f"{model_path} 不存在"
        model = YOLO(str(model_path))

        out_dir = WORK_DIR / f"output_patch_{size}"
        infer_time, max_mem, file_list = infer_patch(model, VAL_IMG_DIR, out_dir, size)
        recall, prec = evaluate(out_dir / "labels", VAL_LBL_DIR, file_list, len(CLASS_NAMES))
        f1 = [2*r*p/(r+p) if (r+p) > 0 else 0 for r, p in zip(recall, prec)]

        all_recall[size] = sum(recall) / len(CLASS_NAMES)
        all_prec[size]   = sum(prec)   / len(CLASS_NAMES)
        all_f1[size]     = sum(f1)     / len(CLASS_NAMES)
        all_time[size]   = infer_time
        all_mem[size]    = max_mem

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ➕ 儲存 CSV
    csv_path = WORK_DIR / "compare_patch_sizes.csv"
    with open(csv_path, "w") as f:
        f.write("PatchSize,Recall,Precision,F1,Time(s),MaxMemory(MB)\n")
        for s in PATCH_SIZES:
            f.write(f"{s},{all_recall[s]:.4f},{all_prec[s]:.4f},{all_f1[s]:.4f},{all_time[s]:.2f},{all_mem[s]:.1f}\n")
    print("✅ compare_patch_sizes.csv 已儲存")

    # ➕ 畫圖
    def plot_metric(metric_dict, title, fname):
        labels = [str(k) for k in metric_dict.keys()]
        values = list(metric_dict.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, alpha=0.7)

        # 每個 bar 上標數值（字體放大）
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.3f}", ha='center', va='bottom', fontsize=18)

        plt.title(title, fontsize=20)
        plt.xlabel("Patch Size", fontsize=18)
        plt.ylabel(title, fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, axis="y", ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(WORK_DIR / fname)
        plt.close()



    plot_metric(all_recall, "Average Recall", "recall_patch_sizes.png")
    plot_metric(all_prec,   "Average Precision", "precision_patch_sizes.png")
    plot_metric(all_f1,     "Average F1-score", "f1_patch_sizes.png")
    plot_metric(all_mem,    "Max GPU Memory (MB)", "memory_patch_sizes.png")

    print("🎉 所有 patch size 模型比較已完成！")
