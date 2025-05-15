# compare_yolo_models_patchmerge_split5_v3.py
# 2025‑05‑02  Adam_W  (新增 Precision 與 F1 圖表)
"""
與 v2 相比，本版做了三項主要更新：
1. `evaluate()`  now also counts **Pred** per class，回傳 Precision；外部計算 F1。
2. 於主程式彙整 **Average Precision、F1**，並輸出
      ‑ precision_sets.png
      ‑ f1_sets.png
3. 其餘結構、路徑與 Patch‑Merge 推論流程維持不變，可直接覆蓋舊檔執行。

使用方式
---------
```bash
python compare_yolo_models_patchmerge_split5_v3.py
```
執行完畢後，在 `WORK_DIR` 會多出三張長條圖：
‣ recall_sets.png   – 平均召回率
‣ precision_sets.png – 平均 Precision
‣ f1_sets.png        – 平均 F1‑score
"""

import random, csv, shutil, os
from pathlib import Path
from typing import List, Dict, Tuple
import cv2, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.ops import nms
import numpy as np

# ──────────────────── 參數區 ────────────────────
PATCH_MODEL_PATH = r"runs_patch\exp_patch\weights\best.pt"
FULL_MODEL_PATH  = r"runs_full\exp_full\weights\best.pt"

IMG_DIR   = Path(r"D:\Github\RandomPick_v6_6_Full\val\images")   # ← val 影像
LBL_DIR   = Path(r"D:\Github\RandomPick_v6_6_Full\val\labels")   # ← val 標註
WORK_DIR  = Path(r"D:\Github\CompareResultFinal")                       # ← 輸出報表與圖
BASE_DIR  = Path(r"D:\Github\RandomPick_v6_6_Full")                # ← 根目錄，放 test*/val

CLASS_NAMES = ['ship', 'aquaculture cage', 'buoy']
CONF_THRES, IOU_THRES, EDGE_THRES = 0.5, 0.5, 5
CROP_W = CROP_H = 640
SPLIT_SEED = 42

COLORS = {0:(0,255,0), 1:(0,0,255), 2:(255,0,0)}  # BGR for visualisation
# ────────────────────────────────────────────────


# ──────────────── 公用函式 ────────────────
def xywhn_to_xyxy(box):
    x,y,w,h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]

def calc_iou(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter  = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0: return 0.0
    area1  = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2  = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter)

def draw_boxes(img, boxes):
    """boxes: [cls, x, y, w, h, conf] (norm)"""
    h, w = img.shape[:2]
    for cls, x, y, bw, bh, conf in boxes:
        x1, y1 = int((x-bw/2)*w), int((y-bh/2)*h)
        x2, y2 = int((x+bw/2)*w), int((y+bh/2)*h)
        cv2.rectangle(img, (x1,y1), (x2,y2), COLORS.get(cls,(255,255,255)), 2)
        cv2.putText(img, f"{CLASS_NAMES[cls]} {conf:.2f}",
                    (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLORS.get(cls,(255,255,255)), 1)


# ───────────── Patch‑merge helpers ─────────────
def is_near_patch_edge(box, img_w, img_h, crop_w, crop_h, thr=EDGE_THRES):
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

def boxes_are_adjacent(b1, b2, max_dist=EDGE_THRES):
    """b1/b2: [cls, x, y, w, h, conf] (pixels)"""
    x1_min, y1_min = b1[1] - b1[3]/2, b1[2] - b1[4]/2
    x1_max, y1_max = b1[1] + b1[3]/2, b1[2] + b1[4]/2
    x2_min, y2_min = b2[1] - b2[3]/2, b2[2] - b2[4]/2
    x2_max, y2_max = b2[1] + b2[3]/2, b2[2] + b2[4]/2

    horiz_touch = abs(x1_min - x2_max) < max_dist or abs(x1_max - x2_min) < max_dist
    vert_overlap = not (y1_max < y2_min or y2_max < y1_min)
    vert_touch   = abs(y1_min - y2_max) < max_dist or abs(y1_max - y2_min) < max_dist
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

def merge_boxes_across_patches(boxes, img_w, img_h, crop_w, crop_h, thr=EDGE_THRES):
    """boxes: list[[cls,x,y,w,h,conf]] (pixels)"""
    used, merged = set(), []
    # 建 patch -> idx list 對照表，加速鄰 patch 檢索
    patch_map: Dict[Tuple[int,int], List[int]] = {}
    for idx, b in enumerate(boxes):
        col, row = int(b[1] // crop_w), int(b[2] // crop_h)
        patch_map.setdefault((row,col), []).append(idx)

    for idx, b in enumerate(boxes):
        if idx in used: continue
        cur = b.copy()
        if is_near_patch_edge(b, img_w, img_h, crop_w, crop_h, thr):
            col, row = int(b[1] // crop_w), int(b[2] // crop_h)
            neighbours = [(row-1,col), (row+1,col), (row,col-1), (row,col+1)]
            for nb in neighbours:
                for n_idx in patch_map.get(nb, []):
                    if n_idx in used: continue
                    nb_box = boxes[n_idx]
                    if nb_box[0] != b[0]: continue          # 同一 class
                    if boxes_are_adjacent(cur, nb_box, thr):
                        cur = merge_two_boxes(cur, nb_box)
                        used.add(n_idx)
        merged.append(cur)
        used.add(idx)
    return merged


# ──────────────── 推論函式 ────────────────
# (與 v2 相同，僅略去註解，核心無改動)

def _infer_patch(model:YOLO, img_dir:Path, out_dir:Path, files:List[str]):
    if (out_dir/'labels').exists() and any(out_dir.joinpath('labels').iterdir()):
        return
    (out_dir/'labels').mkdir(parents=True,exist_ok=True)
    (out_dir/'images').mkdir(parents=True,exist_ok=True)

    for fn in tqdm(files, desc=f"Patch▶{out_dir.name}", dynamic_ncols=True, leave=False):
        img = cv2.imread(str(img_dir/fn))
        if img is None: continue
        h, w = img.shape[:2]; raw=[]
        for r in range(h // CROP_H):
            for c in range(w // CROP_W):
                x0, y0 = c*CROP_W, r*CROP_H
                patch  = img[y0:y0+CROP_H, x0:x0+CROP_W]
                for res in model(patch, verbose=False):
                    for b in res.boxes:
                        if b.conf.item() < CONF_THRES: continue
                        cls = int(b.cls)
                        x,y,bw,bh = b.xywh[0].tolist()  # pixels (patch)
                        raw.append([cls, x0+x, y0+y, bw, bh, b.conf.item()])

        if not raw: continue
        # Class‑wise NMS
        nms_boxes=[]
        for cls in set(b[0] for b in raw):
            cls_boxes = [b for b in raw if b[0]==cls]
            xyxy  = torch.tensor([[b[1]-b[3]/2, b[2]-b[4]/2,
                                   b[1]+b[3]/2, b[2]+b[4]/2] for b in cls_boxes])
            confs = torch.tensor([b[5] for b in cls_boxes])
            keep  = nms(xyxy, confs, IOU_THRES)
            nms_boxes.extend([cls_boxes[i] for i in keep])

        merged = merge_boxes_across_patches(nms_boxes, w, h, CROP_W, CROP_H, EDGE_THRES)
        final  = [[b[0], b[1]/w, b[2]/h, b[3]/w, b[4]/h, b[5]] for b in merged]

        (out_dir/'labels'/f"{Path(fn).stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {cf:.4f}"
                      for c,x,y,bw,bh,cf in final), "utf-8")
        vis = img.copy(); draw_boxes(vis, final)
        cv2.imwrite(str(out_dir/'images'/fn), vis)

def _infer_full(model:YOLO, img_dir:Path, out_dir:Path, files:List[str]):
    if (out_dir/'labels').exists() and any(out_dir.joinpath('labels').iterdir()):
        return
    (out_dir/'labels').mkdir(parents=True,exist_ok=True)
    (out_dir/'images').mkdir(parents=True,exist_ok=True)

    for fn in tqdm(files, desc=f"Full ▶{out_dir.name}", dynamic_ncols=True, leave=False):
        img = cv2.imread(str(img_dir/fn))
        if img is None: continue
        h, w = img.shape[:2]; final=[]
        for b in model(img, verbose=False)[0].boxes:
            if b.conf.item() < CONF_THRES: continue
            cls = int(b.cls)
            x,y,bw,bh = b.xywhn[0].tolist()   # 已歸一化
            final.append([cls,x,y,bw,bh, b.conf.item()])

        (out_dir/'labels'/f"{Path(fn).stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {cf:.4f}"
                      for c,x,y,bw,bh,cf in final), "utf-8")
        vis = img.copy(); draw_boxes(vis, final)
        cv2.imwrite(str(out_dir/'images'/fn), vis)


# ──────────────── 評估 ────────────────

def evaluate(pred_dir:Path, gt_dir:Path,
             img_names:List[str], class_num:int
             ) -> Tuple[List[int], List[int], List[int], List[float], List[float]]:
    """
    回傳 (TP_list, GT_list, Pred_list, Recall_list, Precision_list)
    *GT 僅計算 img_names 中的影像*
    """
    TP  = [0]*class_num
    GT  = [0]*class_num
    Pred= [0]*class_num

    for fn in img_names:
        stem = Path(fn).stem
        gtf  = gt_dir/f"{stem}.txt"
        if not gtf.exists(): continue

        gt = {i:[] for i in range(class_num)}
        pr = {i:[] for i in range(class_num)}

        # 讀取 GT
        for ln in gtf.read_text().splitlines():
            parts = ln.split()
            if len(parts) < 5: continue
            c,x,y,w,h = map(float, parts[:5])
            if w<=0 or h<=0: continue
            gt[int(c)].append(xywhn_to_xyxy([x,y,w,h]))

        # 讀取 prediction
        pf = pred_dir/f"{stem}.txt"
        if pf.exists():
            for ln in pf.read_text().splitlines():
                parts = ln.split()
                if len(parts) < 5: continue
                c,x,y,w,h = map(float, parts[:5])
                if w<=0 or h<=0: continue
                pr[int(c)].append(xywhn_to_xyxy([x,y,w,h]))

        #→ 統計
        for cls in range(class_num):
            GT[cls]   += len(gt[cls])
            Pred[cls] += len(pr[cls])
            matched = [False]*len(pr[cls])
            for g in gt[cls]:
                for i,p in enumerate(pr[cls]):
                    if matched[i]: continue
                    if calc_iou(g, p) >= IOU_THRES:
                        TP[cls]+=1; matched[i]=True; break

    recall    = [TP[c]/GT[c]   if GT[c]   else 0 for c in range(class_num)]
    precision = [TP[c]/Pred[c] if Pred[c] else 0 for c in range(class_num)]
    return TP, GT, Pred, recall, precision


# ───────────────── 主程式 ─────────────────
if __name__ == "__main__":
    random.seed(SPLIT_SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    patch_model = YOLO(PATCH_MODEL_PATH)
    full_model  = YOLO(FULL_MODEL_PATH)

    # ─── 建立 val → test1~5 分割 ───
    img_list = sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    random.shuffle(img_list)

    chunks: List[np.ndarray] = np.array_split(img_list, 5)
    splits  = {f"test{i+1}": list(chunks[i]) for i in range(5)}
    splits["val"] = img_list  # val 仍使用完整清單

    # ─── 彙整 dict ───
    patch_avg_rec, full_avg_rec = {}, {}
    patch_avg_prec, full_avg_prec = {}, {}
    patch_avg_f1, full_avg_f1 = {}, {}

    compare_rows = [["Set", "Mode"] + CLASS_NAMES + ["AvgRecall", "AvgPrecision", "AvgF1"]]

    for set_name, flist in splits.items():
        # (1) 複製 val → test* 影像與標註
        if set_name != "val":
            sub_img = BASE_DIR/set_name/"images"
            sub_lbl = BASE_DIR/set_name/"labels"
            if not sub_img.exists():
                sub_img.mkdir(parents=True, exist_ok=True)
                sub_lbl.mkdir(parents=True, exist_ok=True)
                for fname in flist:
                    shutil.copy(IMG_DIR/fname, sub_img/fname)
                    shutil.copy(LBL_DIR/f"{Path(fname).stem}.txt", sub_lbl/f"{Path(fname).stem}.txt")

        # (2) 推論
        p_out = WORK_DIR/f"output_patch_{set_name}"
        f_out = WORK_DIR/f"output_full_{set_name}"
        _infer_patch(patch_model, IMG_DIR, p_out, flist)
        _infer_full (full_model , IMG_DIR, f_out, flist)

        # (3) 評估
        tp_p, gt, pred_p, rec_p, prec_p = evaluate(p_out/'labels', LBL_DIR, flist, len(CLASS_NAMES))
        tp_f, _,  pred_f, rec_f, prec_f = evaluate(f_out/'labels', LBL_DIR, flist, len(CLASS_NAMES))

        # F1
        f1_p = [2*r*p/(r+p) if (r+p)>0 else 0 for r,p in zip(rec_p, prec_p)]
        f1_f = [2*r*p/(r+p) if (r+p)>0 else 0 for r,p in zip(rec_f, prec_f)]

        # 平均
        patch_avg_rec [set_name] = sum(rec_p)  / len(CLASS_NAMES)
        full_avg_rec  [set_name] = sum(rec_f)  / len(CLASS_NAMES)
        patch_avg_prec[set_name] = sum(prec_p) / len(CLASS_NAMES)
        full_avg_prec [set_name] = sum(prec_f) / len(CLASS_NAMES)
        patch_avg_f1  [set_name] = sum(f1_p)   / len(CLASS_NAMES)
        full_avg_f1   [set_name] = sum(f1_f)   / len(CLASS_NAMES)

        # csv 行
        compare_rows.append([set_name, "patch"] + [f"{r:.3f}" for r in rec_p] + [f"{patch_avg_rec [set_name]:.3f}", f"{patch_avg_prec[set_name]:.3f}", f"{patch_avg_f1[set_name]:.3f}"])
        compare_rows.append([set_name, "full" ] + [f"{r:.3f}" for r in rec_f] + [f"{full_avg_rec  [set_name]:.3f}", f"{full_avg_prec [set_name]:.3f}", f"{full_avg_f1[set_name]:.3f}"])

        # 每類詳細摘要
        cs_rows = [["Class", "GT", "Patch TP", "Patch Pred", "Patch Recall", "Patch Precision", "Patch F1", "Full TP", "Full Pred", "Full Recall", "Full Precision", "Full F1"]]
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cs_rows.append([cls_name, gt[cls_idx], tp_p[cls_idx], pred_p[cls_idx], f"{rec_p[cls_idx]:.3f}", f"{prec_p[cls_idx]:.3f}", f"{f1_p[cls_idx]:.3f}",
                                       tp_f[cls_idx], pred_f[cls_idx], f"{rec_f[cls_idx]:.3f}", f"{prec_f[cls_idx]:.3f}", f"{f1_f[cls_idx]:.3f}"])
        (WORK_DIR/f"class_summary_{set_name}.csv").write_text("\n".join(",".join(map(str,r)) for r in cs_rows), "utf-8")

    # (4) compare_sets.csv
    (WORK_DIR/"compare_sets.csv").write_text("\n".join(",".join(r) for r in compare_rows), "utf-8")
    print("✅ compare_sets.csv 已輸出")

    # (5) 繪圖
    sets = sorted(splits.keys(), key=lambda x: (x != "val", x))
    x = range(len(sets)); w = 0.35

    def _plot(metric_name:str, patch_dict:Dict[str,float], full_dict:Dict[str,float], fname:str):
        plt.figure(figsize=(10, 6))
        plt.bar([i - w/2 for i in x], [patch_dict[s] for s in sets], w, label="Patch", alpha=0.7)
        plt.bar([i + w/2 for i in x], [full_dict [s] for s in sets], w, label="Full" , alpha=0.7)
        plt.xticks(list(x), sets)
        plt.ylabel(f"Average {metric_name}")
        plt.title(f"Patch vs Full – Average {metric_name} per Set")
        plt.grid(axis="y", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(WORK_DIR/fname)
        plt.close()
        print(f"📈 {fname} 已輸出")

    _plot("Recall",    patch_avg_rec,  full_avg_rec,  "recall_sets.png")
    _plot("Precision", patch_avg_prec, full_avg_prec, "precision_sets.png")
    _plot("F1‑score",  patch_avg_f1,  full_avg_f1,   "f1_sets.png")

    print("🎉 全部完成！")
