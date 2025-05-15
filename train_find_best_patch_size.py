import os
import gc
import torch
from ultralytics import YOLO
from pathlib import Path

# PATCH_SIZES = [320, 480, 640, 800, 960]
# ✅ 設定參數區
# PATCH_SIZES = [480, 640, 800, 960]
PATCH_SIZES = [960]
ROOT_DIR = Path("../RandomPick_v6_Train_Patched")
SAVE_DIR = Path("runs_patch_{size}")
MODEL_ARCH = "yolo11n.pt"
EPOCHS = 12
BATCH_SIZE = 8
DEVICE = 0  # 或 "cpu"

CLASS_NAMES = [
    "ship",
    "aquaculture cage",
    "buoy"
]

# ✅ 執行每個 patch size 的訓練流程
for size in PATCH_SIZES:
    dataset_path = ROOT_DIR / f"patch_{size}"
    yaml_path = dataset_path / "data.yaml"
    save_dir = SAVE_DIR.with_name(f"runs_patch_{size}")

    # ⬇️ 自動產生 YAML（使用絕對路徑）
    abs_dataset_path = dataset_path.resolve().as_posix()
    yaml_content = f"""\
train: {abs_dataset_path}/train/images
val: {abs_dataset_path}/val/images

nc: {len(CLASS_NAMES)}
names:
"""
    for name in CLASS_NAMES:
        yaml_content += f"  - {name}\n"

    yaml_path.write_text(yaml_content, encoding="utf-8")

    # ⬇️ 執行訓練
    print(f"\n🚀 開始訓練 Patch Size: {size}x{size}")
    print(f"📁 資料集路徑: {dataset_path}")
    print(f"📝 YAML 寫入至: {yaml_path}")
    print(f"💾 模型儲存至: {save_dir}\n")

    model = YOLO(MODEL_ARCH)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=size,
        device=DEVICE,
        amp=False,       # ❗關閉混合精度避免 NaN
        lr0=0.002,       # ❗調降初始學習率
        project=str(save_dir),
        name="exp_patch",
        exist_ok=True,
        accumulate=2
    )

    # ✅ 訓練後釋放資源
    del model
    torch.cuda.empty_cache()
    gc.collect()

print("✅ 所有 patch size 訓練已完成。")
