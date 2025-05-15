import os
import gc
import torch
from pathlib import Path
from ultralytics import YOLO
import subprocess

# ✅ 設定參數區
# PATCH_SIZES = [320, 480, 640, 800, 960]
PATCH_SIZES = [ 960]
ROOT_DIR = Path("../RandomPick_v6_Train_Patched")
SAVE_DIR = Path("runs_patch_{size}")
MODEL_ARCH = "yolo11n.pt"
EPOCHS = 12
DEFAULT_BATCH = 16
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
    yaml_content = f"""train: {abs_dataset_path}/train/images
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

    if size == 960:
        # ✅ 使用 CLI 執行並模擬 batch=16（accumulate=2）
        print("🧠 [960x960] 使用 batch=8 並累加 simulate batch=16（accumulate=2）")
        cmd = [
            "yolo",
            "train",
            f"model={MODEL_ARCH}",
            f"data={str(yaml_path)}",
            f"epochs={EPOCHS}",
            "batch=8",
            f"imgsz={size}",
            f"device={DEVICE}",
            "amp=False",
            "lr0=0.002",
            f"project={str(save_dir)}",
            "name=exp_patch",
            "exist_ok=True",
            "accumulate=2"
        ]
        subprocess.run(cmd)
    else:
        # ✅ 其他 patch size 直接用 batch=16，無累加
        model = YOLO(MODEL_ARCH)
        model.train(
            data=str(yaml_path),
            epochs=EPOCHS,
            batch=DEFAULT_BATCH,
            imgsz=size,
            device=DEVICE,
            amp=False,
            lr0=0.002,
            project=str(save_dir),
            name="exp_patch",
            exist_ok=True,
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()

print("✅ 所有 patch size 訓練已完成。")
