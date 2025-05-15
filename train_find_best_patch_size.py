import os
from ultralytics import YOLO
from pathlib import Path

# ✅ 設定參數區
PATCH_SIZES = [320, 480, 640, 800, 960]
ROOT_DIR = Path(r"../RandomPick_v6_Train_Patched")  # ⚠️ 改這裡
SAVE_DIR = Path("runs_patch_{size}")
MODEL_ARCH = "yolov11n.pt"  
EPOCHS = 12
BATCH_SIZE = 16
DEVICE = 0  # 改成 'cpu' 或指定 GPU 編號

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

    # ⬇️ 自動產生 YAML
    yaml_content = f"""\n\
train: {dataset_path.as_posix()}/train/images
val: {dataset_path.as_posix()}/val/images

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
        project=str(save_dir),
        name="exp_patch",
        exist_ok=True
    )

print("✅ 所有 patch size 訓練已完成。")
