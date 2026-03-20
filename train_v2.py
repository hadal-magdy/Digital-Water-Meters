# ============================================================
# train_v2.py  —  Better model with more data + augmentation
# Run: python train_v2.py
# ============================================================

import os
import shutil
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────
ROBOFLOW_API_KEY = "ynztZAbimP2p3bAq6Hw0"
COMBINED_DIR     = "combined_dataset"
MODEL_NAME       = "meter_model_v2"
CLASS_NAMES      = ['0','1','2','3','4','5','6','7','8','9','N']

# ── Step 1: Download extra datasets ───────────────────────────
print("=" * 55)
print("  STEP 1 — Downloading datasets from Roboflow")
print("=" * 55)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

datasets_to_download = [
    # (workspace, project, version)
    ("seeed-studio-dbk14", "digital-meter-water",    1),  # original
    ("seeed-studio-dbk14", "digital-meter-water-gas", 1), # gas meters (similar digits)
    ("roboflow-100",       "meter-reader",            1), # general meter reader
]

downloaded_dirs = []

for workspace, project, version in datasets_to_download:
    try:
        print(f"\n📥  Downloading: {project} v{version}")
        proj    = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8")
        # Roboflow saves to folder named after project
        folder  = f"{project}-{version}"
        if os.path.exists(folder):
            downloaded_dirs.append(folder)
            print(f"✅  Saved to: {folder}")
        else:
            # Try finding the folder that was just created
            for d in os.listdir("."):
                if project.replace("-", "") in d.replace("-", "").lower() and os.path.isdir(d):
                    downloaded_dirs.append(d)
                    print(f"✅  Found at: {d}")
                    break
    except Exception as e:
        print(f"⚠️   Could not download {project}: {e}")
        print("    Skipping — will continue with other datasets")

# Always include original dataset as fallback
if not downloaded_dirs:
    downloaded_dirs = ["digital-meter-water-1"]
    print("\n⚠️   Using only original dataset")

print(f"\n✅  Datasets ready: {downloaded_dirs}")


# ── Step 2: Merge all datasets into one ───────────────────────
print("\n" + "=" * 55)
print("  STEP 2 — Merging all datasets")
print("=" * 55)

for split in ["train", "valid"]:
    os.makedirs(f"{COMBINED_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{COMBINED_DIR}/{split}/labels", exist_ok=True)

total_images = 0

for ds_dir in downloaded_dirs:
    for split in ["train", "valid"]:
        img_src = f"{ds_dir}/{split}/images"
        lbl_src = f"{ds_dir}/{split}/labels"

        if not os.path.exists(img_src):
            continue

        images = [f for f in os.listdir(img_src) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in images:
            # Add dataset prefix to avoid filename conflicts
            prefix   = ds_dir.replace("/", "_").replace("-", "_")
            new_name = f"{prefix}_{img_file}"

            # Copy image
            src = os.path.join(img_src, img_file)
            dst = f"{COMBINED_DIR}/{split}/images/{new_name}"
            shutil.copy2(src, dst)

            # Copy label
            lbl_file = img_file.rsplit(".", 1)[0] + ".txt"
            lbl_src_path = os.path.join(lbl_src, lbl_file)
            lbl_dst_path = f"{COMBINED_DIR}/{split}/labels/{new_name.rsplit('.', 1)[0]}.txt"

            if os.path.exists(lbl_src_path):
                shutil.copy2(lbl_src_path, lbl_dst_path)

        count = len(images)
        total_images += count
        print(f"  ✅  {ds_dir}/{split}  →  {count} images copied")

print(f"\n  Total images in combined dataset: {total_images}")


# ── Step 3: Create combined data.yaml ─────────────────────────
print("\n" + "=" * 55)
print("  STEP 3 — Creating combined data.yaml")
print("=" * 55)

data_yaml = {
    "path":  os.path.abspath(COMBINED_DIR),
    "train": "train/images",
    "val":   "valid/images",
    "nc":    11,
    "names": CLASS_NAMES
}

yaml_path = f"{COMBINED_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"  ✅  Saved: {yaml_path}")

# Count final dataset size
train_count = len(os.listdir(f"{COMBINED_DIR}/train/images"))
valid_count = len(os.listdir(f"{COMBINED_DIR}/valid/images"))
print(f"  Train images: {train_count}")
print(f"  Valid images: {valid_count}")


# ── Step 4: Train with augmentation ───────────────────────────
print("\n" + "=" * 55)
print("  STEP 4 — Training YOLOv8 with augmentation")
print("=" * 55)
print("  This will take a while. Grab a coffee ☕")
print("=" * 55 + "\n")

model = YOLO("yolov8s.pt")

model.train(
    data    = yaml_path,
    epochs  = 30,           # more epochs for better learning
    imgsz   = 416,
    batch   = 16,
    name    = MODEL_NAME,

    # ── Augmentation settings ──────────────────────────────
    hsv_h   = 0.02,   # tiny hue shift (meters don't change colour much)
    hsv_s   = 0.7,    # saturation variation (different lighting)
    hsv_v   = 0.4,    # brightness variation (dark/bright conditions)
    degrees = 5,      # slight rotation (camera not perfectly straight)
    translate= 0.1,   # slight position shift
    scale   = 0.3,    # zoom variation (different distances)
    shear   = 2.0,    # slight shear (perspective)
    perspective=0.0005, # slight perspective warp
    flipud  = 0.0,    # no vertical flip (meters are always upright)
    fliplr  = 0.0,    # no horizontal flip (digits read left→right)
    mosaic  = 1.0,    # combine 4 images (helps generalise)
    mixup   = 0.1,    # blend images (helps with varied backgrounds)
    patience= 15,     # stop early if no improvement for 15 epochs
)

print("\n✅  Training Done!")
print(f"  Model saved to: runs/detect/{MODEL_NAME}/weights/best.pt")
print("\n  Now update app.py to use the new model:")
print(f"  model_folders = sorted(glob.glob('runs/detect/{MODEL_NAME}*'))")
print("\n  Or just run app.py — it auto-picks the latest model folder! 🚀")
