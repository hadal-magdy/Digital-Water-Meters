# ============================================================
# Digital Water Meters - VS Code Version
# ============================================================

# ── STEP 1: Download Dataset ─────────────────────────────────
from roboflow import Roboflow

rf = Roboflow(api_key="ynztZAbimP2p3bAq6Hw0")
project = rf.workspace("seeed-studio-dbk14").project("digital-meter-water")
dataset = project.version(1).download("yolov8")

print("✅ Dataset downloaded!")


# ── STEP 2: Explore Images ────────────────────────────────────
import os
import cv2
import matplotlib
matplotlib.use('Agg')  # no pop-up windows — saves images to files instead
import matplotlib.pyplot as plt

images_dir = "digital-meter-water-1/train/images/"
images = os.listdir(images_dir)

print(f"Total images: {len(images)}")
print("\nFirst 5 images:")
for img in images[:5]:
    print(img)


# ── STEP 3: Show One Image (saved to file) ────────────────────
img_path = images_dir + images[0]
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img_rgb)
plt.title("Water Meter Image")
plt.axis("off")
plt.savefig("output_step3_sample_image.png", bbox_inches='tight')
plt.close()

print(f"✅ Image saved: output_step3_sample_image.png")


# ── STEP 4: Draw Bounding Boxes on One Image ──────────────────
class_names = ['0','1','2','3','4','5','6','7','8','9','N']

img_path = "digital-meter-water-1/train/images/1683618660-8624434_png_jpg.rf.3bf8237db131225acbb19c6e9b50d20b.jpg"
lbl_path = "digital-meter-water-1/train/labels/1683618660-8624434_png_jpg.rf.3bf8237db131225acbb19c6e9b50d20b.txt"

# Fallback: if that specific image doesn't exist, use the first available one
if not os.path.exists(img_path):
    img_path = images_dir + images[0]
    lbl_path = "digital-meter-water-1/train/labels/" + images[0].replace(".jpg", ".txt")

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(lbl_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * w
    y_center = float(parts[2]) * h
    bw       = float(parts[3]) * w
    bh       = float(parts[4]) * h

    x1 = int(x_center - bw / 2)
    y1 = int(y_center - bh / 2)
    x2 = int(x_center + bw / 2)
    y2 = int(y_center + bh / 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, class_names[class_id], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.title("Meter with Bounding Boxes")
plt.axis("off")
plt.savefig("output_step4_bounding_boxes.png", bbox_inches='tight')
plt.close()

print("✅ Saved: output_step4_bounding_boxes.png")


# ── STEP 5: Crop Individual Digits ───────────────────────────
crops = []

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * w
    y_center = float(parts[2]) * h
    bw       = float(parts[3]) * w
    bh       = float(parts[4]) * h

    x1 = int(x_center - bw / 2)
    y1 = int(y_center - bh / 2)
    x2 = int(x_center + bw / 2)
    y2 = int(y_center + bh / 2)

    crop = img[y1:y2, x1:x2]
    crops.append((class_names[class_id], crop))

if crops:
    fig, axes = plt.subplots(1, len(crops), figsize=(15, 5))
    if len(crops) == 1:
        axes = [axes]
    for i, (label, crop) in enumerate(crops):
        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[i].imshow(crop_rgb)
        axes[i].set_title(f"Digit: {label}")
        axes[i].axis("off")

    plt.suptitle("Cropped Digits from Meter")
    plt.savefig("output_step5_cropped_digits.png", bbox_inches='tight')
    plt.close()
    print("✅ Saved: output_step5_cropped_digits.png")


# ── STEP 6: Sort Digits Left → Right & Read Full Number ──────
crops_with_pos = []

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * w
    y_center = float(parts[2]) * h
    bw       = float(parts[3]) * w
    bh       = float(parts[4]) * h

    x1 = int(x_center - bw / 2)
    y1 = int(y_center - bh / 2)
    x2 = int(x_center + bw / 2)
    y2 = int(y_center + bh / 2)

    crop = img[y1:y2, x1:x2]
    crops_with_pos.append((x_center, class_names[class_id], crop))

crops_with_pos.sort(key=lambda item: item[0])
full_reading = ''.join([label for _, label, _ in crops_with_pos])
print(f"Meter Reading: {full_reading}")

if crops_with_pos:
    fig, axes = plt.subplots(1, len(crops_with_pos), figsize=(15, 5))
    if len(crops_with_pos) == 1:
        axes = [axes]
    for i, (_, label, crop) in enumerate(crops_with_pos):
        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[i].imshow(crop_rgb)
        axes[i].set_title(f"{label}")
        axes[i].axis("off")

    plt.suptitle(f"Meter Reading: {full_reading}")
    plt.savefig("output_step6_sorted_reading.png", bbox_inches='tight')
    plt.close()
    print("✅ Saved: output_step6_sorted_reading.png")


# ── STEP 7: Process All Images ────────────────────────────────
labels_dir = "digital-meter-water-1/train/labels/"
all_readings = []

for img_file in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_file)
    lbl_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))

    if not os.path.exists(lbl_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    with open(lbl_path, "r") as f:
        lines = f.readlines()

    crops_with_pos = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bw       = float(parts[3]) * w
        bh       = float(parts[4]) * h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        crop = img[y1:y2, x1:x2]
        crops_with_pos.append((x_center, class_names[class_id], crop))

    crops_with_pos.sort(key=lambda item: item[0])
    full_reading = ''.join([label for _, label, _ in crops_with_pos])
    all_readings.append({"image": img_file, "reading": full_reading})

print(f"Total images processed: {len(all_readings)}\n")
for r in all_readings[:10]:
    print(f"📊 {r['image'][:30]}...  →  Reading: {r['reading']}")


# ── STEP 8: Stats ─────────────────────────────────────────────
total  = len(all_readings)
with_N = [r for r in all_readings if 'N' in r['reading']]
clean  = [r for r in all_readings if 'N' not in r['reading']]

print(f"\nTotal images:       {total}")
print(f"Clean readings:     {len(clean)}  ({len(clean)/total*100:.1f}%)")
print(f"Readings with N:    {len(with_N)}  ({len(with_N)/total*100:.1f}%)")


# ── STEP 9: Save One Test Reading to JSON File ────────────────
import json
import datetime

test_reading = clean[0]

payload = {
    "meter_id":  "WM-HOME-001",
    "reading":   test_reading['reading'],
    "unit":      "m3",
    "timestamp": datetime.datetime.now().isoformat(),
    "source":    "image_ocr",
    "image":     test_reading['image'],
    "status":    "ok"
}

with open("output_step9_test_reading.json", "w") as f:
    json.dump(payload, f, indent=2)

print("\n📦 Test reading saved to: output_step9_test_reading.json")
print(json.dumps(payload, indent=2))


# ── STEP 10: Improve Readings (fix N digits) ──────────────────
def improve_reading(reading):
    if 'N' not in reading:
        return reading

    result = ""
    for i, char in enumerate(reading):
        if char == 'N':
            prev_digit = None
            for j in range(i-1, -1, -1):
                if reading[j] != 'N':
                    prev_digit = int(reading[j])
                    break
            if prev_digit is not None:
                result += str((prev_digit + 1) % 10)
            else:
                result += '0'
        else:
            result += char
    return result

improved_readings = []
for r in all_readings:
    improved = improve_reading(r['reading'])
    improved_readings.append({
        "image":            r['image'],
        "original_reading": r['reading'],
        "improved_reading": improved
    })

still_bad = [r for r in improved_readings if 'N' in r['improved_reading']]
clean     = [r for r in improved_readings if 'N' not in r['improved_reading']]

print(f"\nTotal images:        {len(improved_readings)}")
print(f"Clean readings:      {len(clean)} ({len(clean)/len(improved_readings)*100:.1f}%)")
print(f"Still has N:         {len(still_bad)}")

print("\nSample improvements:")
count = 0
for r in improved_readings:
    if r['original_reading'] != r['improved_reading']:
        print(f"  {r['original_reading']} → {r['improved_reading']} ✅")
        count += 1
    if count >= 10:
        break


# ── STEP 11: Save All Improved Readings to JSON File ─────────
all_payloads = []

for r in improved_readings:
    all_payloads.append({
        "meter_id":  "WM-HOME-001",
        "reading":   r['improved_reading'],
        "unit":      "m3",
        "timestamp": datetime.datetime.now().isoformat(),
        "image":     r['image']
    })

with open("output_step11_all_readings.json", "w") as f:
    json.dump(all_payloads, f, indent=2)

print(f"\n✅ All {len(all_payloads)} readings saved to: output_step11_all_readings.json")


# ── STEP 12: Train YOLOv8 Model ──────────────────────────────
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="digital-meter-water-1/data.yaml",
    epochs=10,
    imgsz=416,
    batch=16,
    name="meter_model"
)

print("✅ Training Done!")


# ── STEP 13: Test Trained Model on Validation Images ─────────
# Finds the latest trained model folder automatically
import glob

model_folders = sorted(glob.glob("runs/detect/meter_model*"))
if not model_folders:
    print("❌ No trained model found. Make sure training completed.")
else:
    best_model_path = model_folders[-1] + "/weights/best.pt"
    print(f"✅ Loading model from: {best_model_path}")

    model = YOLO(best_model_path)

    val_dir  = "digital-meter-water-1/valid/images/"
    img_path = val_dir + os.listdir(val_dir)[0]
    results  = model.predict(img_path, conf=0.5)

    img_out = results[0].plot()
    img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 5))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.savefig("output_step13_yolo_prediction.png", bbox_inches='tight')
    plt.close()

    print("✅ Saved: output_step13_yolo_prediction.png")
    print("\nDetected digits:")
    for box in results[0].boxes:
        print(f"  Digit: {class_names[int(box.cls)]}  confidence: {box.conf.item():.2f}")


    # ── STEP 14: Auto-Read All Validation Images with YOLOv8 ─────
    def read_meter(img_path):
        results = model.predict(img_path, conf=0.5, verbose=False)
        boxes   = results[0].boxes

        if len(boxes) == 0:
            return None

        digits = []
        for box in boxes:
            x_center = box.xywh[0][0].item()
            digit    = class_names[int(box.cls)]
            digits.append((x_center, digit))

        digits.sort(key=lambda x: x[0])
        reading = ''.join([d[1] for d in digits])
        reading = improve_reading(reading)
        return reading

    val_images   = os.listdir(val_dir)
    results_list = []

    for img_file in val_images[:20]:
        reading = read_meter(val_dir + img_file)
        if reading:
            results_list.append({"image": img_file, "reading": reading})
            print(f"📊 {img_file[:30]}...  →  {reading}")

    print(f"\n✅ Done! {len(results_list)} meters read automatically!")


    # ── STEP 15: Save YOLOv8 Results to JSON File ────────────────
    yolo_payloads = []

    for r in results_list:
        yolo_payloads.append({
            "meter_id":  "WM-HOME-001",
            "reading":   r['reading'],
            "unit":      "m3",
            "timestamp": datetime.datetime.now().isoformat(),
            "image":     r['image'],
            "source":    "yolov8_auto"
        })

    with open("output_step15_yolo_readings.json", "w") as f:
        json.dump(yolo_payloads, f, indent=2)

    print(f"✅ {len(yolo_payloads)} YOLOv8 readings saved to: output_step15_yolo_readings.json")
