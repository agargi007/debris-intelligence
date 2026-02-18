from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import base64
from datetime import datetime

# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

ALLOWED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# ---------------- LOAD MODEL ---------------- #

model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

print("✅ YOLO + DeepSORT Ready")

# ---------------- FASTAPI ---------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTILS ---------------- #

def detect_and_track(frame):
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def enhance_image_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced

# ---------------- IMAGE ENDPOINT ---------------- #

@app.post("/detect-image/")
async def detect_image(file: UploadFile = File(...)):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Load original
    original_img = cv2.imread(file_path)
    original_img = cv2.resize(original_img, (FRAME_WIDTH, FRAME_HEIGHT))

    # Enhance
    enhanced_img = enhance_image_clahe(original_img.copy())

    # Run detection on enhanced
    img = enhanced_img.copy()
    results = model(img)[0]

    class_counts = {}
    confidence_list = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        confidence_list.append(conf)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    total_objects = sum(class_counts.values())

    class_percentages = {
        cls: round((count / total_objects) * 100, 2)
        for cls, count in class_counts.items()
    } if total_objects > 0 else {}

    avg_confidence = round(sum(confidence_list) / len(confidence_list), 3) if confidence_list else 0

    # Save images
    original_path = os.path.join(OUTPUT_DIR, f"orig_{timestamp}.jpg")
    enhanced_path = os.path.join(OUTPUT_DIR, f"enh_{timestamp}.jpg")
    detected_path = os.path.join(OUTPUT_DIR, f"det_{timestamp}.jpg")

    cv2.imwrite(original_path, original_img)
    cv2.imwrite(enhanced_path, enhanced_img)
    cv2.imwrite(detected_path, img)

    return {
        "total_objects": total_objects,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "average_confidence": avg_confidence,
        "original_base64": image_to_base64(original_path),
        "enhanced_base64": image_to_base64(enhanced_path),
        "detected_base64": image_to_base64(detected_path)
    }

# ---------------- VIDEO + HEATMAP ENDPOINT ---------------- #

@app.post("/detect-video-with-heatmap/")
async def detect_video_with_heatmap(file: UploadFile = File(...)):

    filename = file.filename.lower()
    file_ext = os.path.splitext(filename)[1]

    if file_ext not in ALLOWED_VIDEO_FORMATS:
        return {"error": f"Unsupported format. Allowed: {ALLOWED_VIDEO_FORMATS}"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    geo_heatmap = defaultdict(int)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(file_path)

    if file_ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_video_path = os.path.join(OUTPUT_DIR, f"det_{timestamp}{file_ext}")
    out = cv2.VideoWriter(out_video_path, fourcc, 25, (FRAME_WIDTH, FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        tracks = detect_and_track(frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, w, h = map(int, track.to_ltrb())
            cx, cy = int(l+w/2), int(t+h/2)

            geo_heatmap[(cx, cy)] += 1
            cv2.rectangle(frame, (l,t), (l+w,t+h), (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()

    if len(geo_heatmap) == 0:
        return {
            "message": "Video processed but no detections found.",
            "output_video": out_video_path
        }

    # Create heatmap
    xs = [p[0] for p in geo_heatmap]
    ys = [p[1] for p in geo_heatmap]
    weights = [geo_heatmap[p] for p in geo_heatmap]

    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, c=weights, cmap='hot', s=10)
    plt.colorbar(label="Debris Frequency")
    plt.title("Debris Spatial Heatmap")

    heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_{timestamp}.png")
    plt.savefig(heatmap_path)
    plt.close()

    df = pd.DataFrame([
        {"x_pixel": x, "y_pixel": y, "count": geo_heatmap[(x,y)]}
        for (x,y) in geo_heatmap
    ])

    csv_path = os.path.join(OUTPUT_DIR, f"heatmap_data_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    return {
        "output_video": out_video_path,
        "heatmap_image_base64": image_to_base64(heatmap_path),
        "heatmap_csv": csv_path
    }
