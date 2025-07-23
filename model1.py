# ✅ INSTALL IF NEEDED:
# pip install ultralytics opencv-python pillow streamlit

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
import os

# === Load YOLOv8 Model ===
model = YOLO("yolov8n.pt")

# === Estimate Distance ===
def estimate_distance_from_area(area):
    if area == 0:
        return float('inf')
    return round(10000 / (area + 1), 2)


# === Labels ===
scene_keywords = ["person", "cow", "dog", "cat", "sheep", "horse"]
vehicle_keywords = ["car", "truck", "bus", "motorcycle"]
potential_child_keywords = ["sports ball", "teddy bear"]
accident_keywords = ["fire", "smoke", "stop sign", "debris", "broken", "hazard"]

# === Turn Indicator Detection ===
def detect_indicators(frame, vehicle_boxes):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    indicators = []

    for (x1, y1, x2, y2) in vehicle_boxes:
        vehicle_roi = mask[y1:y2, x1:x2]
        contours, _ = cv2.findContours(vehicle_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / h if h != 0 else 0
            if 5 < w < 25 and 5 < h < 25 and 0.5 < aspect_ratio < 2 and area > 20:
                indicators.append((x + x1, y + y1, w, h))

    return indicators

# === Accident Detection ===
def detect_accident(vehicle_boxes, labels_detected, frame_shape):
    has_damage_context = any(label.lower() in accident_keywords for label in labels_detected)
    if not has_damage_context:
        return False

    frame_h, frame_w = frame_shape[:2]

    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            (x1a, y1a, x2a, y2a) = vehicle_boxes[i]
            (x1b, y1b, x2b, y2b) = vehicle_boxes[j]

            xa = max(x1a, x1b)
            ya = max(y1a, y1b)
            xb = min(x2a, x2b)
            yb = min(y2a, y2b)

            if xa < xb and ya < yb:
                inter_area = (xb - xa) * (yb - ya)
                area_a = (x2a - x1a) * (y2a - y1a)
                area_b = (x2b - x1b) * (y2b - y1b)
                min_area = min(area_a, area_b)
                overlap_ratio = inter_area / min_area if min_area > 0 else 0

                if (
                    overlap_ratio > 0.3 and
                    min_area > 0.05 * frame_h * frame_w and
                    has_damage_context
                ):
                    return True
    return False

# === Analyze Frame ===
def analyze_frame(frame):
    results = model(frame, verbose=False)[0]
    frame_height, frame_width = frame.shape[:2]
    boxes_detected = []
    labels_detected = []
    vehicle_boxes = []
    alert_messages = []

    if not results.boxes:
        alert_messages.append("❌ No objects detected!")
        return frame, alert_messages

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0].item())
        label = model.names[cls]

        boxes_detected.append((x1, y1, x2, y2))
        labels_detected.append(label)
        if label in vehicle_keywords:
            vehicle_boxes.append((x1, y1, x2, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if detect_accident(vehicle_boxes, labels_detected, frame.shape):
        alert_messages.append("⚠ ACCIDENT DETECTED! VISIBLE DAMAGE & CRASH!")

    for (x1, y1, x2, y2), label in zip(boxes_detected, labels_detected):
        area = (x2 - x1) * (y2 - y1)
        dist = estimate_distance_from_area(area)

        if label in vehicle_keywords and dist < 30:
            alert_messages.append(f"ALERT: VEHICLE VERY CLOSE ({dist}m) - POSSIBLE COLLISION!")
        elif label in scene_keywords and dist < 70:
            alert_messages.append(f"ALERT: {label.upper()} ahead in {dist}m - SLOW DOWN!")
        elif label in potential_child_keywords:
            alert_messages.append(f"⚠ ALERT: {label.upper()} detected - CHILD MAY FOLLOW!")

    indicator_points = detect_indicators(frame, vehicle_boxes)
    indicator_left = False
    indicator_right = False

    for (cx, cy, w, h) in indicator_points:
        if cx < frame.shape[1] // 2:
            indicator_left = True
        else:
            indicator_right = True

    if indicator_left:
        alert_messages.append("LEFT INDICATOR ON → MOVE RIGHT")
    if indicator_right:
        alert_messages.append("RIGHT INDICATOR ON → MOVE LEFT")

    # === Limit Alerts to Max 2: Prioritize accident/collision first ===
    important_alerts = []
    secondary_alerts = []

    for alert in alert_messages:
        if "ACCIDENT" in alert or "COLLISION" in alert or "⚠" in alert:
            important_alerts.append(alert)
        else:
            secondary_alerts.append(alert)

    final_alerts = important_alerts[:1] + secondary_alerts[:1]

    if not final_alerts:
        final_alerts.append("✅ All Clear")

    return frame, final_alerts

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("🚘 Autonomous Vehicle Alert System")

option = st.radio("Choose Input Method", ["Upload Video or Image", "Use Webcam"])

# === Upload Handler ===
if option == "Upload Video or Image":
    uploaded_file = st.file_uploader("Upload video or image", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        # === Video ===
        if "video" in file_type:
            cap = cv2.VideoCapture(tfile.name)
            FRAME_WINDOW = st.image([])
            alert_box = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, alerts = analyze_frame(frame)
                FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                with alert_box.container():
                    st.markdown("### 🚨 Alerts")
                    for alert in alerts:
                        if "⚠" in alert or "ALERT" in alert or "ACCIDENT" in alert:
                            st.error(alert)
                        elif "Clear" in alert:
                            st.success(alert)
                        else:
                            st.warning(alert)
            cap.release()

        # === Image ===
        elif "image" in file_type:
            image = Image.open(tfile.name)
            frame = np.array(image)
            annotated, alerts = analyze_frame(frame)
            st.image(annotated, caption="Detected Image")

            st.markdown("### 🚨 Alerts")
            for alert in alerts:
                if "⚠" in alert or "ALERT" in alert or "ACCIDENT" in alert:
                    st.error(alert)
                elif "Clear" in alert:
                    st.success(alert)
                else:
                    st.warning(alert)

# === Webcam Handler ===
elif option == "Use Webcam":
    FRAME_WINDOW = st.image([])
    alert_box = st.empty()
    run = st.checkbox("Start Webcam")

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, alerts = analyze_frame(frame)
        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        with alert_box.container():
            st.markdown("### 🚨 Alerts")
            for alert in alerts:
                if "⚠" in alert or "ALERT" in alert or "ACCIDENT" in alert:
                    st.error(alert)
                elif "Clear" in alert:
                    st.success(alert)
                else:
                    st.warning(alert)

    cap.release()