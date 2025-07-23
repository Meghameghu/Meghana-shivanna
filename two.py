# ✅ INSTALL DEPENDENCIES (Run in terminal if needed)
# pip install ultralytics transformers opencv-python pillow streamlit

import streamlit as st
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile
import os
import asyncio
import time
import math

# Patch for asyncio issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load Models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
yolo_model = YOLO("yolov8n.pt")  # Using YOLOv8n for better performance

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Enhanced zero-shot prompts for scene classification
prompts = [
    "a safe driving scene",
    "a pedestrian is crossing suddenly",
    "a vehicle is about to crash",
    "two vehicles have collided",
    "a child might run into the road",
    "a cyclist is dangerously close",
    "a fallen object is on the road",
    "a fire or smoke ahead",
    "a construction site on the road",
    "an animal crossing the road",
    "a dangerous situation ahead",
    "a ball rolling on the road",
    "a vehicle approaching rapidly"
]

# Constants for distance estimation
KNOWN_VEHICLE_WIDTH = 1.8  # Average car width in meters
FOCAL_LENGTH = 1000        # Approximate focal length in pixels
DANGER_DISTANCE = 50       # meters
WARNING_DISTANCE = 100     # meters

# Classify scene with CLIP
@st.cache_resource
def classify_scene_with_clip(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(text=prompts, images=pil_image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().detach().cpu().numpy()
    ranked = sorted(zip(prompts, probs), key=lambda x: x[1], reverse=True)
    return ranked

# Lane detection using color thresholding and Hough transform
def detect_lanes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 40, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    mask_poly = np.zeros_like(edges)
    polygon = np.array([[
        (int(width*0.1), height),
        (int(width*0.9), height),
        (int(width*0.55), int(height*0.6)),
        (int(width*0.45), int(height*0.6))
    ]], np.int32)
    cv2.fillPoly(mask_poly, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask_poly)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    lane_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return cv2.addWeighted(frame, 0.8, lane_image, 1, 1)

# Detect vehicle indicator lights (improved detection)
def detect_vehicle_indicators(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    indicator_left = indicator_right = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            if x < frame.shape[1] // 2:
                indicator_left = True
            else:
                indicator_right = True
    return indicator_left, indicator_right

# Estimate distance to objects using known width and apparent size in image
def estimate_distance(pixel_width):
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_VEHICLE_WIDTH * FOCAL_LENGTH) / pixel_width

# Enhanced risk assessment with distance estimation
def assess_risks(results, frame_width, frame_height):
    risks = []
    vehicle_classes = ["car", "truck", "bus", "motorcycle"]
    
    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy())
            class_name = results.names[cls]
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            width = x2 - x1
            height = y2 - y1
            
            # Calculate position in frame (0-1 from left to right)
            x_center = (x1 + x2) / 2 / frame_width
            y_center = (y1 + y2) / 2 / frame_height
            
            # Estimate distance for vehicles
            if class_name in vehicle_classes:
                distance = estimate_distance(width)
                
                # Determine if vehicle is approaching (growing in size)
                risk_level = "none"
                if distance < DANGER_DISTANCE:
                    risk_level = "danger"
                elif distance < WARNING_DISTANCE:
                    risk_level = "warning"
                
                risks.append({
                    "class": class_name,
                    "distance": distance,
                    "risk_level": risk_level,
                    "position": (x_center, y_center),
                    "box": (x1, y1, x2, y2)
                })
            elif class_name in ["person", "bicycle"]:
                # Pedestrians and cyclists are always high risk
                risks.append({
                    "class": class_name,
                    "distance": None,
                    "risk_level": "danger",
                    "position": (x_center, y_center),
                    "box": (x1, y1, x2, y2)
                })
            elif class_name == "sports ball":  # Fixed ball detection
                risks.append({
                    "class": class_name,
                    "distance": None,
                    "risk_level": "warning",
                    "position": (x_center, y_center),
                    "box": (x1, y1, x2, y2)
                })
    
    return risks

# Analyze frame with enhanced detection and risk assessment
@st.cache_data(show_spinner=False)
def analyze_frame(frame):
    results = yolo_model(frame, verbose=False)[0]
    classes = results.names
    detected = [classes[int(cls)] for cls in results.boxes.cls.cpu().numpy()] if results.boxes is not None else []
    annotated_frame = frame.copy()

    # Draw bounding boxes with risk assessment
    risks = assess_risks(results, frame.shape[1], frame.shape[0])
    
    for risk in risks:
        x1, y1, x2, y2 = risk["box"]
        class_name = risk["class"]
        
        # Set color based on risk level
        if risk["risk_level"] == "danger":
            color = (0, 0, 255)  # Red
            label = f"{class_name} - DANGER"
        elif risk["risk_level"] == "warning":
            color = (0, 165, 255)  # Orange
            label = f"{class_name} - WARNING"
        else:
            color = (0, 255, 0)  # Green
            label = f"{class_name}"
            
        # Add distance if available
        if risk["distance"] is not None:
            label += f" ({risk['distance']:.1f}m)"
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Scene classification with CLIP
    clip_results = classify_scene_with_clip(frame)
    top_clip = clip_results[0][0]
    top_prob = clip_results[0][1]

    # Vehicle indicator detection
    indicator_left, indicator_right = detect_vehicle_indicators(frame)
    
    # Lane detection
    lane_frame = detect_lanes(annotated_frame)

    # Determine warnings and alerts
    warning = None
    alert_message = None
    
    # Check for immediate dangers
    danger_objects = [r for r in risks if r["risk_level"] == "danger"]
    if danger_objects:
        closest = min((r for r in danger_objects if r["distance"] is not None), 
                      key=lambda x: x["distance"] if x["distance"] is not None else float('inf'), 
                      default=None)
        if closest:
            warning = f"⚠ IMMEDIATE DANGER: {closest['class']} at {closest['distance']:.1f}m"
            alert_message = "🛑 EMERGENCY STOP - Collision imminent!"
        else:
            warning = f"⚠ PEDESTRIAN/CYCLIST DETECTED"
            alert_message = "🛑 SLOW DOWN - Vulnerable road user nearby"
    # Check for warnings
    elif any(r["risk_level"] == "warning" for r in risks):
        warning_objects = [r for r in risks if r["risk_level"] == "warning"]
        if any(r["class"] == "sports ball" for r in warning_objects):
            warning = "⚠ Ball detected on road"
            alert_message = "🚨 Caution: Child may follow the ball"
        else:
            warning = "⚠ Potential hazard detected"
            alert_message = "🚨 Caution: Slow down and be prepared to stop"
    # Check vehicle indicators
    elif (indicator_left or indicator_right) and any(obj in detected for obj in ["car", "truck", "bus", "motorcycle"]):
        warning = f"⚠ Vehicle indicating turn {'left' if indicator_left else 'right'}"
        alert_message = "🚨 Caution: Nearby vehicle may change lanes"
    # Check scene classification
    elif top_prob > 0.6 and top_clip in ["a child might run into the road", "a vehicle approaching rapidly"]:
        warning = f"⚠ Scene analysis: {top_clip}"
        alert_message = "🚨 Caution: Potential hazard ahead"

    return lane_frame, detected, clip_results, warning, alert_message, risks

# Streamlit UI
st.set_page_config(page_title="Enhanced Driving Scene Risk Analyzer", layout="centered")
st.title("🚗 Enhanced Driving Scene Risk Analyzer")
st.markdown("""
    Upload an image or video to analyze potential driving risks with:
    - Object detection with distance estimation
    - Scene understanding with CLIP
    - Lane detection
    - Vehicle indicator detection
""")

input_type = st.radio("Select Input Type", ["Image", "Video"])
file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if file:
    if input_type == "Image":
        image = Image.open(file).convert("RGB")
        frame = np.array(image)[:, :, ::-1].copy()
        analyzed_frame, objects, clip_scores, warning, alert_message, risks = analyze_frame(frame)
        analyzed_frame_rgb = cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(analyzed_frame_rgb, caption="Analyzed Scene", use_container_width=True)
        with col2:
            st.subheader("Analysis Results")
            st.write(f"**Top Scene Classification:** {clip_scores[0][0]} ({clip_scores[0][1]:.2f} confidence)")
            
            if risks:
                st.write("**Detected Risks:**")
                for risk in risks:
                    st.write(f"- {risk['class']}: {risk['risk_level'].upper()}" + 
                            (f" ({risk['distance']:.1f}m away)" if risk['distance'] else ""))
            
            if warning:
                st.error(warning)
                if alert_message:
                    st.warning(alert_message)
            else:
                st.success("✅ Safe driving scene")

    elif input_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        warning_placeholder = st.empty()
        alert_placeholder = st.empty()
        risk_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            analyzed_frame, _, clip_scores, warning, alert_message, risks = analyze_frame(frame)
            frame_rgb = cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Display analysis information
            info_text = f"**Scene:** {clip_scores[0][0]} ({clip_scores[0][1]:.2f})"
            if risks:
                info_text += "\n**Risks:** " + ", ".join(
                    f"{r['class']}({r['risk_level'][0]})" + 
                    (f" {r['distance']:.1f}m" if r['distance'] else "")
                    for r in risks
                )
            info_placeholder.info(info_text)
            
            if warning:
                warning_placeholder.error(warning)
                if alert_message:
                    alert_placeholder.warning(alert_message)
            else:
                warning_placeholder.empty()
                alert_placeholder.empty()
            
            time.sleep(0.1)  # Control playback speed

        cap.release()
        os.unlink(tfile.name)