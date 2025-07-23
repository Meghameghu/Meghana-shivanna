import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Load YOLOv8 model for object detection
yolo_model = YOLO("yolov8n.pt")  # COCO-trained

# Load ImageNet model (MobileNet) for classification
imagenet_model_name = "google/mobilenet_v2_1.0_224"
processor = AutoImageProcessor.from_pretrained(imagenet_model_name)
classifier = AutoModelForImageClassification.from_pretrained(imagenet_model_name)

# File upload GUI
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select image or video file", filetypes=[("All files", "*.*")])

# Determine input type
video_extensions = ['.mp4', '.avi', '.mov']
image_extensions = ['.jpg', '.jpeg', '.png']

is_video = any(file_path.lower().endswith(ext) for ext in video_extensions)
is_image = any(file_path.lower().endswith(ext) for ext in image_extensions)

# Video or webcam
if file_path and is_video:
    cap = cv2.VideoCapture(file_path)
elif file_path and is_image:
    frame = cv2.imread(file_path)
    cap = None
else:
    cap = cv2.VideoCapture(0)  # Webcam fallback

def crop_center_object(frame, box):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    size = min(frame.shape[0], frame.shape[1]) // 4
    half = size // 2
    start_x = max(center_x - half, 0)
    start_y = max(center_y - half, 0)
    end_x = min(center_x + half, frame.shape[1])
    end_y = min(center_y + half, frame.shape[0])

    return frame[start_y:end_y, start_x:end_x]

def process_frame(frame):
    results = yolo_model(frame)[0]
    center_object_crop = None

    for result in results.boxes:
        cls = int(result.cls)
        label = yolo_model.names[cls]
        conf = result.conf.item()
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        center_object_crop = crop_center_object(frame, result.xyxy[0])

    if center_object_crop is not None:
        pil_img = Image.fromarray(cv2.cvtColor(center_object_crop, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = classifier(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            imagenet_label = classifier.config.id2label[predicted_label]

        cv2.putText(frame, f"ImageNet: {imagenet_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    return frame

if is_image:
    output = process_frame(frame)
    cv2.imshow("Image Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow("AV Detection + Classification", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
