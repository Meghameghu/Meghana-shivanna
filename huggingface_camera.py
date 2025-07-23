import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2

# 🔄 Load the Hugging Face model & processor
model_name = "google/efficientnet-b0"  # You can change to mobilenet or resnet
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# 🎥 Open the webcam
cap = cv2.VideoCapture(0)

print("🚀 Running webcam model... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🖼 Convert frame to PIL Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # 📦 Preprocess and get predictions
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted = logits.argmax(-1).item()
    label = model.config.id2label[predicted]

    # 📝 Display the label on frame
    cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Hugging Face Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
