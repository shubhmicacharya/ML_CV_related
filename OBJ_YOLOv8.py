import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose the appropriate YOLOv8 model (n - nano, s - small, m - medium, l - large, x - extra large)

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start video capture
cam = cv2.VideoCapture(0)
# Set camera properties if needed
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Process the results
    for result in results:
        boxes = result.boxes  # Extract boxes from the result
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
            confidence = box.conf[0]  # Extract confidence
            class_id = int(box.cls[0])  # Extract class id
            label = classes[class_id]

            if confidence > 0.5:  # Confidence threshold
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("VIDEO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
