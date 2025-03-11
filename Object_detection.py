import cv2
import numpy as np
import pyttsx3

# Loading the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Loading class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Loading the image
image = cv2.imread("dog_cat.jpg")
height, width, channels = image.shape

# Preparing the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialization
class_ids = []
confidences = []
boxes = []

# For each detection from each output layer
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Initialize pyttsx3 engine for text-to-speech conversion
engine = pyttsx3.init()

# Convert detected labels to speech
for i in indices:
    #i = i[0]  # Ensure i is an iterable (e.g., tuple) before indexing
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    text = f"I see a {label} with confidence {confidence:.2f}"
    engine.say(text)

# Wait for speech to finish
engine.runAndWait()

# Display the image with bounding boxes
for i in indices:
    #i = i[0]  # Ensure i is an iterable (e.g., tuple) before indexing
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (255, 0, 0)  # Red color for bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
