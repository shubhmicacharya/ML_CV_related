import cv2
import numpy as np
import pyttsx3

# Load YOLOv3-tiny model and classes
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open camera
cam = cv2.VideoCapture(0)

# Initialize pyttsx3 engine for text-to-speech conversion
engine = pyttsx3.init()

# Kalman Filter Initialization
state = np.array([0, 0, 0, 0], dtype='float64')  # State vector: [x, y, dx, dy]
state_cov = np.eye(4, dtype='float64')  # Initial state covariance

dt = 1  # Time step (1 frame)
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='float64')  # State transition matrix
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype='float64')  # Observation matrix
Q = np.eye(4, dtype='float64') * 0.01  # Process noise covariance
R = np.eye(2, dtype='float64') * 1  # Measurement noise covariance

FPS = 30  # Frames per second of the video
PIXELS_TO_METERS = 1 / 100  # Conversion factor: 100 pixels = 1 meter
FOCAL_LENGTH_PIXELS = 800  # Focal length in pixels (example value, this should be calibrated)

KNOWN_OBJECT_HEIGHT = 4  # Known height of the object in meters (example value)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Iterate over each of the layer outputs
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.1)

    # Convert detected labels to speech and calculate distance and velocity
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]


            # Draw bounding box with label and confidence
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0),
                          2)  # Ensure (x, y) and (x + w, y + h) are tuples of integers

            # Calculate distance to the object
            object_height_pixels = h
            distance_meters = (KNOWN_OBJECT_HEIGHT * FOCAL_LENGTH_PIXELS) / object_height_pixels

            # Convert distance to centimeters
            distance_centimeters = distance_meters * 100

            if distance_centimeters < 400:  # 4 meters = 400 centimeters
                cv2.putText(frame, "ALERT !!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 3)
                text = f"Alert !! I see a {label}"  # with confidence {confidence:.2f}"
                engine.say(text)

            else:
                text = f"I see a {label} with confidence {confidence:.2f}"
                engine.say(text)
                cv2.putText(frame, "Distance: {:.2f} cm".format(distance_centimeters), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)


            # Kalman Filter Update: Assuming one object for simplicity
            measurement = np.array([x + w / 2, y + h / 2], dtype='float64')

            # Prediction
            state = A @ state
            state_cov = A @ state_cov @ A.T + Q

            # Measurement update
            S = H @ state_cov @ H.T + R
            K = state_cov @ H.T @ np.linalg.inv(S)
            y = measurement - H @ state
            state = state + K @ y
            state_cov = state_cov - K @ H @ state_cov

            # Calculate velocity in meters per second
            velocity_pixels_per_frame = np.linalg.norm(state[2:])  # Magnitude of velocity vector
            velocity_meters_per_second = velocity_pixels_per_frame * FPS * PIXELS_TO_METERS

            # Display velocity on the frame
            cv2.putText(frame, f"Velocity: {velocity_meters_per_second:.2f} m/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 3)

    # Wait for speech to finish
    engine.runAndWait()

    # Display the processed frame
    cv2.imshow("VIDEO", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cam.release()
cv2.destroyAllWindows()
