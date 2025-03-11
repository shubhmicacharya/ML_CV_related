import cv2
import numpy as np
import pyttsx3

# Initialize video capture
#cap = cv2.VideoCapture('vtest.avi')
cap = cv2.VideoCapture(0)  # Capture from default camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer to save the output video
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1000, 720))

# Read initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

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

KNOWN_OBJECT_HEIGHT = 0.5  # Known height of the object in meters (example value)

while cap.isOpened():
    # Frame preprocessing
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    measurement = None

    # Detecting objects
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 800:
            continue
        cx, cy = x + w // 2, y + h // 2
        measurement = np.array([cx, cy], dtype='float64')

        # Draw bounding box and status text
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

        # Calculate distance to the object
        object_height_pixels = h
        distance_meters = (KNOWN_OBJECT_HEIGHT * FOCAL_LENGTH_PIXELS) / object_height_pixels

        if (distance_meters < 4):
            text = f"Alert !!"# I see a {label}"  # with confidence {confidence:.2f}"
            engine.say(text)
            cv2.putText(frame1, "ALERT !!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 3)
        else:
            # Display distance on the frame
            cv2.putText(frame1, "Distance: {:.2f} centim".format(distance_meters), (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), 3)


        break  # Assume one object for simplicity

    if measurement is not None:
        # Kalman Filter Update
        # Prediction
        state = A @ state
        state_cov = A @ state_cov @ A.T + Q

        # Measurement update
        S = H @ state_cov @ H.T + R
        K = state_cov @ H.T @ np.linalg.inv(S)
        y = measurement - H @ state
        state = state + K @ y
        state_cov = state_cov - K @ H @ state_cov

    # Predict the next position (not used for drawing in this code)
    predicted_position = state[:2]

    # Drawing the estimated position
    if measurement is not None:
        cv2.circle(frame1, (int(predicted_position[0]), int(predicted_position[1])), 5, (255, 0, 0), -1)

        # Calculate velocity in pixels per frame
        velocity_pixels_per_frame = np.linalg.norm(state[2:])  # Magnitude of velocity vector

        # Convert velocity to meters per second
        velocity_meters_per_second = velocity_pixels_per_frame * FPS * PIXELS_TO_METERS

        # Convert velocity to kilometers per hour
        velocity_km_per_hour = velocity_meters_per_second * 3.6

        # Display velocity in meters per second and kilometers per hour
        cv2.putText(frame1, "Velocity: {:.2f} m/s".format(velocity_meters_per_second), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 3)
        #cv2.putText(frame1, "Velocity: {:.2f} km/h".format(velocity_km_per_hour), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    #1, (255, 255, 255), 3)

    # Resize and write the output frame
    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    # Wait for speech to finish
    engine.runAndWait()
    cv2.imshow("feed", frame1)

    # Read the next frame
    frame1 = frame2
    ret, frame2 = cap.read()

    # Break the loop on 'Esc' key press
    if cv2.waitKey(40) == 27:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
out.release()
