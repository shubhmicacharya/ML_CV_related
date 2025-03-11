import cv2
import mediapipe as mp
import numpy as np
import  time
import pyttsx3

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

engine=pyttsx3.init()

Straight_counter = 0
Leaning_Left_counter=0
Leaning_right_counter=0
Slouching_counter=0
Severely_Slouched_counter=0
start_time = time.time()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def detect_posture(frame):
    global Straight_counter, Leaning_Left_counter, Leaning_right_counter, Slouching_counter, Severely_Slouched_counter, start_time
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = pose.process(frame_rgb)

    posture = "Unknown"
    color = (255, 255, 255)  # White

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        right_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        left_hip = [landmarks[23].x * w, landmarks[23].y * h]
        right_hip = [landmarks[24].x * w, landmarks[24].y * h]

        torso_angle = calculate_angle(left_shoulder, left_hip, right_hip)
        shoulder_diff = left_shoulder[1] - right_shoulder[1]


        if torso_angle > 160:
            posture = "Straight"
            color = (0, 255, 0)  # Green
            if time.time() - start_time >= 5:
                Straight_counter += 1
                start_time = time.time()

        elif torso_angle < 160 and torso_angle > 140:
            if shoulder_diff > 20:
                posture = "Leaning Left"
                color = (255, 165, 0)  # Orange
                if time.time() - start_time >= 5:
                    Leaning_Left_counter += 1
                    start_time = time.time()

            elif shoulder_diff < -20:
                posture = "Leaning Right"
                color = (255, 165, 0)  # Orange
                if time.time() - start_time >= 5:
                    Leaning_right_counter += 1
                    start_time = time.time()

            else:
                posture = "Slouching"
                color = (0, 0, 255)  # Red
                if time.time() - start_time >= 5:
                    Slouching_counter += 1
                    start_time = time.time()
        else:
            posture = "Severely Slouched"
            color = (0, 0, 128)  # Dark Red
            if time.time() - start_time >= 5:
                Severely_Slouched_counter += 1
                start_time = time.time()


                # Display Feedback on Screen
        cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame




cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_posture(frame)

    cv2.imshow("Posture Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Straight_counter: {Straight_counter}")
engine.say(f"Detected Straight posture with {Straight_counter} counts")

print(f"Leaning_Left_counter: {Leaning_Left_counter}")
engine.say(f"Detected Leaning Left posture with {Leaning_Left_counter} counts")

print(f"Leaning_right_counter: {Leaning_right_counter}")
engine.say(f"Detected Leaning Right posture with {Leaning_right_counter} counts")

print(f"Slouching_counter: {Slouching_counter}")
engine.say(f"Detected Slouching posture with {Slouching_counter} counts")

print(f"Severely_Slouched_counter: {Severely_Slouched_counter}")
engine.say(f"Detected Severely Slouched posture with {Severely_Slouched_counter} counts")

engine.runAndWait()
