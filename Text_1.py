import pyttsx3
import pytesseract
import cv2

# Initialize the text-to-speech engine (if needed for future use)
engine = pyttsx3.init()

# Open the default camera
cap = cv2.VideoCapture(0)


def detect_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get bounding boxes
    h, w, _ = frame.shape
    boxes = pytesseract.image_to_boxes(gray)
    text = pytesseract.image_to_string(gray)
    # Draw bounding boxes
    for box in boxes.splitlines():
        b = box.split()
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(frame, (x, h - y2), (x2, h - y), (0, 255, 0), 2)

    return text


if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detected_text = detect_text(frame)
        print("Detected Text:"+detected_text)
        engine.say(detected_text)

        cv2.imshow('Camera Feed', frame)
        engine.runAndWait()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#import pyttsx3
#from PIL import Image
#import pytesseract
#import cv2

#cap = cv2.VideoCapture(0)
#def detect_text(img_path):
    #image = Image.open(img_path)
    #text = pytesseract.image_to_string(image)
    #return text

#if __name__ == "__main__":
    #image_path =  "Image.jpg" # Path to your image file
    #detected_text = detect_text(image_path)
    #print("Detected Text:")
    #print(detected_text)