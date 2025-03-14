import numpy as np
import cv2 as cv

# Open the video file
#cap = cv.VideoCapture('vtest.avi')
cap = cv.VideoCapture('slow_traffic_small.mp4')

# Take the first frame of the video
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    exit()

# Setup initial location of the window
x, y, width, height = 300, 200, 100, 50
track_window = (x, y, width, height)

# Set up the ROI for tracking
roi = frame[y:y + height, x:x + width]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations or move by at least 1 point
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

# Display the region of interest (ROI)
cv.imshow('ROI', roi)

while True:
    ret, frame = cap.read()
    if ret:
        # Convert the frame to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Perform back projection
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply MeanShift to get the new location
        #ret, track_window = cv.meanShift(dst, track_window, term_crit)
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw the tracking rectangle on the image
        x, y, w, h = track_window
        final_image = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 3)

        # Display the frames
        cv.imshow('Back Projection', dst)
        cv.imshow('Tracked Image', final_image)

        # Break the loop if 'ESC' is pressed
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
