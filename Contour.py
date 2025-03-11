import cv2
import numpy as np

# Read the image
image = cv2.imread('img.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
# -1 means drawing all contours
cv2.drawContours(image, contours, 10, (0, 255, 0), 2)
print(str(len(contours)))
for i in range (len(contours)):
    print(contours[i])
#Display the original image with contours
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
