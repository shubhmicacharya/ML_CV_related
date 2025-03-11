import numpy as np
import cv2
cam=cv2.VideoCapture('vtest.avi')
#cam=cv2.VideoCapture(0)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=False)
while(True):
    ret, frame=cam.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    keyword=cv2.waitKey(30)
    if keyword=='q' or keyword==27:
        break
cam.release()
cv2.destroyAllWindows()