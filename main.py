import cv2
print(cv2.__version__)
img= cv2.imread('tiger.jpg',-1)
#img=cv2.rectangle(img, (128,128), (350,67), (0,255,0), 5)
print(img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('TIGER.png',img)