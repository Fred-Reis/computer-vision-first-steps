import cv2

image = cv2.imread('resources/Images/people1.jpg')

image = cv2.resize(image, (800, 600))

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

facial_detect = cv2.CascadeClassifier('resources/Cascades/haarcascade_frontalface_default.xml')

#Parâmetros haarcascades => scaleFactor é o fator de scala da imagem

detections = facial_detect.detectMultiScale(grey_image, scaleFactor=1.3)

for x, y, w, h in detections:
    cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0),  2)

cv2.imshow('image', image)
cv2.waitKey(0)
