import cv2

image = cv2.imread('resources/Images/people1.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eye_detector = cv2.CascadeClassifier('resources/Cascades/haarcascade_eye.xml')

detections = eye_detector.detectMultiScale(gray_image, scaleFactor=1.09, minNeighbors=10, maxSize=(70, 70))

for x, y, w, h in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)