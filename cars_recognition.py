import cv2

image = cv2.imread('resources/Images/car.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

car_detector = cv2.CascadeClassifier('resources/Cascades/cars.xml')

detections = car_detector.detectMultiScale(gray_image, scaleFactor=1.007, minNeighbors=5, maxSize=(70,70), minSize=(50,50))

for x, y, w, h in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)