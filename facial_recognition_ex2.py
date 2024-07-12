import cv2

# PARAMS
#
# // minNeighbors => é a quantidade de detecções mínimas para se gerar um resultado positivo.
#
# // minSize => o tamanho mínimo do objeto a ser detectado
#
# // maxSize => é o tamanho máximo do objeto

image = cv2.imread('resources/Images/people2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier('resources/Cascades/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40),
                                            maxSize=(100, 100))

for x, y, w, h, in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)