import cv2
import dlib

image = cv2.imread('resources/Images/people2.jpg')
cnn_face_detector = dlib.cnn_face_detection_model_v1('resources/Weights/mmod_human_face_detector.dat')

detections = cnn_face_detector(image, 1)

for face in detections:
    l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
