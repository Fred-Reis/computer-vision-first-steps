import cv2
import dlib

image = cv2.imread('resources/Images/people3.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detection_harcascade = cv2.CascadeClassifier('resources/Cascades/haarcascade_frontalface_default.xml')
face_detections_hoc_dlib = dlib.get_frontal_face_detector()
face_detection_cnn_dlib = dlib.cnn_face_detection_model_v1('resources/Weights/mmod_human_face_detector.dat')

# harcascade_detections = face_detection_harcascade.detectMultiScale(gray_image)
hoc_dlib_detections = face_detections_hoc_dlib(image, 7)
# cnn_dlib_detection = face_detection_cnn_dlib(image, 4)

# for x, y, w, h in harcascade_detections:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

for face in hoc_dlib_detections:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)
#
# for face in cnn_dlib_detection:
#     l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
#     cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
