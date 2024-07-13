import cv2
import dlib

image = cv2.imread('resources/Images/people2.jpg')

face_detector_hog = dlib.get_frontal_face_detector()
detections = face_detector_hog(image, 1)
print(len(detections))
for face in detections:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)

