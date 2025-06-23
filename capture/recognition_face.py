import os
import cv2
import time


face_detector = cv2.CascadeClassifier('D:\Hoc_pY\detect_face_project\haarcascade\haarcascade_frontalface_alt.xml')

cam = cv2.VideoCapture(0)
count = 0

while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3,5)
    # time.sleep(0.1)

    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y:y+h, x:x+w], (100,100))
        # cv2.imwrite('datasets_CNN/traindata/train_TTien/TTien_{}.jpg'.format(count), roi )
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 180), 1)
        count += 1

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

