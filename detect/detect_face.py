import cv2
import numpy as np
import time

from tensorflow.keras import models


face_detector = cv2.CascadeClassifier(
    'D:\Hoc_pY\detect_face_project\haarcascade\haarcascade_frontalface_alt.xml')
model = models.load_model('D:\Hoc_pY\detect_face_project\models\model-preFace.h5')

dict = ['Minh', 'Chau', 'Vy']

cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y:y + h, x:x + w], (100, 100))
        rs = dict[np.argmax(model.predict(roi.reshape((-1, 100, 100, 3))))]
        predictions = model.predict(roi.reshape((-1, 100, 100, 3)))
        print(predictions)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 255, 50), 1)
        cv2.putText(frame, rs, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 25, 255), 2)

    cv2.imshow('FRAME', frame)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break