import cv2
import os
import numpy as np

cam = cv2.VideoCapture(0)
# lebar kamera
cam.set(3, 640)

# tinggi kamera
cam.set(4, 480)


face_detector = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read("training/training_result.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None', 'Harmain', 'Maria', 'Fira']

min_width = 0.1 * cam.get(3)
min_heigth = 0.1 * cam.get(4)

while True:
    # frame by frame data kamera
    ret_v, frame = cam.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(
        gray,
        1.2, 5,
        minSize=(round(min_width), round(min_heigth))
    )

    for (x, y, w, h) in face:
        frame = cv2.rectangle(
            frame, 
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        id, confidence = face_recognizer.predict(
            gray[y:y+h, x:x+w]
        )
        if confidence >= 50:
            name_id = names[id]
            confidence_text = f'{round(100 - confidence)}%'
        else:
            name_id = names[0]
            confidence_text = f'{round(100 - confidence)}%'

        cv2.putText(
            frame,
            str(name_id),
            (x+5, y-5),
            font,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            str(confidence_text),
            (x+5, y+h-5),
            font,
            1,
            (255, 255, 0),
            2
        )

    cv2.imshow('Kamera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Selesai')
cam.release()
            
