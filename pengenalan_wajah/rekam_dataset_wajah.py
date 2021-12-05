import cv2
import time

cam = cv2.VideoCapture(0)

# lebar camera
cam.set(3, 640)

# tinggi camera
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')


face_id = input("Masukkan Face ID: ")
print("Harap menghadap camera dan tunggu sampai pengambilan data selesai ....")

# module time.sleep supaya ada jeda dalam pengambilan gambar
time.sleep(3)

counter_data = 1
while True:
    retV, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        frame = cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,255),
            2
        )

        # save file wajah ke dalam folder dataset
        filename = f'face.{face_id}.{counter_data}.jpg'
        cv2.imwrite(f'dataset/{filename}', frame)
        counter_data += 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(
                roi_color,
                (xe, ye),
                (xe+we, ye+he),
                (0, 0, 255),
                1
            )

    cv2.imshow('Kamera' , frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif counter_data >= 30:
        break

print("Recording data selesai")
cam.release()