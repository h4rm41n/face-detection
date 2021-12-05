import os
import cv2
import numpy as np
from PIL import Image


def getImageLabel(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    face_samples = []
    face_ids = []

    for image_path in image_paths:
        PILImg = Image.open(image_path).convert('L')
        img_num = np.array(PILImg, 'uint8')
        face_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_num)
        for (x, y, w, h) in faces:
            face_samples.append(img_num[y:y+h, x:x+w])
            face_ids.append(face_id)

    return face_samples, face_ids

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")

print("training wajah")

face, ids = getImageLabel('dataset')
face_recognizer.train(face,np.array(ids))

face_recognizer.write(f'training/training_result.xml')

print(f'Sebanyak {len(np.unique(ids))} data wajah telah ditraining ke mesin')