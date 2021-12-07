from kalmanfilter import KalmanFilter
import cv2

#mengambil kalman filter
kf = KalmanFilter()

im = cv2.imread("back.jpg")

posisibola =  [(50,100),(100,100),(150,100),(200,100),(250,100),(300,100),(350,100), (400,100), (450,100)]

for pb in posisibola:
    cv2.circle(im, pb, 15, (0,0,255),-1)

    prediksi = kf.predict(pb[0],pb[1])#memprediksi bolanya
    cv2.circle(im,prediksi,15,(0,255,0),4)


for i in range(10):
    prediksi = kf.predict(prediksi[0],prediksi[1])
    cv2.circle(im, prediksi, 15, (20, 220, 0), 4)

cv2.imshow("Gambar",im)
cv2.waitKey(0)