# -*- coding: utf-8 -*-
import cv2

CAMARA_ORBBEC_RGB = 0

cap = cv2.VideoCapture(CAMARA_ORBBEC_RGB, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("No se pudo abrir la camara Orbbec RGB.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Orbbec Astra RGB funcionando.")
print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer imagen.")
        break

    cv2.imshow("Orbbec Astra RGB", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Camara cerrada.")
