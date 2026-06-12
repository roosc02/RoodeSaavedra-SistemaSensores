# -*- coding: utf-8 -*-
import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("No se pudo abrir la camara.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camara RGB/USB iniciada. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer frame.")
        break

    cv2.imshow("Orbbec USB / RGB", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()