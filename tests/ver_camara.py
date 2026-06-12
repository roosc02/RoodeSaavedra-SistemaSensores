# -*- coding: utf-8 -*-
import cv2
import sys

print("Iniciando visor de camara...")

if len(sys.argv) < 2:
    print("Uso: python ver_camara.py NUMERO")
    print("Ejemplo: python ver_camara.py 1")
    exit()

cam_id = int(sys.argv[1])

cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"No se pudo abrir la camara {cam_id}")
    exit()

print(f"Mostrando camara {cam_id}")
print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer frame.")
        break

    cv2.imshow(f"Camara {cam_id}", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Camara cerrada.")
