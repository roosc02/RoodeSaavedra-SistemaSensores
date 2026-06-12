# -*- coding: utf-8 -*-
import cv2

print("Iniciando prueba RGB con OpenCV...")

for i in range(10):
    print(f"Probando camara {i}...")

    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"  Camara {i}: no disponible")
        continue

    print(f"  Camara {i}: disponible")

    ret, frame = cap.read()

    if ret:
        print("  Frame leido correctamente")
        print(f"  Resolucion: {frame.shape}")

        cv2.imshow(f"Camara {i}", frame)
        print("  Presiona una tecla en la ventana para cerrar esta camara.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"  No se pudo leer imagen de camara {i}")

    cap.release()

print("Prueba terminada.")
