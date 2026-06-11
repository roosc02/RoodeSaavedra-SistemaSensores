# -*- coding: utf-8 -*-
import cv2
import numpy as np
from primesense import openni2

OPENNI_PATH = r"C:\Users\roode\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin"

print("Inicializando OpenNI...")
openni2.initialize(OPENNI_PATH)

device = openni2.Device.open_any()
print("Camara Orbbec detectada.")

try:
    print("Creando stream de profundidad...")
    depth_stream = device.create_depth_stream()
    print("Stream creado.")

    print("Iniciando stream de profundidad...")
    depth_stream.start()
    print("Stream iniciado.")

except Exception as e:
    print("ERROR al crear/iniciar profundidad:")
    print(type(e))
    print(e)
    openni2.unload()
    exit()

print("Depth iniciado. Presiona ESC para salir.")

try:
    while True:
        frame = depth_stream.read_frame()
        width = frame.width
        height = frame.height

        depth_data = np.ctypeslib.as_array(frame.get_buffer_as_uint16())
        depth_data = depth_data.reshape((height, width))

        center_depth = int(depth_data[height // 2, width // 2])

        depth_clipped = np.clip(depth_data, 0, 4000)
        depth_display = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / 4000.0)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        cv2.putText(depth_colormap, f"Centro: {center_depth} mm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Orbbec Depth", depth_colormap)

        if cv2.waitKey(1) == 45:
            break

finally:
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()