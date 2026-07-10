# -*- coding: utf-8 -*-
import cv2
import numpy as np
from primesense import openni2

OPENNI_PATH = r"C:\Users\roode\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin"

print("Inicializando OpenNI...")
openni2.initialize(OPENNI_PATH)

device = openni2.Device.open_any()
print("Camara Orbbec detectada.")

ir_stream = device.create_ir_stream()
ir_stream.start()

print("IR iniciado.")
print("Presiona ESC para salir.")

try:
    while True:
        frame = ir_stream.read_frame()

        width = frame.width
        height = frame.height

        ir_data = np.ctypeslib.as_array(frame.get_buffer_as_uint16())
        ir_data = ir_data.reshape((height, width))

        ir_display = cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX)
        ir_display = ir_display.astype(np.uint8)

        ir_display = cv2.equalizeHist(ir_display)
        ir_color = cv2.applyColorMap(ir_display, cv2.COLORMAP_BONE)

        cv2.imshow("Orbbec IR", ir_color)

        if cv2.waitKey(1) == 27:
            break

finally:
    ir_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
    print("Camara cerrada.")