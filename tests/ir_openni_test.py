# -*- coding: utf-8 -*-
from primesense import openni2

OPENNI_PATH = r"C:\Users\roode\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin"

print("Inicializando OpenNI...")
openni2.initialize(OPENNI_PATH)

device = openni2.Device.open_any()
print("Camara detectada.")

try:
    print("Creando stream IR...")
    ir_stream = device.create_ir_stream()
    print("Stream IR creado.")

    print("Iniciando stream IR...")
    ir_stream.start()
    print("Stream IR iniciado.")

    frame = ir_stream.read_frame()
    print("Frame IR leido.")
    print("Ancho:", frame.width)
    print("Alto:", frame.height)

    ir_stream.stop()

except Exception as e:
    print("ERROR IR:")
    print(type(e))
    print(e)

finally:
    openni2.unload()
    print("OpenNI cerrado.")