print("Iniciando prueba Orbbec...")

from pyorbbecsdk import Context

ctx = Context()
device_list = ctx.query_devices()

count = device_list.get_count()
print("Cámaras detectadas:", count)

if count == 0:
    print("No se detectó ninguna cámara Orbbec.")
    print("Revisa cable USB, puerto USB o driver.")
else:
    device = device_list.get_device_by_index(0)
    info = device.get_device_info()

    print("Nombre:", info.get_name())
    print("Serial:", info.get_serial_number())
    print("PID:", info.get_pid())
    print("VID:", info.get_vid())