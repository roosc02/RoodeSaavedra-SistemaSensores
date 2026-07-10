import cv2
import numpy as np
import time
from primesense import openni2

OPENNI_PATH = r"C:\Users\roode\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin"

WINDOW_NAME = "Orbbec IR Tracker - Estable"

ir_stream = None
fullscreen = False
tracking_enabled = True
use_colormap = True

frame_errors = 0
last_key_time = 0


def get_direction(obj_x, obj_y, img_w, img_h, tolerance=70):
    center_x = img_w // 2
    center_y = img_h // 2

    dx = obj_x - center_x
    dy = obj_y - center_y

    horizontal = "CENTRO"
    vertical = "CENTRO"

    if dx < -tolerance:
        horizontal = "MOVER IZQUIERDA"
    elif dx > tolerance:
        horizontal = "MOVER DERECHA"

    if dy < -tolerance:
        vertical = "MOVER ARRIBA"
    elif dy > tolerance:
        vertical = "MOVER ABAJO"

    if horizontal == "CENTRO" and vertical == "CENTRO":
        return "OBJETO CENTRADO"

    if horizontal == "CENTRO":
        return vertical

    if vertical == "CENTRO":
        return horizontal

    return horizontal + " / " + vertical


try:
    print("Inicializando OpenNI...")
    openni2.initialize(OPENNI_PATH)

    device = openni2.Device.open_any()
    print("Camara Orbbec detectada.")

    print("Creando stream IR...")
    ir_stream = device.create_ir_stream()

    print("Iniciando stream IR...")
    ir_stream.start()

    time.sleep(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1000, 750)

    print("IR Tracker iniciado.")
    print("Controles:")
    print("X = salir")
    print("F = pantalla completa")
    print("T = activar/desactivar tracking")
    print("C = activar/desactivar mapa de color")
    print("Nota: ESC y Q ya NO cierran el programa.")

    while True:
        try:
            frame = ir_stream.read_frame()

            width = frame.width
            height = frame.height

            ir_data = np.ctypeslib.as_array(frame.get_buffer_as_uint16())
            ir_data = ir_data.reshape((height, width))

            ir_display = cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX)
            ir_display = ir_display.astype(np.uint8)

            ir_blur = cv2.GaussianBlur(ir_display, (7, 7), 0)

            if use_colormap:
                output = cv2.applyColorMap(ir_display, cv2.COLORMAP_BONE)
            else:
                output = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)

            img_h, img_w = ir_display.shape
            center_x = img_w // 2
            center_y = img_h // 2

            cv2.line(output, (center_x - 25, center_y), (center_x + 25, center_y), (0, 255, 0), 2)
            cv2.line(output, (center_x, center_y - 25), (center_x, center_y + 25), (0, 255, 0), 2)
            cv2.circle(output, (center_x, center_y), 70, (0, 255, 0), 2)

            status_text = "Tracking OFF"

            if tracking_enabled:
                threshold_value = int(np.percentile(ir_blur, 97))
                _, mask = cv2.threshold(ir_blur, threshold_value, 255, cv2.THRESH_BINARY)

                kernel = np.ones((8, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)

                    if area > 80:
                        x, y, w, h = cv2.boundingRect(largest)

                        obj_x = x + w // 2
                        obj_y = y + h // 2

                        direction = get_direction(obj_x, obj_y, img_w, img_h)

                        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.circle(output, (obj_x, obj_y), 8, (0, 0, 255), -1)
                        cv2.line(output, (center_x, center_y), (obj_x, obj_y), (255, 0, 0), 2)

                        status_text = direction

                        cv2.putText(
                            output,
                            f"x={obj_x}, y={obj_y}, area={int(area)}",
                            (20, img_h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                    else:
                        status_text = "OBJETO PEQUENO / RUIDO"
                else:
                    status_text = "SIN OBJETO DETECTADO"

            cv2.putText(output, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(output, "X salir | F fullscreen | T tracking | C color",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.putText(output, f"Errores frame: {frame_errors}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(30) & 0xFF
            now = time.time()

            # Antirrebote: evita que una tecla se active muchas veces seguidas
            if key != 255 and now - last_key_time > 0.35:
                last_key_time = now

                if key == ord("x"):
                    print("Cerrando por tecla X...")
                    break

                elif key == ord("f"):
                    fullscreen = not fullscreen
                    print("Fullscreen:", "ON" if fullscreen else "OFF")

                    if fullscreen:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

                elif key == ord("t"):
                    tracking_enabled = not tracking_enabled
                    print("Tracking:", "ON" if tracking_enabled else "OFF")

                elif key == ord("c"):
                    use_colormap = not use_colormap
                    print("Mapa de color:", "ON" if use_colormap else "OFF")

        except Exception as frame_error:
            frame_errors += 1
            print("Error de frame, pero sigo ejecutando...")
            print(type(frame_error))
            print(frame_error)
            time.sleep(0.1)
            continue

except KeyboardInterrupt:
    print("Interrumpido con Ctrl+C.")

except Exception as e:
    print("ERROR GENERAL:")
    print(type(e))
    print(e)

finally:
    print("Liberando recursos...")

    try:
        if ir_stream is not None:
            ir_stream.stop()
    except Exception as e:
        print("No se pudo detener IR:", e)

    try:
        openni2.unload()
    except Exception as e:
        print("No se pudo descargar OpenNI:", e)

    cv2.destroyAllWindows()
    time.sleep(1)

    print("Camara cerrada.")