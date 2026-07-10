import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from primesense import openni2


OPENNI_PATH = (
    r"C:\Users\roode\Downloads"
    r"\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\samples\bin"
)

RGB_INDEX = 0

PANEL_WIDTH = 640
PANEL_HEIGHT = 480

MAX_CAPTURES = 20
CAPTURE_INTERVAL_SECONDS = 5.0

CAPTURE_DIR = Path("captures")


class Kalman1D:
    """
    Filtro Kalman simple para suavizar una medición numérica.
    Sirve para profundidad, intensidad IR o promedio RGB.
    """

    def __init__(self, process_noise=1e-2, measurement_noise=10.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.estimate = 0.0
        self.error = 1.0
        self.initialized = False

    def update(self, measurement):
        measurement = float(measurement)

        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
            return self.estimate

        # Predicción
        self.error = self.error + self.process_noise

        # Ganancia de Kalman
        kalman_gain = self.error / (
            self.error + self.measurement_noise
        )

        # Corrección
        self.estimate = self.estimate + kalman_gain * (
            measurement - self.estimate
        )

        self.error = (1.0 - kalman_gain) * self.error

        return self.estimate


class KalmanImage:
    """
    Filtro Kalman simple aplicado a una imagen completa.
    Suaviza cada píxel usando información de frames anteriores.
    """

    def __init__(self, process_noise=1e-2, measurement_noise=20.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.estimate = None
        self.error = None

    def update(self, measurement):
        measurement = measurement.astype(np.float32)

        if self.estimate is None:
            self.estimate = measurement.copy()
            self.error = np.ones_like(measurement, dtype=np.float32)
            return self.estimate

        self.error = self.error + self.process_noise

        kalman_gain = self.error / (
            self.error + self.measurement_noise
        )

        self.estimate = self.estimate + kalman_gain * (
            measurement - self.estimate
        )

        self.error = (1.0 - kalman_gain) * self.error

        return self.estimate



def normalizar_16_bits(image):
    visible = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return visible.astype(np.uint8)


def extraer_bordes_rgb(rgb_image):
    """
    Extrae bordes de la imagen RGB usando Canny.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    edges_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR,
    )

    return edges_bgr


def extraer_bordes_infrarrojo(ir_visible):
    """
    Extrae bordes de la imagen infrarroja usando Canny.
    """
    edges = cv2.Canny(
        ir_visible,
        40,
        140,
    )

    edges_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR,
    )

    return edges_bgr


def extraer_bordes_profundidad(depth_filtered):
    """
    Extrae bordes por cambios de profundidad.
    Es una aproximación simple a bordes de profundidad.
    """
    depth_8 = normalizar_16_bits(depth_filtered)

    edges = cv2.Canny(
        depth_8,
        40,
        120,
    )

    edges_bgr = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR,
    )

    return edges_bgr


def obtener_datos_profundidad(depth_image):
    valid_depth = depth_image[depth_image > 0]

    if valid_depth.size == 0:
        return 0, 0, 0.0

    minimum = int(valid_depth.min())
    maximum = int(valid_depth.max())
    average = float(valid_depth.mean())

    return minimum, maximum, average


def escribir_texto(image, lines):
    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1], 125),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        image,
        0.35,
        0,
        image,
    )

    y_position = 25

    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y_position += 23

def superponer_bordes(image, edges, color=(0, 255, 0)):
                """
                Superpone los bordes detectados sobre la imagen original.
                Los bordes se dibujan en color verde por defecto.
                """
                if len(image.shape) == 2:
                    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    base = image.copy()

                if len(edges.shape) == 3:
                    edges_gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
                else:
                    edges_gray = edges

                overlay = base.copy()
                overlay[edges_gray > 0] = color

                return overlay


def main():
    rgb_camera = None
    depth_stream = None
    ir_stream = None
    openni_initialized = False

    rgb_frames = 0
    depth_frames = 0
    ir_frames = 0

    rgb_fps = 0.0
    depth_fps = 0.0
    ir_fps = 0.0

    last_report_time = time.perf_counter()
    last_capture_time = last_report_time
    capture_count = 0

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Kalman aplicado a los datos numéricos
    kalman_depth = Kalman1D(
        process_noise=5.0,
        measurement_noise=80.0,
    )

    kalman_ir = Kalman1D(
        process_noise=2.0,
        measurement_noise=50.0,
    )

    kalman_b = Kalman1D(
        process_noise=1.0,
        measurement_noise=25.0,
    )

    kalman_g = Kalman1D(
        process_noise=1.0,
        measurement_noise=25.0,
    )

    kalman_r = Kalman1D(
        process_noise=1.0,
        measurement_noise=25.0,
    )

    # Kalman aplicado a la imagen completa
    kalman_depth_image = KalmanImage(
        process_noise=2.0,
        measurement_noise=80.0,
    )

    kalman_ir_image = KalmanImage(
        process_noise=1.0,
        measurement_noise=40.0,
    )

    try:
        print(f"Abriendo flujo RGB en índice {RGB_INDEX}")

        rgb_camera = cv2.VideoCapture(
            RGB_INDEX,
            cv2.CAP_MSMF,
        )

        if not rgb_camera.isOpened():
            raise RuntimeError(
                f"No se pudo abrir RGB en el índice {RGB_INDEX}."
            )

        rgb_camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            PANEL_WIDTH,
        )

        rgb_camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            PANEL_HEIGHT,
        )

        rgb_camera.set(
            cv2.CAP_PROP_FPS,
            30,
        )

        print("Flujo RGB iniciado")

        print("Inicializando OpenNI")
        openni2.initialize(OPENNI_PATH)
        openni_initialized = True

        device = openni2.Device.open_any()
        print("Dispositivo ORBBEC abierto.")

        depth_stream = device.create_depth_stream()
        ir_stream = device.create_ir_stream()

        depth_stream.start()
        print("Flujo de profundidad iniciado.")

        ir_stream.start()
        print("Flujo infrarrojo iniciado.")

        depth_mode = depth_stream.get_video_mode()
        ir_mode = ir_stream.get_video_mode()

        print(
            f"Profundidad: "
            f"{depth_mode.resolutionX}x{depth_mode.resolutionY} "
            f"@ {depth_mode.fps} FPS"
        )

        print(
            f"Infrarrojo: "
            f"{ir_mode.resolutionX}x{ir_mode.resolutionY} "
            f"@ {ir_mode.fps} FPS"
        )


        while True:
            #RGB
            rgb_ok, rgb_image = rgb_camera.read()

            if not rgb_ok or rgb_image is None:
                raise RuntimeError(
                    "No se pudo recibir el cuadro RGB"
                )

            rgb_frames += 1

            # Profundidad
            depth_frame = depth_stream.read_frame()

            depth_image = np.frombuffer(
                depth_frame.get_buffer_as_uint16(),
                dtype=np.uint16,
            ).reshape(
                depth_frame.height,
                depth_frame.width,
            )
            depth_filtered = kalman_depth_image.update(depth_image)
            depth_frames += 1

            #  Infrarrojo 
            ir_frame = ir_stream.read_frame()

            ir_image = np.frombuffer(
                ir_frame.get_buffer_as_uint16(),
                dtype=np.uint16,
            ).reshape(
                ir_frame.height,
                ir_frame.width,
            )

            ir_filtered = kalman_ir_image.update(ir_image)

            ir_frames += 1

            # Calcular FPS una vez por segundo
            current_time = time.perf_counter()
            elapsed = current_time - last_report_time

            if elapsed >= 1.0:
                rgb_fps = rgb_frames / elapsed
                depth_fps = depth_frames / elapsed
                ir_fps = ir_frames / elapsed

                rgb_frames = 0
                depth_frames = 0
                ir_frames = 0

                last_report_time = current_time

            # Datos RGB
            blue_average = float(
                rgb_image[:, :, 0].mean()
            )

            green_average = float(
                rgb_image[:, :, 1].mean()
            )

            red_average = float(
                rgb_image[:, :, 2].mean()
            )

            blue_kalman = kalman_b.update(blue_average)
            green_kalman = kalman_g.update(green_average)
            red_kalman = kalman_r.update(red_average)

            # Datos de profundidad
            depth_min, depth_max, depth_average = (
                obtener_datos_profundidad(depth_filtered)
            )

            depth_kalman = kalman_depth.update(depth_average)

            # Datos infrarrojos
            ir_min = int(ir_filtered.min())
            ir_max = int(ir_filtered.max())
            ir_average = float(ir_filtered.mean())
            ir_kalman = kalman_ir.update(ir_average)

           #visualizaciones
            rgb_visible = cv2.resize(
                rgb_image,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            depth_visible = normalizar_16_bits(
                depth_filtered
            )

            depth_visible = cv2.applyColorMap(
                depth_visible,
                cv2.COLORMAP_HSV,
            )

            depth_visible = cv2.resize(
                depth_visible,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            ir_visible = normalizar_16_bits(
                ir_filtered
            )

            ir_visible = cv2.cvtColor(
                ir_visible,
                cv2.COLOR_GRAY2BGR,
            )

            ir_visible = cv2.resize(
                ir_visible,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            # Copias limpias para extraer bordes sin texto encima
            rgb_clean = rgb_visible.copy()
            depth_clean = depth_visible.copy()
            ir_clean = ir_visible.copy()

              # Bordes de cada sensor
            rgb_edges = extraer_bordes_rgb(rgb_visible)

            depth_edges = extraer_bordes_profundidad(
                depth_filtered
            )

            ir_edges = extraer_bordes_infrarrojo(
                normalizar_16_bits(ir_filtered)
            )


            
            # Ajustar bordes al mismo tamaño de los paneles
            rgb_edges = cv2.resize(
                rgb_edges,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            depth_edges = cv2.resize(
                depth_edges,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            ir_edges = cv2.resize(
                ir_edges,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            # Superponer bordes sobre cada imagen real
            rgb_with_edges = superponer_bordes(
                rgb_visible,
                rgb_edges,
                color=(0, 255, 0),
            )

            depth_with_edges = superponer_bordes(
                depth_visible,
                depth_edges,
                color=(0, 255, 0),
            )

            ir_with_edges = superponer_bordes(
                ir_visible,
                ir_edges,
                color=(0, 255, 0),
            )



          

            # Ajustar bordes al mismo tamaño
            rgb_edges = cv2.resize(
                rgb_edges,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            depth_edges = cv2.resize(
                depth_edges,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            ir_edges = cv2.resize(
                ir_edges,
                (PANEL_WIDTH, PANEL_HEIGHT),
            )

            


            # Datos sobre RGB
            escribir_texto(
                rgb_with_edges,
                [
                    "RGB",
                    (
                        f"Resolucion: "
                        f"{rgb_image.shape[1]}x"
                        f"{rgb_image.shape[0]}"
                    ),
                    f"FPS: {rgb_fps:.2f}",
                    ( 
                        f"BGR Kalman: " 
                        f"{blue_kalman:.1f}, " 
                        f"{green_kalman:.1f}, " 
                        f"{red_kalman:.1f}" ),
                ],
            )

            # Datos sobre profundidad
            escribir_texto(
                depth_with_edges,
                [
                    "PROFUNDIDAD",
                    (
                        f"Resolucion: "
                        f"{depth_frame.width}x"
                        f"{depth_frame.height}"
                    ),
                    f"FPS: {depth_fps:.2f}",
                    (
                        f"Min: {depth_min} mm | "
                        f"Max: {depth_max} mm"
                    ),
                    f"Promedio: {depth_average:.1f} mm",
                    f"Kalman: {depth_kalman:.1f} mm"
                ],
            )

            # Datos sobre infrarrojo
            escribir_texto(
                ir_with_edges,
                [
                    "INFRARROJO",
                    (
                        f"Resolucion: "
                        f"{ir_frame.width}x"
                        f"{ir_frame.height}"
                    ),
                    f"FPS: {ir_fps:.2f}",
                    (
                        f"Min: {ir_min} | "
                        f"Max: {ir_max}"
                    ),
                    f"Promedio: {ir_average:.1f}",
                    f"Kalman: {ir_kalman:.1f}",
                ],
            )
            # Unir las tres imágenes con bordes superpuestos en una sola pantalla
            combined_view = np.hstack(
                (
                    depth_with_edges,
                    rgb_with_edges,
                    ir_with_edges,
                )
            )
           
            # Guardar capturas automáticamente cada cierto tiempo
            current_capture_time = time.perf_counter()

            if (
                capture_count < MAX_CAPTURES
                and current_capture_time - last_capture_time >= CAPTURE_INTERVAL_SECONDS
            ):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                capture_number = capture_count + 1

                rgb_path = CAPTURE_DIR / f"{capture_number:02d}_rgb_{timestamp}.png"
                depth_path = CAPTURE_DIR / f"{capture_number:02d}_profundidad_{timestamp}.png"
                ir_path = CAPTURE_DIR / f"{capture_number:02d}_infrarrojo_{timestamp}.png"
                combined_path = CAPTURE_DIR / f"{capture_number:02d}_combinada_{timestamp}.png"

                cv2.imwrite(str(rgb_path), rgb_with_edges)
                cv2.imwrite(str(depth_path), depth_with_edges)
                cv2.imwrite(str(ir_path), ir_with_edges)
                cv2.imwrite(str(combined_path), combined_view)

                capture_count += 1
                last_capture_time = current_capture_time

                print(f"Captura {capture_count}/{MAX_CAPTURES} guardada.")

                if capture_count == MAX_CAPTURES:
                    print("Se han tomadod las 20 imágenes")


            cv2.imshow(
                "ORBBEC Astra+ | RGB - Profundidad - Infrarrojo",
                combined_view,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except Exception as error:
        print(f"\nError: {error}")

    finally:
        if ir_stream is not None:
            try:
                ir_stream.stop()
            except Exception:
                pass

        if depth_stream is not None:
            try:
                depth_stream.stop()
            except Exception:
                pass

        if rgb_camera is not None:
            rgb_camera.release()

        if openni_initialized:
            try:
                openni2.unload()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
