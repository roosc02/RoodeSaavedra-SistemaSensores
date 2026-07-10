
"""
Detector de objetos para asistencia a la conducción.

Se utiliza:
- Cámara RGB Orbbec mediante OpenCV.
- Un modelo YOLO V26 ULTRALYTICS preentrenado.
- Filtrado de objetos relevantes para conducción.
- Selección del objeto que representa mayor riesgo.
- Orientación izquierda, centro o derecha.
- Estimación aproximada de rango por tamaño visual.

Objetos contemplados:
- Persona
- Bicicleta
- Automóvil
- Motocicleta
- Autobús
- Camión
- Semáforo
- Señal de alto

"""

import time

import cv2
from ultralytics import YOLO


# CONFIGURACIÓN

CAMERA_INDEX = 0

WINDOW_NAME = "Orbbec RGB - Deteccion para conduccion"

# Modelo para funcionamiento en tiempo real.
# La primera ejecución descargará automáticamente el archivo.
MODEL_PATH = "yolo26n.pt"

# Confianza mínima aceptada.
CONFIDENCE_THRESHOLD = 0.45

# Tamaño usado por YOLO durante la inferencia.
INFERENCE_SIZE = 640

# Zona central considerada como trayectoria frontal.
CENTER_TOLERANCE = 75

# Límites de rango aproximado según el porcentaje de imagen
# ocupado por el objeto.
FAR_LIMIT = 0.025
MEDIUM_LIMIT = 0.12

# Clases de COCO.
TARGET_CLASSES = {
    "person": "PERSONA",
    "bicycle": "BICICLETA",
    "car": "AUTOMOVIL",
    "motorcycle": "MOTOCICLETA",
    "bus": "AUTOBUS",
    "truck": "CAMION",
    "traffic light": "SEMAFORO",
    "stop sign": "SENAL DE ALTO",
}

# Colores BGR empleados para mostrar el nivel de riesgo.
RISK_COLORS = {
    "LEJOS": (0, 255, 0),
    "DISTANCIA MEDIA": (0, 255, 255),
    "CERCA": (0, 0, 255),
}


# FUNCIONES

def get_direction(
    object_x: int,
    frame_width: int
) -> str:
    """
    Determina la posición horizontal del objeto respecto
    al centro del campo visual.
    """

    center_x = frame_width // 2
    difference_x = object_x - center_x

    if difference_x < -CENTER_TOLERANCE:
        return "IZQUIERDA"

    if difference_x > CENTER_TOLERANCE:
        return "DERECHA"

    return "CENTRO"


def get_approximate_range(
    object_area: float,
    frame_area: float
) -> tuple[str, float]:
    """
    Clasifica el rango según el área visual ocupada.

    Esta clasificación es aproximada y no representa
    una distancia física.
    """

    area_ratio = object_area / frame_area

    if area_ratio < FAR_LIMIT:
        return "LEJOS", area_ratio

    if area_ratio < MEDIUM_LIMIT:
        return "DISTANCIA MEDIA", area_ratio

    return "CERCA", area_ratio


def calculate_priority(
    area_ratio: float,
    direction: str,
    confidence: float
) -> float:

    center_bonus = 2.0 if direction == "CENTRO" else 1.0

    return area_ratio * center_bonus * confidence


def draw_center_zone(
    frame,
    center_x: int,
    center_y: int
) -> None:
 
 #Zona frontal de referencia

    cv2.rectangle(
        frame,
        (
            center_x - CENTER_TOLERANCE,
            0
        ),
        (
            center_x + CENTER_TOLERANCE,
            frame.shape[0]
        ),
        (255, 255, 0),
        2
    )

    cv2.line(
        frame,
        (center_x - 20, center_y),
        (center_x + 20, center_y),
        (255, 255, 0),
        2
    )

    cv2.line(
        frame,
        (center_x, center_y - 20),
        (center_x, center_y + 20),
        (255, 255, 0),
        2
    )


# Carga el modelo deYOLO

print("Cargando modelo")

try:
    model = YOLO(MODEL_PATH)

except Exception as error:
    print("No fue posible cargar el modelo YOLO.")
    print(error)
    raise SystemExit(1)

print("Modelo cargado correctamente.")


# Apertura la cámara

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("No se pudo abrir la cámara RGB.")
    raise SystemExit(1)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# Configuración de ventana

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    960,
    720
)

fullscreen = False
detection_enabled = True
show_all_detections = True

previous_time = time.time()
fps = 0.0


print()
print("Detector iniciado.")
print("Controles:")
print("ESC = salir")
print("F = pantalla completa")
print("T = activar o desactivar detección")
print("A = mostrar todos o solo el objeto prioritario")


try:
    while True:

        success, frame = cap.read()

        if not success or frame is None:
            print("No fue posible leer un frame.")
            continue

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height

        center_x = frame_width // 2
        center_y = frame_height // 2

        output = frame.copy()

        draw_center_zone(
            output,
            center_x,
            center_y
        )

        detections = []
        priority_detection = None

        if detection_enabled:

            # Inferencia del modelo sobre el frame actual
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=INFERENCE_SIZE,
                verbose=False
            )

            result = results[0]

            # Revisión de las cajas detectadas.
            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                # Ignorar clases que no interesan al proyecto
                if class_name not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                box_width = max(1, x2 - x1)
                box_height = max(1, y2 - y1)

                object_area = box_width * box_height

                object_x = x1 + box_width // 2
                object_y = y1 + box_height // 2

                direction = get_direction(
                    object_x,
                    frame_width
                )

                approximate_range, area_ratio = (
                    get_approximate_range(
                        object_area,
                        frame_area
                    )
                )

                priority = calculate_priority(
                    area_ratio,
                    direction,
                    confidence
                )

                detection = {
                    "class_name": class_name,
                    "spanish_name": TARGET_CLASSES[class_name],
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                    "center": (object_x, object_y),
                    "direction": direction,
                    "range": approximate_range,
                    "area_ratio": area_ratio,
                    "priority": priority,
                }

                detections.append(detection)

                if (
                    priority_detection is None
                    or detection["priority"]
                    > priority_detection["priority"]
                ):
                    priority_detection = detection


        # Dibujo de detecciones
     
        for detection in detections:

            # Cuando esta opción está apagada solamente se
            # dibuja el objeto con prioridad más alta.
            if (
                not show_all_detections
                and detection is not priority_detection
            ):
                continue

            x1, y1, x2, y2 = detection["box"]
            object_x, object_y = detection["center"]

            range_name = detection["range"]
            color = RISK_COLORS[range_name]

            label = (
                f'{detection["spanish_name"]} '
                f'{detection["confidence"] * 100:.0f}%'
            )

            position_text = (
                f'{detection["direction"]} | '
                f'{range_name}'
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.circle(
                output,
                (object_x, object_y),
                5,
                color,
                -1
            )

            cv2.putText(
                output,
                label,
                (x1, max(22, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            cv2.putText(
                output,
                position_text,
                (x1, max(42, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )


        # Objeto principal detectado
    
        if priority_detection is not None:

            object_x, object_y = priority_detection["center"]

            priority_text = (
                f'PRIORIDAD: '
                f'{priority_detection["spanish_name"]} | '
                f'{priority_detection["direction"]} | '
                f'{priority_detection["range"]}'
            )

            priority_color = RISK_COLORS[
                priority_detection["range"]
            ]

            cv2.line(
                output,
                (center_x, center_y),
                (object_x, object_y),
                priority_color,
                3
            )

        else:
            if detection_enabled:
                priority_text = "SIN OBJETOS RELEVANTES"
            else:
                priority_text = "DETECCION DESACTIVADA"

            priority_color = (255, 255, 255)


       # Cálculo de FPS
 
        current_time = time.time()
        elapsed = current_time - previous_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            fps = 0.85 * fps + 0.15 * instant_fps

        previous_time = current_time


       
        cv2.rectangle(
            output,
            (0, 0),
            (frame_width, 82),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            priority_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            priority_color,
            2
        )

        cv2.putText(
            output,
            (
                f"Objetos: {len(detections)} | "
                f"FPS: {fps:.1f}"
            ),
            (15, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            "Q salir | F fullscreen | T deteccion | A vista",
            (15, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            WINDOW_NAME,
            output
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

        if key == ord("t"):
            detection_enabled = not detection_enabled

            print(
                "Detección:",
                "ACTIVADA"
                if detection_enabled
                else "DESACTIVADA"
            )

        if key == ord("a"):
            show_all_detections = not show_all_detections

            print(
                "Visualización:",
                "TODOS LOS OBJETOS"
                if show_all_detections
                else "SOLO OBJETO PRIORITARIO"
            )

        if key == ord("f"):
            fullscreen = not fullscreen

            if fullscreen:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )

            else:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )


finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Cámara cerrada.")
