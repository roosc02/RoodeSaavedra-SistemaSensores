from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from primesense import openni2

from motores import MotorConfig, MotorController, MotorPins
from navegar import NavigationConfig, NavigationDecision, decide_navigation
from yolo_tracking import (
    COCO_DATASET_CLASS_NAMES,
    TrajectoryDatasetRecorder,
    YoloSingleObjectTracker,
)


APP_TITLE = "Orbbec Astra+ | RGB + Profundidad"
PANEL_SIZE = (640, 480)
CAPTURE_DIR = Path("captures")
CALIBRATION_DIR = Path("calibration_data")
CONFIG_PATH = Path("camera_config.json")
COCO_TARGET_CLASSES = "person,car,dog,stop sign"

# Cambia esta ruta por la carpeta que contiene OpenNI2.dll y los drivers.
DEFAULT_OPENNI_PATH = (
    r"C:\Users\roode\Downloads"
    r"\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\samples\bin"
)

COLOR_RANGES = {
    "rojo": ((0, 100, 70), (10, 255, 255), (170, 100, 70), (179, 255, 255)),
    "verde": ((35, 70, 50), (85, 255, 255)),
    "azul": ((90, 80, 50), (135, 255, 255)),
    "amarillo": ((18, 90, 80), (35, 255, 255)),
    "naranja": ((8, 100, 80), (22, 255, 255)),
    "morado": ((130, 60, 40), (170, 255, 255)),
    "blanco": ((0, 0, 180), (179, 60, 255)),
    "negro": ((0, 0, 0), (179, 255, 65)),
}

SHAPES = {"circulo", "triangulo", "cuadrado", "rectangulo", "poligono", "cualquiera"}
DETECTION_MODES = ("ninguno", "color", "forma", "yolo")
COLOR_KEY_TARGETS = {
    "4": "rojo",
    "5": "verde",
    "6": "azul",
    "7": "amarillo",
    "8": "blanco",
    "9": "negro",
    "n": "naranja",
    "v": "morado",
}
SHAPE_KEY_TARGETS = {
    "b": "circulo",
    "y": "triangulo",
    "w": "cuadrado",
    "e": "rectangulo",
    "d": "cualquiera",
}


@dataclass
class DetectionSettings:
    mode: str = "ninguno"
    target: str = ""
    minimum_area_px: float = 800.0

    def cycle_mode(self) -> None:
        index = DETECTION_MODES.index(self.mode)
        self.mode = DETECTION_MODES[(index + 1) % len(DETECTION_MODES)]
        if self.mode == "color":
            self.target = next(iter(COLOR_RANGES))
        elif self.mode == "forma":
            self.target = "cualquiera"
        elif self.mode == "yolo":
            self.target = "cualquiera"
        else:
            self.target = ""

    def cycle_target(self) -> None:
        if self.mode == "color":
            choices = tuple(COLOR_RANGES)
        elif self.mode == "forma":
            choices = tuple(sorted(SHAPES))
        else:
            return
        try:
            index = choices.index(self.target)
        except ValueError:
            index = -1
        self.target = choices[(index + 1) % len(choices)]

    def set_color(self, color: str) -> None:
        if color not in COLOR_RANGES:
            return
        self.mode = "color"
        self.target = color

    def set_shape(self, shape: str) -> None:
        if shape not in SHAPES:
            return
        self.mode = "forma"
        self.target = shape

    def set_yolo(self, target: str = "cualquiera") -> None:
        self.mode = "yolo"
        self.target = target or "cualquiera"

    def disable(self) -> None:
        self.mode = "ninguno"
        self.target = ""

    def adjust_minimum_area(self, amount: float) -> None:
        self.minimum_area_px = float(
            np.clip(self.minimum_area_px + amount, 100.0, 100000.0)
        )


@dataclass
class EdgeSettings:
    selected_sensor: str = "rgb"
    rgb_low: int = 80
    rgb_high: int = 160
    depth_low: int = 40
    depth_high: int = 120

    def thresholds(self, sensor: str) -> tuple[int, int]:
        return (
            int(getattr(self, f"{sensor}_low")),
            int(getattr(self, f"{sensor}_high")),
        )

    def adjust(self, threshold: str, amount: int) -> None:
        field = f"{self.selected_sensor}_{threshold}"
        value = int(np.clip(getattr(self, field) + amount, 0, 255))
        setattr(self, field, value)
        low, high = self.thresholds(self.selected_sensor)
        if low > high:
            if threshold == "low":
                setattr(self, f"{self.selected_sensor}_high", low)
            else:
                setattr(self, f"{self.selected_sensor}_low", high)


@dataclass
class CameraConfig:
    openni_path: str = DEFAULT_OPENNI_PATH
    rgb_index: int = 0
    rgb_flip_horizontal: bool = True
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    rgb_dist_coeffs: Optional[list[float]] = None
    depth_scale_mm: float = 1.0
    depth_to_rgb_offset_x: int = 0
    depth_to_rgb_offset_y: int = 0
    depth_to_rgb_scale: float = 1.0
    depth_fx: Optional[float] = None
    depth_fy: Optional[float] = None
    depth_cx: Optional[float] = None
    depth_cy: Optional[float] = None
    depth_to_rgb_rotation: Optional[list[list[float]]] = None
    depth_to_rgb_translation_mm: Optional[list[float]] = None
    yolo_model: str = "yolo26n.pt"
    yolo_target_class: str = COCO_TARGET_CLASSES
    yolo_traffic_only: bool = False
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.50
    yolo_device: str = "cpu"
    yolo_tracker_config: str = "bytetrack_stop.yaml"
    yolo_dataset_dir: str = "evidencias_pruebas/coco_objetos"
    trajectory_samples: int = 10
    trajectory_interval_seconds: float = 0.30
    trajectory_auto_capture: bool = True
    trajectory_auto_stable_frames: int = 5
    motor_enabled: bool = False
    motor_dry_run: bool = True
    motor_max_speed: float = 0.30
    motor_left_forward_pin: Optional[int] = None
    motor_left_backward_pin: Optional[int] = None
    motor_right_forward_pin: Optional[int] = None
    motor_right_backward_pin: Optional[int] = None

    @classmethod
    def load(cls, path: Path) -> "CameraConfig":
        if not path.exists():
            config = cls()
            path.write_text(
                json.dumps(config.__dict__, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return config
        values = json.loads(path.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        config = cls(**{key: value for key, value in values.items() if key in allowed})
        if any(key not in values for key in allowed):
            config.save(path)
        return config

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.__dict__, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def has_intrinsics(self) -> bool:
        return bool(self.fx and self.fy and self.fx > 0 and self.fy > 0)

    @property
    def has_depth_to_rgb_calibration(self) -> bool:
        depth_intrinsics = (
            self.depth_fx,
            self.depth_fy,
            self.depth_cx,
            self.depth_cy,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )
        return bool(
            all(value is not None for value in depth_intrinsics)
            and self.depth_to_rgb_rotation is not None
            and self.depth_to_rgb_translation_mm is not None
        )

    def apply_alignment(self, alignment: "AlignmentSettings") -> None:
        self.depth_to_rgb_offset_x = int(alignment.offset_x)
        self.depth_to_rgb_offset_y = int(alignment.offset_y)
        self.depth_to_rgb_scale = float(alignment.scale)


@dataclass
class AlignmentSettings:
    offset_x: int = 0
    offset_y: int = 0
    scale: float = 1.0

    @classmethod
    def from_config(cls, config: CameraConfig) -> "AlignmentSettings":
        return cls(
            offset_x=int(config.depth_to_rgb_offset_x),
            offset_y=int(config.depth_to_rgb_offset_y),
            scale=float(config.depth_to_rgb_scale),
        )

    def adjust_offset(self, dx: int, dy: int) -> None:
        self.offset_x = int(np.clip(self.offset_x + dx, -400, 400))
        self.offset_y = int(np.clip(self.offset_y + dy, -400, 400))

    def adjust_scale(self, amount: float) -> None:
        self.scale = float(np.clip(self.scale + amount, 0.50, 1.80))

    def reset(self) -> None:
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0


@dataclass
class CaptureOptions:
    mark_for_calibration: bool = True

    def toggle_calibration_mark(self) -> None:
        self.mark_for_calibration = not self.mark_for_calibration


@dataclass
class Detection:
    contour: Optional[np.ndarray]
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area_px: float
    label: str
    depth_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    area_mm2: Optional[float] = None
    source: str = "clasico"
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    track_id: Optional[int] = None
    depth_source: Optional[str] = None
    depth_valid_pixels: int = 0


class TemporalKalman:
    """Filtro Kalman vectorizado: una estimación independiente por píxel."""

    def __init__(self, process_noise: float, measurement_noise: float):
        self.q = np.float32(process_noise)
        self.r = np.float32(measurement_noise)
        self.estimate: Optional[np.ndarray] = None
        self.error: Optional[np.ndarray] = None

    def update(self, image: np.ndarray) -> np.ndarray:
        measurement = image.astype(np.float32, copy=False)
        if self.estimate is None or self.estimate.shape != measurement.shape:
            self.estimate = measurement.copy()
            self.error = np.ones(measurement.shape, dtype=np.float32)
            return self.estimate

        self.error += self.q
        gain = self.error / (self.error + self.r)
        self.estimate += gain * (measurement - self.estimate)
        self.error *= 1.0 - gain
        return self.estimate


class FpsCounter:
    def __init__(self):
        self.frames = 0
        self.value = 0.0
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        self.frames += 1
        now = time.perf_counter()
        elapsed = now - self.last_time
        if elapsed >= 1.0:
            self.value = self.frames / elapsed
            self.frames = 0
            self.last_time = now
        return self.value


def ask_detection_settings() -> DetectionSettings:
    print("\n--- Selección de objetivo ---")
    print("1) Detectar por color")
    print("2) Detectar por forma")
    print("3) Solo visualizar sensores")
    option = input("Opción [3]: ").strip() or "3"

    if option == "1":
        choices = ", ".join(COLOR_RANGES)
        target = input(f"Color ({choices}) [rojo]: ").strip().lower() or "rojo"
        if target not in COLOR_RANGES:
            print("Color no reconocido; se usará rojo.")
            target = "rojo"
        return DetectionSettings(mode="color", target=target)

    if option == "2":
        choices = ", ".join(sorted(SHAPES))
        target = input(f"Forma ({choices}) [cualquiera]: ").strip().lower() or "cualquiera"
        if target not in SHAPES:
            print("Forma no reconocida; se detectará cualquiera.")
            target = "cualquiera"
        return DetectionSettings(mode="forma", target=target)

    return DetectionSettings()


def normalize_u8(image: np.ndarray, valid_only: bool = False) -> np.ndarray:
    if valid_only:
        valid = image[image > 0]
        if valid.size == 0:
            return np.zeros(image.shape, dtype=np.uint8)
        low, high = np.percentile(valid, (2, 98))
        if high <= low:
            return np.zeros(image.shape, dtype=np.uint8)
        normalized = np.clip((image - low) * 255.0 / (high - low), 0, 255)
        normalized[image <= 0] = 0
        return normalized.astype(np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edges_only(
    source_gray: np.ndarray,
    low: int,
    high: int,
    color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve únicamente los bordes sobre fondo negro."""
    blurred = cv2.GaussianBlur(source_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    result = np.zeros((*source_gray.shape, 3), dtype=np.uint8)
    result[edges > 0] = color
    return result, edges


def canny_edges_overlay(
    source_bgr: np.ndarray,
    low: int,
    high: int,
    color: tuple[int, int, int],
    dim_factor: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    """Dibuja Canny encima de la imagen RGB original."""
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    result = np.clip(source_bgr.astype(np.float32) * dim_factor, 0, 255).astype(
        np.uint8
    )
    result[edges > 0] = color
    return result, edges


def shape_name(contour: np.ndarray) -> str:
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    vertices = len(polygon)
    if vertices == 3:
        return "triangulo"
    if vertices == 4:
        x, y, width, height = cv2.boundingRect(polygon)
        ratio = width / max(height, 1)
        return "cuadrado" if 0.88 <= ratio <= 1.12 else "rectangulo"
    if vertices >= 8:
        area = cv2.contourArea(contour)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
        if circularity >= 0.72:
            return "circulo"
    return "poligono"


def find_target(
    rgb: np.ndarray,
    settings: DetectionSettings,
    previous_center: Optional[tuple[int, int]] = None,
) -> Optional[Detection]:
    # Detector clasico aislado. Se conserva para pruebas historicas, pero el
    # bucle principal trabaja exclusivamente con YOLO y nunca llama esta funcion.
    if settings.mode == "ninguno":
        return None

    if settings.mode == "color":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        limits = COLOR_RANGES[settings.target]
        mask = cv2.inRange(hsv, np.array(limits[0]), np.array(limits[1]))
        if len(limits) == 4:
            mask |= cv2.inRange(hsv, np.array(limits[2]), np.array(limits[3]))
        label = settings.target
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.Canny(gray, 60, 160)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        label = settings.target

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray, str]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings.minimum_area_px:
            continue
        detected_shape = shape_name(contour)
        if settings.mode == "forma" and settings.target not in ("cualquiera", detected_shape):
            continue
        candidates.append((area, contour, detected_shape if settings.mode == "forma" else label))

    if not candidates:
        return None

    if previous_center is None:
        area, contour, detected_label = max(candidates, key=lambda item: item[0])
    else:
        def tracking_score(item: tuple[float, np.ndarray, str]) -> float:
            area, contour, _ = item
            x, y, width, height = cv2.boundingRect(contour)
            center_x, center_y = x + width / 2.0, y + height / 2.0
            distance = np.hypot(
                center_x - previous_center[0],
                center_y - previous_center[1],
            )
            # Favorece continuidad espacial y, en empate, contornos grandes.
            return float(distance - min(area, 20000.0) * 0.002)

        area, contour, detected_label = min(candidates, key=tracking_score)
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
    else:
        center = (x + width // 2, y + height // 2)
    return Detection(contour, (x, y, width, height), center, area, detected_label)


def find_yolo_target(
    rgb: np.ndarray,
    tracker: YoloSingleObjectTracker,
) -> Optional[Detection]:
    result = tracker.update(rgb)
    if result is None:
        return None
    x, y, width, height = result.bbox
    contour = np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.int32,
    )
    return Detection(
        contour=contour,
        bbox=result.bbox,
        center=result.center,
        area_px=result.area_px,
        label=result.label,
        source="yolo",
        confidence=result.confidence,
        class_id=result.class_id,
        track_id=result.track_id,
    )


def yolo_result_to_detection(result) -> Detection:
    x, y, width, height = result.bbox
    contour = np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.int32,
    )
    return Detection(
        contour=contour,
        bbox=result.bbox,
        center=result.center,
        area_px=result.area_px,
        label=result.label,
        source="yolo",
        confidence=result.confidence,
        class_id=result.class_id,
        track_id=result.track_id,
    )


def find_yolo_detections(
    rgb: np.ndarray,
    tracker: YoloSingleObjectTracker,
) -> list[Detection]:
    return [
        yolo_result_to_detection(result)
        for result in tracker.update_all(rgb)
    ]


def select_primary_detection(
    detections: list[Detection],
    active_track_id: Optional[int],
    previous_center: Optional[tuple[int, int]],
) -> Optional[Detection]:
    if not detections:
        return None
    if active_track_id is not None:
        same_track = [
            detection for detection in detections if detection.track_id == active_track_id
        ]
        if same_track:
            return max(same_track, key=lambda item: item.confidence or 0.0)
    if previous_center is not None:
        return min(
            detections,
            key=lambda item: np.hypot(
                item.center[0] - previous_center[0],
                item.center[1] - previous_center[1],
            )
            - 40.0 * float(item.confidence or 0.0),
        )
    return max(
        detections,
        key=lambda item: float(item.confidence or 0.0) * np.sqrt(item.area_px),
    )


def mirror_detection_horizontal(detection: Detection, image_width: int) -> None:
    """Lleva una deteccion del RGB real al RGB espejado usado por la interfaz."""
    x, y, width, height = detection.bbox
    mirrored_x = max(0, image_width - x - width)
    detection.bbox = (mirrored_x, y, width, height)
    detection.center = (image_width - 1 - detection.center[0], detection.center[1])
    detection.contour = np.array(
        [
            [[mirrored_x, y]],
            [[mirrored_x + width, y]],
            [[mirrored_x + width, y + height]],
            [[mirrored_x, y + height]],
        ],
        dtype=np.int32,
    )


def map_bbox(
    bbox: tuple[int, int, int, int],
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    return (
        int(x * target_width / source_width),
        int(y * target_height / source_height),
        max(1, int(width * target_width / source_width)),
        max(1, int(height * target_height / source_height)),
    )


def align_depth_to_rgb_manual(
    depth: np.ndarray,
    rgb_shape: tuple[int, ...],
    alignment: AlignmentSettings,
) -> np.ndarray:
    """Alinea profundidad al tamaño/plano RGB con escala y desplazamiento manual."""
    rgb_height, rgb_width = rgb_shape[:2]
    resized_width = max(1, int(round(rgb_width * alignment.scale)))
    resized_height = max(1, int(round(rgb_height * alignment.scale)))
    resized = cv2.resize(
        depth,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    aligned = np.zeros((rgb_height, rgb_width), dtype=depth.dtype)

    dst_x0 = max(0, alignment.offset_x)
    dst_y0 = max(0, alignment.offset_y)
    src_x0 = max(0, -alignment.offset_x)
    src_y0 = max(0, -alignment.offset_y)
    width = min(rgb_width - dst_x0, resized_width - src_x0)
    height = min(rgb_height - dst_y0, resized_height - src_y0)
    if width <= 0 or height <= 0:
        return aligned

    aligned[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = resized[
        src_y0 : src_y0 + height,
        src_x0 : src_x0 + width,
    ]
    return aligned


def align_depth_to_rgb_calibrated(
    depth: np.ndarray,
    rgb_shape: tuple[int, ...],
    config: CameraConfig,
) -> np.ndarray:
    """Reproyecta profundidad al plano RGB usando intrínsecos y extrínsecos."""
    if not config.has_depth_to_rgb_calibration:
        raise ValueError("Faltan matrices para calibracion profundidad -> RGB.")

    rgb_height, rgb_width = rgb_shape[:2]
    valid_y, valid_x = np.nonzero(depth)
    if valid_x.size == 0:
        return np.zeros((rgb_height, rgb_width), dtype=depth.dtype)

    z_depth = depth[valid_y, valid_x].astype(np.float32) * float(config.depth_scale_mm)
    x_depth = (valid_x.astype(np.float32) - float(config.depth_cx)) * z_depth / float(
        config.depth_fx
    )
    y_depth = (valid_y.astype(np.float32) - float(config.depth_cy)) * z_depth / float(
        config.depth_fy
    )
    points_depth = np.vstack((x_depth, y_depth, z_depth))

    rotation = np.asarray(config.depth_to_rgb_rotation, dtype=np.float32)
    translation = np.asarray(config.depth_to_rgb_translation_mm, dtype=np.float32).reshape(
        3,
        1,
    )
    points_rgb = rotation @ points_depth + translation
    z_rgb = points_rgb[2]
    in_front = z_rgb > 0
    if not np.any(in_front):
        return np.zeros((rgb_height, rgb_width), dtype=depth.dtype)

    points_rgb = points_rgb[:, in_front]
    z_rgb = z_rgb[in_front]
    source_depth = depth[valid_y, valid_x][in_front]
    u_rgb = np.rint(float(config.fx) * points_rgb[0] / z_rgb + float(config.cx)).astype(
        np.int32
    )
    v_rgb = np.rint(float(config.fy) * points_rgb[1] / z_rgb + float(config.cy)).astype(
        np.int32
    )
    inside = (u_rgb >= 0) & (u_rgb < rgb_width) & (v_rgb >= 0) & (v_rgb < rgb_height)
    u_rgb = u_rgb[inside]
    v_rgb = v_rgb[inside]
    z_rgb_raw = np.rint(z_rgb[inside] / float(config.depth_scale_mm)).astype(depth.dtype)
    source_depth = source_depth[inside]

    aligned = np.zeros((rgb_height, rgb_width), dtype=depth.dtype)
    z_buffer = np.full((rgb_height, rgb_width), np.inf, dtype=np.float32)
    z_metric = z_rgb[inside]
    for u, v, z_value, raw_value in zip(u_rgb, v_rgb, z_metric, z_rgb_raw):
        if z_value < z_buffer[v, u]:
            z_buffer[v, u] = z_value
            aligned[v, u] = raw_value
    return aligned


def align_depth_to_rgb(
    depth: np.ndarray,
    rgb_shape: tuple[int, ...],
    config: CameraConfig,
    alignment: AlignmentSettings,
) -> tuple[np.ndarray, str]:
    if config.has_depth_to_rgb_calibration:
        return align_depth_to_rgb_calibrated(depth, rgb_shape, config), "calibrada"
    return align_depth_to_rgb_manual(depth, rgb_shape, alignment), "manual"


def depth_rgb_overlay(rgb: np.ndarray, aligned_depth: np.ndarray) -> np.ndarray:
    depth_u8 = normalize_u8(aligned_depth, valid_only=True)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    mask = aligned_depth > 0
    overlay = rgb.copy()
    if np.any(mask):
        blended = (
            rgb[mask].astype(np.float32) * 0.45
            + depth_color[mask].astype(np.float32) * 0.55
        )
        overlay[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    return overlay


def median_depth_in_bbox(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale_mm: float,
) -> tuple[Optional[float], Optional[str], int]:
    x, y, width, height = bbox
    if width <= 0 or height <= 0 or depth.size == 0:
        return None, None, 0

    def read_roi(
        left: float,
        top: float,
        roi_width: float,
        roi_height: float,
    ) -> tuple[np.ndarray, int]:
        x1 = int(np.clip(round(left), 0, depth.shape[1]))
        y1 = int(np.clip(round(top), 0, depth.shape[0]))
        x2 = int(np.clip(round(left + roi_width), 0, depth.shape[1]))
        y2 = int(np.clip(round(top + roi_height), 0, depth.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return np.empty(0, dtype=depth.dtype), 0
        roi = depth[y1:y2, x1:x2]
        valid = roi[roi > 0]
        return valid, int(valid.size)

    min_valid_pixels = max(8, min(80, int(width * height * 0.002)))
    inset_x = max(1, width // 5)
    inset_y = max(1, height // 5)
    candidates = [
        (
            "centro bbox",
            x + inset_x,
            y + inset_y,
            width - 2 * inset_x,
            height - 2 * inset_y,
        ),
        ("bbox completo", x, y, width, height),
        (
            "bbox expandido",
            x - width * 0.15,
            y - height * 0.15,
            width * 1.30,
            height * 1.30,
        ),
    ]
    total_valid = 0
    for source, left, top, roi_width, roi_height in candidates:
        valid, count = read_roi(left, top, roi_width, roi_height)
        total_valid += count
        if count >= min_valid_pixels:
            return float(np.median(valid) * scale_mm), source, count

    center_x = x + width / 2.0
    center_y = y + height / 2.0
    for radius in (16, 32, 64, 96):
        valid, count = read_roi(
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2,
        )
        if count >= max(5, min_valid_pixels // 2):
            return float(np.median(valid) * scale_mm), f"cercana {radius}px", count

    return None, None, total_valid


def measure_detection(
    detection: Detection,
    rgb_shape: tuple[int, ...],
    depth: np.ndarray,
    config: CameraConfig,
) -> None:
    depth_mm, depth_source, depth_valid_pixels = median_depth_in_bbox(
        depth,
        detection.bbox,
        config.depth_scale_mm,
    )
    detection.depth_mm = depth_mm
    detection.depth_source = depth_source
    detection.depth_valid_pixels = depth_valid_pixels
    if detection.depth_mm is None or not config.has_intrinsics:
        return

    _, _, width_px, height_px = detection.bbox
    z = detection.depth_mm
    detection.width_mm = width_px * z / float(config.fx)
    detection.height_mm = height_px * z / float(config.fy)
    pixel_size_x = z / float(config.fx)
    pixel_size_y = z / float(config.fy)
    detection.area_mm2 = detection.area_px * pixel_size_x * pixel_size_y


def draw_detection(
    panel: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = bbox
    cv2.rectangle(panel, (x, y), (x + width, y + height), color, 2)
    cv2.drawMarker(
        panel,
        (x + width // 2, y + height // 2),
        color,
        cv2.MARKER_CROSS,
        18,
        2,
    )
    cv2.putText(
        panel,
        label,
        (x, max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def detection_display_label(detection: Detection) -> str:
    parts = [detection.label, f"{detection.area_px:.0f}px2"]
    if detection.confidence is not None:
        parts.append(f"{detection.confidence:.0%}")
    if detection.track_id is not None:
        parts.append(f"ID {detection.track_id}")
    if detection.depth_mm is not None:
        parts.append(f"{detection.depth_mm:.0f}mm")
    if detection.area_mm2 is not None:
        parts.append(f"{detection.area_mm2 / 100.0:.1f}cm2")
    return " | ".join(parts)


def detection_color(label: str) -> tuple[int, int, int]:
    colors = {
        "persona": (255, 180, 60),
        "carro": (80, 220, 255),
        "perro": (180, 120, 255),
        "senalamiento_trafico": (0, 220, 255),
    }
    return colors.get(label, (0, 220, 255))


def add_panel_title(panel: np.ndarray, title: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (16, 16, 16), -1)
    cv2.putText(
        panel,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def telemetry_panel(
    lines_by_sensor: dict[str, list[str]],
    edge_settings: EdgeSettings,
    detection_settings: DetectionSettings,
    capture_options: CaptureOptions,
    alignment_settings: AlignmentSettings,
    alignment_mode: str,
    trajectory_recorder: TrajectoryDatasetRecorder,
    yolo_inference_ms: Optional[float],
    navigation_decision: Optional[NavigationDecision],
    motor_status: str,
) -> np.ndarray:
    width, height = PANEL_SIZE
    panel = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(
        panel,
        "OPCIONES Y DATOS",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    selected = edge_settings.selected_sensor
    selected_low, selected_high = edge_settings.thresholds(selected)
    cv2.putText(
        panel,
        f"Ajuste Canny: {selected.upper()} | bajo {selected_low} | alto {selected_high}",
        (18, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (90, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "1 RGB / 2 PROF | A/Z bajo +/- | S/X alto +/-",
        (18, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "YOLO26 COCO: persona | carro | perro | senalamiento",
        (18, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (120, 230, 170),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "RGB detecta bbox | profundidad mide sobre bbox alineado",
        (18, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Arboles fuera: COCO no trae clase tree/arbol",
        (18, 137),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"Ensamble RGB+PROF: {alignment_mode.upper()} | "
            f"X {alignment_settings.offset_x:+d} | Y {alignment_settings.offset_y:+d} | "
            f"escala {alignment_settings.scale:.2f}"
        ),
        (18, 157),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 210, 90),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            "I/K sube-baja | J/L izq-der | U/O escala | R reinicia | "
            f"F [{'X' if capture_options.mark_for_calibration else ' '}] calibracion"
        ),
        (18, 176),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    yolo_text = f"YOLO26 COCO: AUTO | {trajectory_recorder.status}"
    if yolo_inference_ms is not None:
        yolo_text += f" | {yolo_inference_ms:.1f} ms"
    cv2.putText(
        panel,
        yolo_text,
        (18, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (120, 230, 170),
        1,
        cv2.LINE_AA,
    )
    nav_text = "NAVEGACION: sin datos"
    if navigation_decision is not None:
        nav_text = f"NAVEGACION: {navigation_decision.text[:88]}"
    cv2.putText(
        panel,
        nav_text,
        (18, 214),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"MOTORES: {motor_status}",
        (18, 233),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 210, 90),
        1,
        cv2.LINE_AA,
    )
    sensor_names = ["profundidad", "rgb"]
    column_width = width // 2
    for column, sensor_name in enumerate(sensor_names):
        x = 14 + column * column_width
        y = 244
        cv2.putText(
            panel,
            sensor_name.upper(),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (90, 220, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25
        for line in lines_by_sensor[sensor_name]:
            cv2.putText(
                panel,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            y += 18

    y = 385
    cv2.line(panel, (18, y - 20), (width - 18, y - 20), (65, 65, 65), 1)
    cv2.putText(
        panel,
        "OBJETIVO",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (90, 220, 255),
        2,
        cv2.LINE_AA,
    )
    y += 24
    for line in lines_by_sensor["objetivo"]:
        cv2.putText(
            panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += 18

    cv2.putText(
        panel,
        "Q/Esc salir | C datos crudos | P captura | YOLO guarda 10 muestras automaticas",
        (18, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return panel


def stats(image: np.ndarray, ignore_zero: bool = False) -> tuple[float, float, float]:
    values = image[image > 0] if ignore_zero else image.reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.min()), float(values.max()), float(values.mean())


def save_raw_capture(
    rgb: np.ndarray,
    depth: np.ndarray,
    aligned_depth: np.ndarray,
    detection: Optional[Detection],
    config: CameraConfig,
    alignment: AlignmentSettings,
    alignment_mode: str,
    capture_options: CaptureOptions,
) -> Path:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = CALIBRATION_DIR / f"orbbec_raw_{stamp}.npz"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "rgb_order": "BGR",
        "depth_unit": "raw OpenNI units",
        "depth_scale_mm": config.depth_scale_mm,
        "depth_to_rgb_mode": alignment_mode,
        "depth_to_rgb_offset_x": alignment.offset_x,
        "depth_to_rgb_offset_y": alignment.offset_y,
        "depth_to_rgb_scale": alignment.scale,
        "capture_marked_for_calibration": capture_options.mark_for_calibration,
        "detection_bbox_rgb": detection.bbox if detection else None,
        "detection_center_rgb": detection.center if detection else None,
        "detection_label": detection.label if detection else None,
        "detection_source": detection.source if detection else None,
        "detection_confidence": detection.confidence if detection else None,
        "detection_class_id": detection.class_id if detection else None,
        "detection_track_id": detection.track_id if detection else None,
        "detection_area_px": detection.area_px if detection else None,
        "detection_depth_mm": detection.depth_mm if detection else None,
        "detection_width_mm": detection.width_mm if detection else None,
        "detection_height_mm": detection.height_mm if detection else None,
        "detection_area_mm2": detection.area_mm2 if detection else None,
        "detection_area_cm2": detection.area_mm2 / 100.0
        if detection and detection.area_mm2 is not None
        else None,
    }
    np.savez_compressed(
        destination,
        rgb_bgr=rgb,
        depth_raw=depth,
        depth_aligned_to_rgb=aligned_depth,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    cv2.imwrite(str(CALIBRATION_DIR / f"rgb_{stamp}.png"), rgb)
    cv2.imwrite(str(CALIBRATION_DIR / f"depth_{stamp}.png"), depth)
    cv2.imwrite(str(CALIBRATION_DIR / f"depth_aligned_to_rgb_{stamp}.png"), aligned_depth)
    return destination


def make_telemetry(
    rgb: np.ndarray,
    depth: np.ndarray,
    edge_masks: dict[str, np.ndarray],
    fps: dict[str, float],
    settings: DetectionSettings,
    detection: Optional[Detection],
    config: CameraConfig,
) -> dict[str, list[str]]:
    depth_min, depth_max, depth_mean = stats(depth, ignore_zero=True)
    bgr_mean = rgb.mean(axis=(0, 1))
    total_pixels = {
        "rgb": rgb.shape[0] * rgb.shape[1],
        "depth": depth.size,
    }
    active_pixels = {
        "rgb": int(np.count_nonzero(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))),
        "depth": int(np.count_nonzero(depth)),
    }

    def pixel_line(sensor: str) -> str:
        active = active_pixels[sensor]
        total = total_pixels[sensor]
        percentage = 100.0 * active / max(total, 1)
        return f"Activos: {active}/{total} ({percentage:.1f}%)"

    def edge_line(sensor: str) -> str:
        edge_count = int(np.count_nonzero(edge_masks[sensor]))
        percentage = 100.0 * edge_count / max(total_pixels[sensor], 1)
        return f"Bordes: {edge_count} ({percentage:.1f}%)"

    objective = [f"Modo: {settings.mode} {settings.target}".strip()]
    if detection:
        objective.extend(
            [
                f"Detectado: {detection.label} ({detection.source})",
                f"Centro RGB: {detection.center}",
                f"Area: {detection.area_px:.0f} px2",
                f"Distancia: {detection.depth_mm:.1f} mm"
                if detection.depth_mm is not None
                else "Distancia: sin datos",
            ]
        )
        if detection.confidence is not None:
            objective.insert(
                2,
                f"Confianza: {detection.confidence:.1%} | Track ID: {detection.track_id}",
            )
        if detection.area_mm2 is not None:
            objective.extend(
                [
                    f"Ancho x alto: {detection.width_mm:.1f} x {detection.height_mm:.1f} mm",
                    f"Area aproximada: {detection.area_mm2:.1f} mm2",
                    f"Area aproximada: {detection.area_mm2 / 100.0:.2f} cm2",
                ]
            )
        else:
            objective.append("Medida real: requiere fx/fy calibrados")
    else:
        objective.append("Detectado: no")

    return {
        "objetivo": objective,
        "profundidad": [
            f"Resolucion: {depth.shape[1]}x{depth.shape[0]}",
            f"FPS: {fps['depth']:.1f}",
            pixel_line("depth"),
            edge_line("depth"),
            f"Min / max: {depth_min:.0f} / {depth_max:.0f}",
            f"Media valida: {depth_mean:.1f} unidades",
        ],
        "rgb": [
            f"Resolucion: {rgb.shape[1]}x{rgb.shape[0]}",
            f"FPS: {fps['rgb']:.1f}",
            pixel_line("rgb"),
            edge_line("rgb"),
            f"B / G / R media: {bgr_mean[0]:.1f} / {bgr_mean[1]:.1f} / {bgr_mean[2]:.1f}",
        ],
    }


def trajectory_sample_metadata(
    rgb: np.ndarray,
    depth: np.ndarray,
    aligned_depth: np.ndarray,
    config: CameraConfig,
    alignment: AlignmentSettings,
    alignment_mode: str,
    timestamps_ns: dict[str, int],
    fps: dict[str, float],
    model_name: str,
    rgb_camera_properties: dict[str, float],
) -> dict[str, object]:
    depth_min, depth_max, depth_mean = stats(depth, ignore_zero=True)
    aligned_min, aligned_max, aligned_mean = stats(aligned_depth, ignore_zero=True)
    bgr_mean = rgb.mean(axis=(0, 1))
    return {
        "timestamp": datetime.now().isoformat(),
        "sensor_read_timestamps_ns": timestamps_ns,
        "sensor_read_span_ms": (
            max(timestamps_ns.values()) - min(timestamps_ns.values())
        )
        / 1_000_000.0,
        "rgb_resolution": [rgb.shape[1], rgb.shape[0]],
        "depth_resolution": [depth.shape[1], depth.shape[0]],
        "fps": fps,
        "rgb_dtype": str(rgb.dtype),
        "depth_dtype": str(depth.dtype),
        "rgb_active_pixels": int(
            np.count_nonzero(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))
        ),
        "rgb_mean_bgr": [float(value) for value in bgr_mean],
        "rgb_camera_properties": rgb_camera_properties,
        "depth_valid_pixels": int(np.count_nonzero(depth)),
        "depth_min_raw": depth_min,
        "depth_max_raw": depth_max,
        "depth_mean_raw": depth_mean,
        "depth_aligned_valid_pixels": int(np.count_nonzero(aligned_depth)),
        "depth_aligned_min_raw": aligned_min,
        "depth_aligned_max_raw": aligned_max,
        "depth_aligned_mean_raw": aligned_mean,
        "depth_scale_mm": config.depth_scale_mm,
        "rgb_intrinsics": {
            "fx": config.fx,
            "fy": config.fy,
            "cx": config.cx,
            "cy": config.cy,
            "distortion": config.rgb_dist_coeffs,
        },
        "depth_to_rgb_mode": alignment_mode,
        "depth_to_rgb_offset_x": alignment.offset_x,
        "depth_to_rgb_offset_y": alignment.offset_y,
        "depth_to_rgb_scale": alignment.scale,
        "yolo_model": model_name,
    }


def main() -> None:
    config = CameraConfig.load(CONFIG_PATH)
    config_changed = False
    if config.yolo_traffic_only:
        config.yolo_traffic_only = False
        config_changed = True
    model_name = Path(config.yolo_model).name.lower()
    if not model_name.startswith("yolo26"):
        config.yolo_model = "yolo26n.pt"
        config_changed = True
    if config.yolo_target_class.strip().lower() in {
        "senalamiento_trafico",
        "señalamiento_trafico",
        "alto",
        "stop",
        "stop sign",
    }:
        config.yolo_target_class = COCO_TARGET_CLASSES
        config_changed = True
    desired_targets = {
        item.strip().lower()
        for item in COCO_TARGET_CLASSES.split(",")
        if item.strip()
    }
    configured_targets = {
        item.strip().lower()
        for item in str(config.yolo_target_class).replace(";", ",").split(",")
        if item.strip()
    }
    if configured_targets != desired_targets:
        config.yolo_target_class = COCO_TARGET_CLASSES
        config_changed = True
    if config.yolo_dataset_dir in {
        "yolo_dataset",
        "evidencias_pruebas/senalamiento_trafico",
    }:
        config.yolo_dataset_dir = "evidencias_pruebas/coco_objetos"
        config_changed = True
    if config.yolo_confidence < 0.15:
        config.yolo_confidence = 0.25
        config_changed = True
    if config_changed:
        config.save(CONFIG_PATH)
        print("Configuracion YOLO actualizada a YOLO26 COCO.")
    # Las opciones se modifican dentro de la ventana principal. No se abre un
    # formulario separado porque cerrar/aplicar ese formulario puede destruir
    # las ventanas administradas por OpenCV en algunas configuraciones.
    settings = DetectionSettings(mode="yolo", target="coco")
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    rgb_camera = None
    depth_stream = None
    openni_initialized = False
    rgb_filter = TemporalKalman(process_noise=1.5, measurement_noise=30.0)
    depth_filter = TemporalKalman(process_noise=2.0, measurement_noise=80.0)
    edge_settings = EdgeSettings()
    alignment_settings = AlignmentSettings.from_config(config)
    capture_options = CaptureOptions()
    navigation_config = NavigationConfig(depth_scale_mm=config.depth_scale_mm)
    motor_controller = MotorController(
        MotorConfig(
            enabled=config.motor_enabled,
            dry_run=config.motor_dry_run,
            max_speed=config.motor_max_speed,
            pins=MotorPins(
                config.motor_left_forward_pin,
                config.motor_left_backward_pin,
                config.motor_right_forward_pin,
                config.motor_right_backward_pin,
            ),
        )
    )
    fps_counters = {name: FpsCounter() for name in ("rgb", "depth")}
    screenshot_count = 0
    tracked_center: Optional[tuple[int, int]] = None
    lost_frames = 0
    rgb_read_failures = 0
    yolo_tracker: Optional[YoloSingleObjectTracker] = None
    yolo_load_error: Optional[str] = None
    trajectory_recorder = TrajectoryDatasetRecorder(
        Path(config.yolo_dataset_dir),
        samples_per_trajectory=config.trajectory_samples,
        interval_seconds=config.trajectory_interval_seconds,
        class_names=COCO_DATASET_CLASS_NAMES,
    )
    auto_candidate_key: Optional[str] = None
    auto_stable_frames = 0
    last_auto_recorded_key: Optional[str] = None
    recording_missing_frames = 0
    navigation_decision: Optional[NavigationDecision] = None

    try:
        rgb_camera = cv2.VideoCapture(config.rgb_index, cv2.CAP_MSMF)
        if not rgb_camera.isOpened():
            raise RuntimeError(f"No se pudo abrir RGB en el indice {config.rgb_index}.")
        rgb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, PANEL_SIZE[0])
        rgb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_SIZE[1])
        rgb_camera.set(cv2.CAP_PROP_FPS, 30)

        openni2.initialize(config.openni_path)
        openni_initialized = True
        device = openni2.Device.open_any()
        depth_stream = device.create_depth_stream()
        depth_stream.start()

        print("\nAplicación iniciada. Usa Q o Esc para salir.")
        print(f"Estado de motores: {motor_controller.status}")
        while True:
            rgb_ok, rgb = rgb_camera.read()
            rgb_timestamp_ns = time.time_ns()
            if not rgb_ok or rgb is None:
                rgb_read_failures += 1
                if rgb_read_failures >= 30:
                    raise RuntimeError(
                        "No se pudo recibir RGB durante 30 intentos consecutivos."
                    )
                time.sleep(0.03)
                continue
            rgb_read_failures = 0
            if config.rgb_flip_horizontal:
                rgb = cv2.flip(rgb, 1)

            depth_frame = depth_stream.read_frame()
            depth_timestamp_ns = time.time_ns()
            depth_raw = np.frombuffer(
                depth_frame.get_buffer_as_uint16(), dtype=np.uint16
            ).reshape(depth_frame.height, depth_frame.width).copy()

            fps = {name: counter.tick() for name, counter in fps_counters.items()}
            rgb_smoothed = rgb_filter.update(rgb).astype(np.uint8)
            depth_smoothed = depth_filter.update(depth_raw)
            aligned_depth_raw, alignment_mode = align_depth_to_rgb(
                depth_raw,
                rgb.shape,
                config,
                alignment_settings,
            )
            aligned_depth_smoothed, _ = align_depth_to_rgb(
                depth_smoothed.astype(depth_raw.dtype),
                rgb.shape,
                config,
                alignment_settings,
            )

            if yolo_tracker is None and yolo_load_error is None:
                try:
                    print(f"Cargando modelo YOLO: {config.yolo_model} ...")
                    yolo_tracker = YoloSingleObjectTracker(
                        model_path=config.yolo_model,
                        target_class=config.yolo_target_class,
                        confidence=config.yolo_confidence,
                        iou=config.yolo_iou,
                        device=config.yolo_device,
                        traffic_only=config.yolo_traffic_only,
                        tracker_config=config.yolo_tracker_config,
                    )
                    print(
                        "YOLO26 COCO listo. Clases activas: "
                        f"{config.yolo_target_class} | modelo: {config.yolo_model}"
                    )
                except Exception as error:
                    yolo_load_error = str(error)
                    print(f"No se pudo iniciar YOLO: {yolo_load_error}")
            yolo_rgb = cv2.flip(rgb, 1) if config.rgb_flip_horizontal else rgb
            detections = (
                find_yolo_detections(yolo_rgb, yolo_tracker)
                if yolo_tracker is not None
                else []
            )
            for item in detections:
                if config.rgb_flip_horizontal:
                    mirror_detection_horizontal(item, rgb.shape[1])
                measure_detection(item, rgb.shape, aligned_depth_raw, config)

            detection = select_primary_detection(
                detections,
                trajectory_recorder.track_id if trajectory_recorder.active else None,
                tracked_center,
            )
            if detection is not None:
                tracked_center = detection.center
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > 15:
                    tracked_center = None

            navigation_decision = decide_navigation(
                detections,
                aligned_depth_raw,
                navigation_config,
            )
            motor_controller.apply_decision(navigation_decision)

            if config.trajectory_auto_capture:
                if detection is not None and detection.source == "yolo":
                    current_key = (
                        f"track:{detection.track_id}"
                        if detection.track_id is not None
                        else f"roi:{detection.center[0] // 40}:{detection.center[1] // 40}"
                    )
                    if current_key == auto_candidate_key:
                        auto_stable_frames += 1
                    else:
                        auto_candidate_key = current_key
                        auto_stable_frames = 1

                    same_recording_track = (
                        trajectory_recorder.track_id is None
                        or detection.track_id == trajectory_recorder.track_id
                    )
                    if trajectory_recorder.active and not same_recording_track:
                        recording_missing_frames += 1
                    else:
                        recording_missing_frames = 0

                    if (
                        not trajectory_recorder.active
                        and auto_stable_frames
                        >= max(1, int(config.trajectory_auto_stable_frames))
                        and current_key != last_auto_recorded_key
                    ):
                        directory = trajectory_recorder.start(detection.label)
                        if yolo_tracker is not None:
                            yolo_tracker.selected_track_id = detection.track_id
                        recording_missing_frames = 0
                        print(
                            f"{detection.label} detectado: "
                            f"trayectoria automatica iniciada: {directory}"
                        )
                else:
                    auto_candidate_key = None
                    auto_stable_frames = 0
                    if trajectory_recorder.active:
                        recording_missing_frames += 1

                if trajectory_recorder.active and recording_missing_frames > 30:
                    trajectory_recorder.cancel()
                    recording_missing_frames = 0
                    print("Trayectoria automatica cancelada: se perdio el objeto.")

            if (
                detection is not None
                and detection.source == "yolo"
                and trajectory_recorder.ready_for(detection.track_id)
            ):
                metadata_path = trajectory_recorder.save_sample(
                    rgb=rgb,
                    depth_raw=depth_raw,
                    depth_aligned=aligned_depth_raw,
                    detection=detection,
                    metadata=trajectory_sample_metadata(
                        rgb,
                        depth_raw,
                        aligned_depth_raw,
                        config,
                        alignment_settings,
                        alignment_mode,
                        {
                            "rgb": rgb_timestamp_ns,
                            "depth": depth_timestamp_ns,
                        },
                        fps,
                        config.yolo_model,
                        {
                            "brightness": float(
                                rgb_camera.get(cv2.CAP_PROP_BRIGHTNESS)
                            ),
                            "contrast": float(rgb_camera.get(cv2.CAP_PROP_CONTRAST)),
                            "saturation": float(
                                rgb_camera.get(cv2.CAP_PROP_SATURATION)
                            ),
                            "gain": float(rgb_camera.get(cv2.CAP_PROP_GAIN)),
                            "exposure": float(rgb_camera.get(cv2.CAP_PROP_EXPOSURE)),
                            "focus": float(rgb_camera.get(cv2.CAP_PROP_FOCUS)),
                        },
                    ),
                )
                print(
                    "Muestra de trayectoria guardada: "
                    f"{trajectory_recorder.sample_count}/"
                    f"{trajectory_recorder.samples_per_trajectory} | {metadata_path}"
                )
                if not trajectory_recorder.active:
                    last_auto_recorded_key = auto_candidate_key
                    print(
                        "Trayectoria completada: "
                        f"{trajectory_recorder.last_completed_directory}"
                    )

            rgb_low, rgb_high = edge_settings.thresholds("rgb")
            rgb_panel, rgb_edges = canny_edges_overlay(
                rgb_smoothed, rgb_low, rgb_high, (0, 255, 0)
            )

            depth_u8 = normalize_u8(aligned_depth_smoothed, valid_only=True)
            depth_low, depth_high = edge_settings.thresholds("depth")
            depth_panel, depth_edges = canny_edges_only(
                depth_u8, depth_low, depth_high, (255, 255, 255)
            )
            overlay_panel = depth_rgb_overlay(rgb, aligned_depth_smoothed)
            overlay_panel[depth_edges > 0] = (0, 255, 255)

            for item in detections:
                item_label = detection_display_label(item)
                item_color = detection_color(item.label)
                draw_detection(rgb_panel, item.bbox, item_label, item_color)
                draw_detection(overlay_panel, item.bbox, item_label, item_color)
                draw_detection(depth_panel, item.bbox, f"ROI {item.label}", item_color)

            panels = []
            for panel, title in (
                (rgb_panel, "RGB + Kalman + Canny"),
                (overlay_panel, "ENSAMBLE RGB + PROFUNDIDAD"),
                (depth_panel, "PROFUNDIDAD + Kalman + Canny"),
            ):
                panel = cv2.resize(panel, PANEL_SIZE, interpolation=cv2.INTER_AREA)
                add_panel_title(panel, title)
                panels.append(panel)

            telemetry = telemetry_panel(
                make_telemetry(
                    rgb,
                    aligned_depth_raw,
                    {
                        "rgb": rgb_edges,
                        "depth": depth_edges,
                    },
                    fps,
                    settings,
                    detection,
                    config,
                ),
                edge_settings,
                settings,
                capture_options,
                alignment_settings,
                alignment_mode,
                trajectory_recorder,
                yolo_tracker.inference_ms if yolo_tracker is not None else None,
                navigation_decision,
                motor_controller.status,
            )
            combined = np.vstack(
                (np.hstack((panels[0], panels[1])), np.hstack((panels[2], telemetry)))
            )
            cv2.imshow(APP_TITLE, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Salida solicitada con Q o Esc.")
                break
            if key in (ord("c"), ord("C")):
                path = save_raw_capture(
                    rgb,
                    depth_raw,
                    aligned_depth_raw,
                    detection,
                    config,
                    alignment_settings,
                    alignment_mode,
                    capture_options,
                )
                area_text = (
                    f", area={detection.area_px:.0f}px2"
                    if detection is not None
                    else ", area=sin objetivo"
                )
                print(
                    f"Datos crudos guardados: {path} "
                    f"| calibracion={capture_options.mark_for_calibration}"
                    f"{area_text}"
                )
            if key in (ord("p"), ord("P")):
                screenshot_count += 1
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = CAPTURE_DIR / f"panel_{screenshot_count:03d}_{stamp}.png"
                cv2.imwrite(str(path), combined)
                print(f"Captura de pantalla guardada: {path}")
            if key in (ord("g"), ord("G")):
                config.apply_alignment(alignment_settings)
                config.save(CONFIG_PATH)
                print(
                    "Ensamble guardado en camera_config.json: "
                    f"X={alignment_settings.offset_x}, "
                    f"Y={alignment_settings.offset_y}, "
                    f"escala={alignment_settings.scale:.3f}"
                )
            if key in (ord("f"), ord("F")):
                capture_options.toggle_calibration_mark()
                print(
                    "Marca de calibracion: "
                    f"{'ACTIVA' if capture_options.mark_for_calibration else 'INACTIVA'}"
                )
            if key in (10, 13):
                if trajectory_recorder.active:
                    trajectory_recorder.cancel()
                    print("Trayectoria cancelada.")
                else:
                    print("La captura de trayectorias es automatica; no necesitas Enter.")
            if key == ord("1"):
                edge_settings.selected_sensor = "rgb"
            if key == ord("2"):
                edge_settings.selected_sensor = "depth"
            if key in (ord("a"), ord("A")):
                edge_settings.adjust("low", 5)
            if key in (ord("z"), ord("Z")):
                edge_settings.adjust("low", -5)
            if key in (ord("s"), ord("S")):
                edge_settings.adjust("high", 5)
            if key in (ord("x"), ord("X")):
                edge_settings.adjust("high", -5)
            if key in (ord("i"), ord("I")):
                alignment_settings.adjust_offset(0, -2)
            if key in (ord("k"), ord("K")):
                alignment_settings.adjust_offset(0, 2)
            if key in (ord("j"), ord("J")):
                alignment_settings.adjust_offset(-2, 0)
            if key in (ord("l"), ord("L")):
                alignment_settings.adjust_offset(2, 0)
            if key in (ord("u"), ord("U")):
                alignment_settings.adjust_scale(-0.01)
            if key in (ord("o"), ord("O")):
                alignment_settings.adjust_scale(0.01)
            if key in (ord("r"), ord("R")):
                alignment_settings.reset()

    except KeyboardInterrupt:
        print("\nSalida solicitada desde el teclado.")
    except Exception as error:
        print(f"\nError: {error}")
        traceback.print_exc()
    finally:
        motor_controller.close()
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
