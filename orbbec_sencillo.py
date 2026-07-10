from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from primesense import openni2


APP_TITLE = "Orbbec sencillo | RGB segmenta + profundidad mide"
PANEL_SIZE = (640, 480)
CONFIG_PATH = Path("orbbec_sencillo_config.json")
FALLBACK_CONFIG_PATH = Path("camera_config.json")

DEFAULT_OPENNI_PATH = (
    r"C:\Users\roode\Downloads"
    r"\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\samples\bin"
)

RED_HSV_RANGES = ((0, 90, 45), (10, 255, 255), (170, 90, 45), (179, 255, 255))
BLUE_FLOOR_HSV_RANGE = ((90, 80, 50), (135, 255, 255))
TARGET_COLOR = "rojo"
TARGET_OBJECT = "cubo rojo"


@dataclass
class SimpleConfig:
    openni_path: str = DEFAULT_OPENNI_PATH
    rgb_index: int = 0
    rgb_backend: str = "auto"
    processing_width: int = 320
    processing_height: int = 240
    rgb_flip_horizontal: bool = True
    depth_flip_with_rgb: bool = True
    enable_hardware_registration: bool = False
    auto_depth_alignment: bool = True
    auto_alignment_max_shift_px: int = 160
    auto_alignment_smoothing: float = 0.80
    auto_alignment_search_radius_px: int = 30
    auto_alignment_search_step_px: int = 8
    auto_alignment_downscale: float = 0.50
    auto_alignment_max_edge_distance_px: float = 22.0
    auto_alignment_min_score: float = 0.035
    auto_alignment_max_jump_px: float = 35.0
    auto_alignment_update_interval_frames: int = 6
    alignment_roi_y_start_ratio: float = 0.25
    object_guided_alignment: bool = True
    object_guided_alignment_margin_px: int = 80
    object_guided_alignment_max_shift_px: int = 85
    object_guided_alignment_min_score: float = 0.18
    object_guided_alignment_smoothing: float = 0.65
    overlay_depth_alpha: float = 0.38
    depth_scale_mm: float = 1.0
    min_valid_depth_mm: float = 80.0
    max_valid_depth_mm: float = 4500.0
    depth_hold_frames: int = 6
    depth_to_rgb_offset_x: int = 0
    depth_to_rgb_offset_y: int = 0
    depth_to_rgb_scale: float = 1.0
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    depth_fx: Optional[float] = None
    depth_fy: Optional[float] = None
    depth_cx: Optional[float] = None
    depth_cy: Optional[float] = None
    depth_to_rgb_rotation: Optional[list[list[float]]] = None
    depth_to_rgb_translation_mm: Optional[list[float]] = None
    depth_to_rgb_homography: Optional[list[list[float]]] = None
    minimum_area_px: float = 200.0
    object_roi_y_start_ratio: float = 0.30
    texture_threshold: float = 85.0
    canny_low: int = 60
    canny_high: int = 160
    object_min_depth_valid_percent: float = 15.0
    object_depth_search_margin_px: int = 45
    object_hold_frames: int = 8
    object_distance_smoothing: float = 0.65
    require_object_depth: bool = False
    floor_roi_y_start_ratio: float = 0.55
    floor_min_blue_percent: float = 35.0
    floor_near_depth_mm: float = 900.0
    floor_max_near_percent: float = 8.0
    capture_enabled: bool = True
    capture_interval_seconds: float = 60.0
    capture_output_dir: str = "capturas_orbbec_sencillo"
    kalman_rgb_process_noise: float = 1.5
    kalman_rgb_measurement_noise: float = 30.0
    kalman_depth_process_noise: float = 2.0
    kalman_depth_measurement_noise: float = 80.0

    @classmethod
    def load(cls) -> "SimpleConfig":
        source = CONFIG_PATH if CONFIG_PATH.exists() else FALLBACK_CONFIG_PATH
        if not source.exists():
            config = cls()
            config.save()
            return config

        values = json.loads(source.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        config = cls(**{key: value for key, value in values.items() if key in allowed})
        if source != CONFIG_PATH:
            config.save()
        return config

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def has_depth_to_rgb_calibration(self) -> bool:
        values = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.depth_fx,
            self.depth_fy,
            self.depth_cx,
            self.depth_cy,
        )
        return bool(
            all(value is not None and float(value) > 0 for value in values)
            and self.depth_to_rgb_rotation is not None
            and self.depth_to_rgb_translation_mm is not None
        )

    @property
    def has_depth_to_rgb_homography(self) -> bool:
        if self.depth_to_rgb_homography is None:
            return False
        matrix = np.asarray(self.depth_to_rgb_homography, dtype=np.float32)
        return matrix.shape == (3, 3) and np.isfinite(matrix).all()


@dataclass
class Detection:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area_px: float
    color: str
    shape: str
    texture: str
    texture_score: float
    distance_mm: Optional[float]
    valid_depth_pixels: int
    depth_valid_percent: float
    reliability: str


@dataclass
class SensorData:
    rgb_resolution: tuple[int, int] = (0, 0)
    depth_resolution: tuple[int, int] = (0, 0)
    rgb_fps: float = 0.0
    depth_fps: float = 0.0
    depth_valid_pixels: int = 0
    depth_valid_percent: float = 0.0
    depth_min_mm: Optional[float] = None
    depth_max_mm: Optional[float] = None
    depth_mean_mm: Optional[float] = None
    depth_video_mode: str = "sin datos"
    rgb_exposure: Optional[float] = None
    rgb_gain: Optional[float] = None
    rgb_brightness: Optional[float] = None
    alignment_offset_x: float = 0.0
    alignment_offset_y: float = 0.0
    alignment_flip_depth: bool = False
    alignment_score: float = 0.0
    alignment_mode: str = "sin datos"
    registration_status: str = "sin datos"


@dataclass
class FloorData:
    roi_start_percent: float = 0.0
    depth_valid_percent: float = 0.0
    depth_median_mm: Optional[float] = None
    blue_percent: float = 0.0
    left_blue_percent: float = 0.0
    center_blue_percent: float = 0.0
    right_blue_percent: float = 0.0
    left_near_percent: float = 0.0
    center_near_percent: float = 0.0
    right_near_percent: float = 0.0
    status: str = "sin datos"


@dataclass
class AlignmentState:
    offset_x: float = 0.0
    offset_y: float = 0.0
    flip_depth: bool = True
    score: float = 0.0


class TemporalKalman:
    def __init__(self, process_noise: float, measurement_noise: float) -> None:
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


def normalize_u8(image: np.ndarray, valid_only: bool = False) -> np.ndarray:
    values = image[image > 0] if valid_only else image.reshape(-1)
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image.astype(np.float32) - float(low)) * (255.0 / (high - low))
    if valid_only:
        normalized[image <= 0] = 0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def depth_mm_values(depth: np.ndarray, config: SimpleConfig) -> np.ndarray:
    return depth.astype(np.float32) * float(config.depth_scale_mm)


def valid_depth_mask(depth: np.ndarray, config: SimpleConfig) -> np.ndarray:
    depth_mm = depth_mm_values(depth, config)
    return (
        (depth > 0)
        & (depth_mm >= float(config.min_valid_depth_mm))
        & (depth_mm <= float(config.max_valid_depth_mm))
    )


def normalize_depth_u8(depth: np.ndarray, config: SimpleConfig) -> np.ndarray:
    mask = valid_depth_mask(depth, config)
    values = depth[mask]
    if values.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        return np.zeros(depth.shape, dtype=np.uint8)
    normalized = (depth.astype(np.float32) - float(low)) * (255.0 / (high - low))
    normalized[~mask] = 0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def safe_camera_property(camera: cv2.VideoCapture, prop: int) -> Optional[float]:
    value = float(camera.get(prop))
    if value == 0.0:
        return None
    return value


def describe_depth_video_mode(depth_stream) -> str:
    try:
        mode = depth_stream.get_video_mode()
    except Exception:
        return "sin datos"

    width = getattr(mode, "resolutionX", None)
    height = getattr(mode, "resolutionY", None)
    fps = getattr(mode, "fps", None)
    pixel_format = getattr(mode, "pixelFormat", None)
    parts = []
    if width and height:
        parts.append(f"{width}x{height}")
    if fps:
        parts.append(f"{fps} fps")
    if pixel_format is not None:
        parts.append(str(pixel_format))
    return " | ".join(parts) if parts else str(mode)


def enable_depth_to_color_registration(device, enabled: bool) -> str:
    if not enabled:
        return "desactivado; usando bordes targetless"
    mode = getattr(openni2, "IMAGE_REGISTRATION_DEPTH_TO_COLOR", None)
    if mode is None:
        return "no disponible en openni2"
    try:
        if hasattr(device, "is_image_registration_mode_supported"):
            if not device.is_image_registration_mode_supported(mode):
                return "no soportado por dispositivo"
        device.set_image_registration_mode(mode)
        return "activado"
    except Exception as error:
        return f"no soportado; usando bordes targetless ({error})"


def depth_alignment_mode(config: SimpleConfig) -> str:
    if config.has_depth_to_rgb_calibration:
        return "calibracion_3d"
    if config.has_depth_to_rgb_homography:
        return "homografia"
    if config.auto_depth_alignment:
        return "targetless_bordes"
    return "manual"


def depth_sensor_stats(
    depth: np.ndarray,
    config: SimpleConfig,
) -> tuple[int, float, Optional[float], Optional[float], Optional[float]]:
    mask = valid_depth_mask(depth, config)
    valid = depth[mask]
    valid_pixels = int(valid.size)
    valid_percent = 100.0 * valid_pixels / max(depth.size, 1)
    if valid_pixels == 0:
        return valid_pixels, valid_percent, None, None, None
    depth_mm = valid.astype(np.float32) * float(config.depth_scale_mm)
    return (
        valid_pixels,
        valid_percent,
        float(depth_mm.min()),
        float(depth_mm.max()),
        float(depth_mm.mean()),
    )


def blue_floor_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    lower, upper = BLUE_FLOOR_HSV_RANGE
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def mask_percent(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)


def zone_slices(width: int) -> dict[str, tuple[int, int]]:
    return {
        "left": (0, int(width * 0.33)),
        "center": (int(width * 0.33), int(width * 0.67)),
        "right": (int(width * 0.67), width),
    }


def analyze_floor(rgb: np.ndarray, depth: np.ndarray, config: SimpleConfig) -> FloorData:
    height, width = rgb.shape[:2]
    start_ratio = float(np.clip(config.floor_roi_y_start_ratio, 0.0, 0.95))
    start_y = int(round(height * start_ratio))
    rgb_roi = rgb[start_y:height, :]
    depth_roi = depth[start_y:height, :]
    if rgb_roi.size == 0 or depth_roi.size == 0:
        return FloorData(roi_start_percent=start_ratio * 100.0)

    floor_mask = blue_floor_mask(rgb_roi)
    depth_valid = valid_depth_mask(depth_roi, config)
    depth_valid_percent = 100.0 * float(np.count_nonzero(depth_valid)) / float(depth_roi.size)
    depth_values = depth_roi[depth_valid]
    depth_median = None
    if depth_values.size > 0:
        depth_median = float(np.median(depth_values.astype(np.float32) * float(config.depth_scale_mm)))

    near_mask = (
        depth_valid
        & (depth_mm_values(depth_roi, config) <= float(config.floor_near_depth_mm))
    )
    blue_percent = mask_percent(floor_mask)
    zones = zone_slices(width)

    blue_by_zone: dict[str, float] = {}
    near_by_zone: dict[str, float] = {}
    for name, (x1, x2) in zones.items():
        blue_by_zone[name] = mask_percent(floor_mask[:, x1:x2])
        near_by_zone[name] = mask_percent(near_mask[:, x1:x2])

    center_ok = (
        blue_by_zone["center"] >= float(config.floor_min_blue_percent)
        and near_by_zone["center"] <= float(config.floor_max_near_percent)
    )
    if center_ok:
        status = "centro transitable"
    elif blue_percent < float(config.floor_min_blue_percent):
        status = "piso no confirmado"
    else:
        status = "centro no confiable"

    return FloorData(
        roi_start_percent=start_ratio * 100.0,
        depth_valid_percent=depth_valid_percent,
        depth_median_mm=depth_median,
        blue_percent=blue_percent,
        left_blue_percent=blue_by_zone["left"],
        center_blue_percent=blue_by_zone["center"],
        right_blue_percent=blue_by_zone["right"],
        left_near_percent=near_by_zone["left"],
        center_near_percent=near_by_zone["center"],
        right_near_percent=near_by_zone["right"],
        status=status,
    )


def build_sensor_data(
    rgb: np.ndarray,
    depth: np.ndarray,
    rgb_camera: cv2.VideoCapture,
    depth_video_mode: str,
    rgb_fps: float,
    depth_fps: float,
    config: SimpleConfig,
    alignment_state: AlignmentState,
    registration_status: str,
) -> SensorData:
    valid_pixels, valid_percent, min_mm, max_mm, mean_mm = depth_sensor_stats(
        depth,
        config,
    )
    return SensorData(
        rgb_resolution=(rgb.shape[1], rgb.shape[0]),
        depth_resolution=(depth.shape[1], depth.shape[0]),
        rgb_fps=rgb_fps,
        depth_fps=depth_fps,
        depth_valid_pixels=valid_pixels,
        depth_valid_percent=valid_percent,
        depth_min_mm=min_mm,
        depth_max_mm=max_mm,
        depth_mean_mm=mean_mm,
        depth_video_mode=depth_video_mode,
        rgb_exposure=safe_camera_property(rgb_camera, cv2.CAP_PROP_EXPOSURE),
        rgb_gain=safe_camera_property(rgb_camera, cv2.CAP_PROP_GAIN),
        rgb_brightness=safe_camera_property(rgb_camera, cv2.CAP_PROP_BRIGHTNESS),
        alignment_offset_x=alignment_state.offset_x,
        alignment_offset_y=alignment_state.offset_y,
        alignment_flip_depth=alignment_state.flip_depth,
        alignment_score=alignment_state.score,
        alignment_mode=depth_alignment_mode(config),
        registration_status=registration_status,
    )


def align_depth_to_rgb_manual(
    depth: np.ndarray,
    rgb_shape: tuple[int, ...],
    config: SimpleConfig,
) -> np.ndarray:
    rgb_height, rgb_width = rgb_shape[:2]
    resized_width = max(1, int(round(rgb_width * config.depth_to_rgb_scale)))
    resized_height = max(1, int(round(rgb_height * config.depth_to_rgb_scale)))
    resized = cv2.resize(depth, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    aligned = np.zeros((rgb_height, rgb_width), dtype=depth.dtype)

    dst_x0 = max(0, int(config.depth_to_rgb_offset_x))
    dst_y0 = max(0, int(config.depth_to_rgb_offset_y))
    src_x0 = max(0, -int(config.depth_to_rgb_offset_x))
    src_y0 = max(0, -int(config.depth_to_rgb_offset_y))
    width = min(rgb_width - dst_x0, resized_width - src_x0)
    height = min(rgb_height - dst_y0, resized_height - src_y0)
    if width <= 0 or height <= 0:
        return aligned

    aligned[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = resized[
        src_y0 : src_y0 + height,
        src_x0 : src_x0 + width,
    ]
    return aligned


def align_depth_to_rgb_homography(
    depth_raw: np.ndarray,
    rgb_shape: tuple[int, ...],
    config: SimpleConfig,
) -> np.ndarray:
    depth = cv2.flip(depth_raw, 1) if config.depth_flip_with_rgb else depth_raw
    base = align_depth_to_rgb_manual(depth, rgb_shape, config)
    matrix = np.asarray(config.depth_to_rgb_homography, dtype=np.float32)
    rgb_height, rgb_width = rgb_shape[:2]
    return cv2.warpPerspective(
        base,
        matrix,
        (rgb_width, rgb_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def translate_depth(depth: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    height, width = depth.shape[:2]
    translated = np.zeros_like(depth)
    dst_x0 = max(0, offset_x)
    dst_y0 = max(0, offset_y)
    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    copy_width = min(width - dst_x0, width - src_x0)
    copy_height = min(height - dst_y0, height - src_y0)
    if copy_width <= 0 or copy_height <= 0:
        return translated
    translated[dst_y0 : dst_y0 + copy_height, dst_x0 : dst_x0 + copy_width] = depth[
        src_y0 : src_y0 + copy_height,
        src_x0 : src_x0 + copy_width,
    ]
    return translated


def align_depth_to_rgb_calibrated(
    depth: np.ndarray,
    rgb_shape: tuple[int, ...],
    config: SimpleConfig,
) -> np.ndarray:
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
    u_rgb = np.rint(float(config.fx) * points_rgb[0] / z_rgb + float(config.cx)).astype(
        np.int32
    )
    v_rgb = np.rint(float(config.fy) * points_rgb[1] / z_rgb + float(config.cy)).astype(
        np.int32
    )
    inside = (u_rgb >= 0) & (u_rgb < rgb_width) & (v_rgb >= 0) & (v_rgb < rgb_height)
    u_rgb = u_rgb[inside]
    v_rgb = v_rgb[inside]
    z_metric = z_rgb[inside]
    if u_rgb.size == 0:
        return np.zeros((rgb_height, rgb_width), dtype=depth.dtype)

    z_raw = np.rint(z_metric / float(config.depth_scale_mm)).astype(depth.dtype)
    aligned = np.zeros((rgb_height, rgb_width), dtype=depth.dtype)
    z_buffer = np.full((rgb_height, rgb_width), np.inf, dtype=np.float32)
    for u, v, z_value, raw_value in zip(u_rgb, v_rgb, z_metric, z_raw):
        if z_value < z_buffer[v, u]:
            z_buffer[v, u] = z_value
            aligned[v, u] = raw_value
    return aligned


def binary_alignment_edges_from_rgb(rgb: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, low, high)


def binary_alignment_edges_from_depth(
    depth: np.ndarray,
    low: int,
    high: int,
    config: SimpleConfig,
) -> np.ndarray:
    depth_u8 = normalize_depth_u8(depth, config)
    depth_u8 = cv2.medianBlur(depth_u8, 5)
    edges = cv2.Canny(depth_u8, low, high)
    valid_edges = cv2.Canny((valid_depth_mask(depth, config).astype(np.uint8) * 255), 60, 120)
    return cv2.bitwise_or(edges, valid_edges)


def blurred_alignment_edges(edges: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(edges, (9, 9), 0).astype(np.float32) / 255.0


def resize_edges_for_alignment(edges: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.99:
        return edges
    height, width = edges.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(edges, size, interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 24, 255, cv2.THRESH_BINARY)
    return resized.astype(np.uint8)


def chamfer_alignment_score(
    distance_to_rgb_edge: np.ndarray,
    depth_edges: np.ndarray,
    offset_x: int,
    offset_y: int,
    max_edge_distance_px: float,
) -> float:
    shifted_edges = translate_depth(depth_edges, offset_x, offset_y)
    edge_mask = shifted_edges > 0
    edge_count = int(np.count_nonzero(edge_mask))
    if edge_count < 35:
        return 0.0

    distances = distance_to_rgb_edge[edge_mask]
    if distances.size == 0:
        return 0.0

    max_distance = max(1.0, float(max_edge_distance_px))
    clipped = np.minimum(distances, max_distance)
    close_ratio = float(np.mean(distances <= 2.5))
    distance_score = 1.0 - float(np.mean(clipped)) / max_distance
    density_reference = max(80.0, float(depth_edges.size) * 0.002)
    density_score = min(1.0, edge_count / density_reference)
    return max(0.0, (0.65 * close_ratio + 0.35 * distance_score) * density_score)


def estimate_depth_alignment(
    rgb: np.ndarray,
    depth: np.ndarray,
    config: SimpleConfig,
) -> tuple[int, int, float]:
    rgb_edges = binary_alignment_edges_from_rgb(rgb, config.canny_low, config.canny_high)
    depth_edges = binary_alignment_edges_from_depth(
        depth,
        config.canny_low,
        config.canny_high,
        config,
    )
    if not np.any(rgb_edges) or not np.any(depth_edges):
        return 0, 0, 0.0

    scale = float(np.clip(config.auto_alignment_downscale, 0.25, 1.0))
    rgb_edges_small = resize_edges_for_alignment(rgb_edges, scale)
    depth_edges_small = resize_edges_for_alignment(depth_edges, scale)
    if not np.any(rgb_edges_small) or not np.any(depth_edges_small):
        return 0, 0, 0.0

    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            blurred_alignment_edges(depth_edges_small),
            blurred_alignment_edges(rgb_edges_small),
        )
    except cv2.error:
        shift_x, shift_y, response = 0.0, 0.0, 0.0

    max_shift = max(1, int(round(config.auto_alignment_max_shift_px * scale)))
    phase_x = int(np.clip(round(shift_x), -max_shift, max_shift))
    phase_y = int(np.clip(round(shift_y), -max_shift, max_shift))
    search_radius = max(1, int(round(config.auto_alignment_search_radius_px * scale)))
    search_step = max(1, int(round(config.auto_alignment_search_step_px * scale)))
    distance_to_rgb_edge = cv2.distanceTransform(255 - rgb_edges_small, cv2.DIST_L2, 3)
    max_edge_distance = max(2.0, float(config.auto_alignment_max_edge_distance_px) * scale)
    base_offsets = {(0, 0), (phase_x, phase_y), (-phase_x, -phase_y)}
    best_x, best_y, best_score = 0, 0, 0.0

    coarse_step = max(search_step * 4, 10)
    for offset_y in range(-max_shift, max_shift + 1, coarse_step):
        for offset_x in range(-max_shift, max_shift + 1, coarse_step):
            score = chamfer_alignment_score(
                distance_to_rgb_edge,
                depth_edges_small,
                offset_x,
                offset_y,
                max_edge_distance,
            )
            if score > best_score:
                best_x, best_y, best_score = offset_x, offset_y, score

    for base_x, base_y in base_offsets:
        for offset_y in range(base_y - search_radius, base_y + search_radius + 1, search_step):
            if abs(offset_y) > max_shift:
                continue
            for offset_x in range(base_x - search_radius, base_x + search_radius + 1, search_step):
                if abs(offset_x) > max_shift:
                    continue
                score = chamfer_alignment_score(
                    distance_to_rgb_edge,
                    depth_edges_small,
                    offset_x,
                    offset_y,
                    max_edge_distance,
                )
                if score > best_score:
                    best_x, best_y, best_score = offset_x, offset_y, score

    refine_radius = max(1, search_step)
    for offset_y in range(best_y - refine_radius, best_y + refine_radius + 1):
        if abs(offset_y) > max_shift:
            continue
        for offset_x in range(best_x - refine_radius, best_x + refine_radius + 1):
            if abs(offset_x) > max_shift:
                continue
            score = chamfer_alignment_score(
                distance_to_rgb_edge,
                depth_edges_small,
                offset_x,
                offset_y,
                max_edge_distance,
            )
            if score > best_score:
                best_x, best_y, best_score = offset_x, offset_y, score

    best_score += max(0.0, float(response)) * 0.0005
    return int(round(best_x / scale)), int(round(best_y / scale)), best_score


def auto_align_depth(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    rgb_shape: tuple[int, ...],
    config: SimpleConfig,
    state: AlignmentState,
) -> np.ndarray:
    if config.has_depth_to_rgb_calibration:
        state.score = 1.0
        return align_depth_to_rgb_calibrated(depth_raw, rgb_shape, config)

    if config.has_depth_to_rgb_homography:
        state.score = 1.0
        state.offset_x = 0.0
        state.offset_y = 0.0
        state.flip_depth = config.depth_flip_with_rgb
        return align_depth_to_rgb_homography(depth_raw, rgb_shape, config)

    if not config.auto_depth_alignment:
        depth = cv2.flip(depth_raw, 1) if config.depth_flip_with_rgb else depth_raw
        return align_depth_to_rgb_manual(depth, rgb_shape, config)

    candidates: list[tuple[float, bool, int, int, np.ndarray]] = []
    for flip_depth in (False, True):
        candidate_raw = cv2.flip(depth_raw, 1) if flip_depth else depth_raw
        candidate = align_depth_to_rgb_manual(candidate_raw, rgb_shape, config)
        offset_x, offset_y, score = estimate_depth_alignment(rgb, candidate, config)
        candidates.append((score, flip_depth, offset_x, offset_y, candidate))

    score, flip_depth, offset_x, offset_y, candidate = max(candidates, key=lambda item: item[0])
    smoothing = float(np.clip(config.auto_alignment_smoothing, 0.0, 0.98))
    if score > 0.0:
        state.flip_depth = flip_depth
        state.offset_x = state.offset_x * smoothing + offset_x * (1.0 - smoothing)
        state.offset_y = state.offset_y * smoothing + offset_y * (1.0 - smoothing)
        state.score = state.score * smoothing + score * (1.0 - smoothing)

    if state.flip_depth != flip_depth:
        candidate_raw = cv2.flip(depth_raw, 1) if state.flip_depth else depth_raw
        candidate = align_depth_to_rgb_manual(candidate_raw, rgb_shape, config)
    return translate_depth(candidate, int(round(state.offset_x)), int(round(state.offset_y)))


def depth_rgb_overlay(
    rgb: np.ndarray,
    aligned_depth: np.ndarray,
    config: SimpleConfig,
) -> np.ndarray:
    depth_u8 = normalize_depth_u8(aligned_depth, config)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    mask = valid_depth_mask(aligned_depth, config)
    alpha = float(np.clip(config.overlay_depth_alpha, 0.05, 0.90))
    overlay = rgb.copy()
    overlay[mask] = np.clip(
        rgb[mask].astype(np.float32) * (1.0 - alpha)
        + depth_color[mask].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return overlay


def depth_visualization(aligned_depth: np.ndarray, config: SimpleConfig) -> np.ndarray:
    depth_u8 = normalize_depth_u8(aligned_depth, config)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~valid_depth_mask(aligned_depth, config)] = (0, 0, 0)
    return depth_color


def make_red_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    limits = RED_HSV_RANGES
    mask = cv2.inRange(hsv, np.array(limits[0]), np.array(limits[1]))
    mask |= cv2.inRange(hsv, np.array(limits[2]), np.array(limits[3]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def apply_object_roi(mask: np.ndarray, config: SimpleConfig) -> np.ndarray:
    roi_mask = mask.copy()
    height = roi_mask.shape[0]
    start_ratio = float(np.clip(config.object_roi_y_start_ratio, 0.0, 0.90))
    start_y = int(round(height * start_ratio))
    if start_y > 0:
        roi_mask[:start_y, :] = 0
    return roi_mask


def canny_edges(rgb: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


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


def cube_face_score(contour: np.ndarray, edges: np.ndarray) -> Optional[float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    x, y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return None

    ratio = width / height
    if not 0.65 <= ratio <= 1.45:
        return None

    rect_area = width * height
    extent = area / max(rect_area, 1)
    if extent < 0.35:
        return None

    edge_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.drawContours(edge_mask, [contour], -1, 255, 2)
    contour_edge_pixels = int(np.count_nonzero((edges > 0) & (edge_mask > 0)))
    edge_score = contour_edge_pixels / max(perimeter, 1.0)
    edge_score = min(edge_score, 1.0)

    vertex_score = 1.0 / (1.0 + abs(len(polygon) - 4))
    ratio_score = 1.0 - min(abs(1.0 - ratio), 0.45) / 0.45
    extent_score = min(extent, 1.0)
    return float(
        area
        * (
            0.35 * vertex_score
            + 0.30 * ratio_score
            + 0.25 * extent_score
            + 0.10 * edge_score
        )
    )


def texture_score(gray: np.ndarray, contour: np.ndarray) -> float:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, width, height = cv2.boundingRect(contour)
    roi = gray[y : y + height, x : x + width]
    roi_mask = mask[y : y + height, x : x + width]
    if roi.size == 0 or np.count_nonzero(roi_mask) < 20:
        return 0.0
    laplacian = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    values = laplacian[roi_mask > 0]
    return float(values.var()) if values.size else 0.0


def texture_name(score: float, threshold: float) -> str:
    return "texturizada" if score >= threshold else "lisa"


def median_depth_in_contour(
    depth: np.ndarray,
    contour: np.ndarray,
    config: SimpleConfig,
) -> tuple[Optional[float], int]:
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    valid = depth[(mask > 0) & valid_depth_mask(depth, config)]
    if valid.size < 8:
        return None, int(valid.size)

    low, high = np.percentile(valid, (10, 90))
    trimmed = valid[(valid >= low) & (valid <= high)]
    if trimmed.size >= 8:
        valid = trimmed
    return float(np.median(valid) * float(config.depth_scale_mm)), int(valid.size)


def object_depth_reliability(
    contour_area: float,
    valid_depth_pixels: int,
    config: SimpleConfig,
) -> tuple[float, str]:
    valid_percent = 100.0 * float(valid_depth_pixels) / max(float(contour_area), 1.0)
    if valid_depth_pixels < 8:
        return valid_percent, "sin profundidad"
    if valid_percent < float(config.object_min_depth_valid_percent):
        return valid_percent, "distancia poco confiable"
    return valid_percent, "confiable"


def find_objects(
    rgb: np.ndarray,
    aligned_depth: np.ndarray,
    config: SimpleConfig,
) -> list[Detection]:
    edges = canny_edges(rgb, config.canny_low, config.canny_high)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    detections: list[Detection] = []
    mask = apply_object_roi(make_red_mask(rgb), config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray, str, str, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.minimum_area_px:
            continue

        detected_shape = shape_name(contour)
        if detected_shape == "circulo":
            continue
        cube_score = cube_face_score(contour, edges)
        if cube_score is None:
            continue

        score = texture_score(gray, contour)
        detected_texture = texture_name(score, config.texture_threshold)
        candidates.append((cube_score, contour, detected_shape, detected_texture, score))

    if not candidates:
        return []

    _, contour, detected_shape, detected_texture, score = max(
        candidates,
        key=lambda item: item[0],
    )
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
    else:
        center = (x + width // 2, y + height // 2)
    distance_mm, valid_pixels = median_depth_in_contour(
        aligned_depth,
        contour,
        config,
    )
    depth_valid_percent, reliability = object_depth_reliability(
        area,
        valid_pixels,
        config,
    )
    if config.require_object_depth and reliability != "confiable":
        return []
    detections.append(
        Detection(
            contour=contour,
            bbox=(x, y, width, height),
            center=center,
            area_px=area,
            color=TARGET_COLOR,
            shape=detected_shape,
            texture=detected_texture,
            texture_score=score,
            distance_mm=distance_mm,
            valid_depth_pixels=valid_pixels,
            depth_valid_percent=depth_valid_percent,
            reliability=reliability,
        )
    )

    return detections


def detection_draw_color() -> tuple[int, int, int]:
    return (0, 0, 255)


def draw_detections(panel: np.ndarray, detections: list[Detection], config: SimpleConfig) -> None:
    if not detections:
        return

    for detection in detections:
        x, y, width, height = detection.bbox
        color = detection_draw_color()
        cv2.drawContours(panel, [detection.contour], -1, color, 2)
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, 2)
        cv2.drawMarker(panel, detection.center, color, cv2.MARKER_CROSS, 18, 2)
        if detection.reliability == "confiable" and detection.distance_mm is not None:
            distance = f"{detection.distance_mm:.0f} mm"
        elif detection.valid_depth_pixels > 0:
            distance = "depth no fiable"
        else:
            distance = "sin depth"
        label = f"cubo {detection.color} | {distance}"
        label_y = y - 8 if y >= 28 else y + height + 22
        cv2.putText(
            panel,
            label,
            (x, min(panel.shape[0] - 8, max(24, label_y))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def save_detection_capture(
    config: SimpleConfig,
    rgb_panel: np.ndarray,
    depth_panel: np.ndarray,
    overlay_panel: np.ndarray,
    detection: Detection,
) -> list[Path]:
    output_dir = Path(config.capture_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    distance = (
        f"{int(round(detection.distance_mm))}mm"
        if detection.reliability == "confiable" and detection.distance_mm is not None
        else "sin_depth"
    )
    prefix = f"{timestamp}_{depth_alignment_mode(config)}_cubo_rojo_{distance}"
    images = {
        "rgb": rgb_panel,
        "profundidad": depth_panel,
        "empalme": overlay_panel,
    }
    saved_paths: list[Path] = []
    for name, image in images.items():
        path = output_dir / f"{prefix}_{name}.png"
        cv2.imwrite(str(path), image)
        saved_paths.append(path)
    return saved_paths


def add_title(panel: np.ndarray, title: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (15, 15, 15), -1)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def format_optional_mm(value: Optional[float]) -> str:
    return f"{value:.0f} mm" if value is not None else "sin datos"


def format_optional_float(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "sin datos"


def make_info_panel(
    config: SimpleConfig,
    detection: Optional[Detection],
    sensor_data: SensorData,
    floor_data: FloorData,
) -> np.ndarray:
    panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), (24, 24, 24), dtype=np.uint8)
    lines = [
        "ORBBEC",
        f"Objetivo fijo: {TARGET_OBJECT}",
        "",
        f"RGB sensor: {sensor_data.rgb_resolution[0]}x{sensor_data.rgb_resolution[1]} | {sensor_data.rgb_fps:.1f} fps",
        f"RGB exp/gain/brillo: {format_optional_float(sensor_data.rgb_exposure)} / {format_optional_float(sensor_data.rgb_gain)} / {format_optional_float(sensor_data.rgb_brightness)}",
        f"Depth alineada: {sensor_data.depth_resolution[0]}x{sensor_data.depth_resolution[1]} | {sensor_data.depth_fps:.1f} fps",
        f"Depth modo: {sensor_data.depth_video_mode}",
        f"Modo empalme depth->RGB: {sensor_data.alignment_mode}",
        f"Registro depth->color: {sensor_data.registration_status}",
        (
            "Depth valido rango: "
            f"{config.min_valid_depth_mm:.0f}-{config.max_valid_depth_mm:.0f} mm"
        ),
        f"Depth validos: {sensor_data.depth_valid_pixels} ({sensor_data.depth_valid_percent:.1f}%)",
        (
            "Depth min/max/media: "
            f"{format_optional_mm(sensor_data.depth_min_mm)} / "
            f"{format_optional_mm(sensor_data.depth_max_mm)} / "
            f"{format_optional_mm(sensor_data.depth_mean_mm)}"
        ),
        "",
        f"Area minima: {config.minimum_area_px:.0f} px2",
        f"Procesamiento: {config.processing_width}x{config.processing_height}",
        f"ROI cubo rojo: desde {config.object_roi_y_start_ratio * 100:.0f}% vertical",
        f"Depth obligatorio para detectar: {'si' if config.require_object_depth else 'no'}",
        f"Umbral textura: {config.texture_threshold:.1f}",
        (
            "Empalme X/Y/escala: "
            f"{config.depth_to_rgb_offset_x:+d} / "
            f"{config.depth_to_rgb_offset_y:+d} / "
            f"{config.depth_to_rgb_scale:.2f}"
        ),
        (
            "Auto empalme X/Y/flip/score: "
            f"{sensor_data.alignment_offset_x:+.1f} / "
            f"{sensor_data.alignment_offset_y:+.1f} / "
            f"{'si' if sensor_data.alignment_flip_depth else 'no'} / "
            f"{sensor_data.alignment_score:.4f}"
        ),
        (
            "Calibracion depth->RGB: "
            f"{'activa' if config.has_depth_to_rgb_calibration else 'no encontrada'}"
        ),
        (
            "Homografia depth->RGB: "
            f"{'activa' if config.has_depth_to_rgb_homography else 'no encontrada'}"
        ),
        "",
        f"Suelo estado: {floor_data.status}",
        (
            "Suelo ROI/depth: "
            f"desde {floor_data.roi_start_percent:.0f}% | "
            f"{floor_data.depth_valid_percent:.1f}% valido | "
            f"mediana {format_optional_mm(floor_data.depth_median_mm)}"
        ),
        (
            "Piso azul L/C/R: "
            f"{floor_data.left_blue_percent:.0f}% / "
            f"{floor_data.center_blue_percent:.0f}% / "
            f"{floor_data.right_blue_percent:.0f}%"
        ),
        (
            "Depth cercano L/C/R: "
            f"{floor_data.left_near_percent:.0f}% / "
            f"{floor_data.center_near_percent:.0f}% / "
            f"{floor_data.right_near_percent:.0f}%"
        ),
    ]
    if detection is not None:
        distance = (
            f"{detection.distance_mm:.1f} mm"
            if detection.reliability == "confiable" and detection.distance_mm is not None
            else "sin dato confiable"
        )
        lines.extend(
            [
                "",
                f"Detectado RGB: cubo {detection.color}/{detection.shape}/{detection.texture}",
                f"Centro RGB: {detection.center}",
                f"Area: {detection.area_px:.0f} px2",
                f"Textura score: {detection.texture_score:.1f}",
                f"Distancia sensor-objeto: {distance}",
                (
                    "Profundidad objeto: "
                    f"{detection.valid_depth_pixels} px "
                    f"({detection.depth_valid_percent:.1f}%) | "
                    f"{detection.reliability}"
                ),
            ]
        )
    else:
        lines.extend(["", "Detectado: no"])

    y = 34
    for index, line in enumerate(lines):
        color = (90, 220, 255) if index == 0 else (235, 235, 235)
        scale = 0.72 if index == 0 else 0.48
        thickness = 2 if index == 0 else 1
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 28 if index == 0 else 22
    return panel


def rgb_backend_candidates(config: SimpleConfig) -> list[int]:
    requested = str(config.rgb_backend).strip().lower()
    named_backends = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    if requested in named_backends:
        return [named_backends[requested], cv2.CAP_ANY]
    if platform.system().lower() == "windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_rgb_camera(index: int, config: SimpleConfig) -> cv2.VideoCapture:
    camera = cv2.VideoCapture()
    for backend in rgb_backend_candidates(config):
        candidate = cv2.VideoCapture(index, backend)
        if not candidate.isOpened():
            candidate.release()
            continue
        camera = candidate
        break
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.processing_width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.processing_height))
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def prepare_rgb_frame(rgb: np.ndarray, config: SimpleConfig) -> np.ndarray:
    width = max(160, int(config.processing_width))
    height = max(120, int(config.processing_height))
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    config = SimpleConfig.load()
    rgb_camera = None
    depth_stream = None
    openni_initialized = False
    frames = 0
    fps = 0.0
    rgb_failures = 0
    last_fps_time = time.perf_counter()
    last_capture_time = 0.0
    rgb_filter = TemporalKalman(
        config.kalman_rgb_process_noise,
        config.kalman_rgb_measurement_noise,
    )
    depth_filter = TemporalKalman(
        config.kalman_depth_process_noise,
        config.kalman_depth_measurement_noise,
    )
    alignment_state = AlignmentState(flip_depth=config.depth_flip_with_rgb)

    try:
        rgb_camera = open_rgb_camera(config.rgb_index, config)
        if not rgb_camera.isOpened():
            raise RuntimeError(f"No se pudo abrir RGB en indice {config.rgb_index}.")

        openni2.initialize(config.openni_path)
        openni_initialized = True
        device = openni2.Device.open_any()
        registration_status = enable_depth_to_color_registration(
            device,
            config.enable_hardware_registration,
        )
        depth_stream = device.create_depth_stream()
        depth_stream.start()
        depth_video_mode = describe_depth_video_mode(depth_stream)

        print("Orbbec sencillo iniciado. Detectando cubo rojo.")
        print(f"Profundidad: {depth_video_mode}")
        print(f"Registro depth->color: {registration_status}")

        while True:
            ok, rgb = rgb_camera.read()
            if not ok or rgb is None:
                rgb_failures += 1
                if rgb_failures == 1:
                    print("RGB sin frame; intentando recuperar stream...")
                if rgb_failures >= 20:
                    print("Reabriendo camara RGB...")
                    rgb_camera.release()
                    time.sleep(0.20)
                    rgb_camera = open_rgb_camera(config.rgb_index, config)
                    rgb_failures = 0
                time.sleep(0.03)
                continue
            rgb_failures = 0
            if config.rgb_flip_horizontal:
                rgb = cv2.flip(rgb, 1)
            rgb = prepare_rgb_frame(rgb, config)
            rgb_detection = rgb.copy()
            rgb_smoothed = rgb_filter.update(rgb).astype(np.uint8)

            depth_frame = depth_stream.read_frame()
            depth_raw = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
            depth_raw = depth_raw.reshape(depth_frame.height, depth_frame.width).copy()
            aligned_depth_raw = auto_align_depth(
                rgb_smoothed,
                depth_raw,
                rgb.shape,
                config,
                alignment_state,
            )
            aligned_depth = depth_filter.update(aligned_depth_raw).astype(depth_raw.dtype)

            frames += 1
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = frames / elapsed
                frames = 0
                last_fps_time = now

            detections = find_objects(rgb_detection, aligned_depth, config)
            detection = detections[0] if detections else None
            floor_data = analyze_floor(rgb_detection, aligned_depth, config)
            sensor_data = build_sensor_data(
                rgb_smoothed,
                aligned_depth,
                rgb_camera,
                depth_video_mode,
                fps,
                fps,
                config,
                alignment_state,
                registration_status,
            )
            rgb_panel = rgb_smoothed.copy()
            depth_panel = depth_visualization(aligned_depth, config)
            overlay_panel = depth_rgb_overlay(rgb_smoothed, aligned_depth, config)
            draw_detections(rgb_panel, detections, config)
            draw_detections(depth_panel, detections, config)
            draw_detections(overlay_panel, detections, config)

            capture_interval = max(1.0, float(config.capture_interval_seconds))
            if (
                config.capture_enabled
                and detection is not None
                and now - last_capture_time >= capture_interval
            ):
                saved_paths = save_detection_capture(
                    config,
                    rgb_panel,
                    depth_panel,
                    overlay_panel,
                    detection,
                )
                last_capture_time = now
                print("Captura guardada:", ", ".join(str(path) for path in saved_paths))

            panels = [
                (rgb_panel, "Sensor RGB + Kalman"),
                (depth_panel, "Sensor profundidad alineado"),
                (overlay_panel, "Empalme RGB + profundidad"),
                (
                    make_info_panel(config, detection, sensor_data, floor_data),
                    "Datos reales del objetivo y sensores",
                ),
            ]
            rendered = []
            for panel, title in panels:
                panel = cv2.resize(panel, PANEL_SIZE, interpolation=cv2.INTER_AREA)
                add_title(panel, title)
                rendered.append(panel)

            combined = np.vstack(
                (
                    np.hstack((rendered[0], rendered[1])),
                    np.hstack((rendered[2], rendered[3])),
                )
            )
            cv2.imshow(APP_TITLE, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("\nSalida solicitada.")
    finally:
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
        config.save()
        print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
