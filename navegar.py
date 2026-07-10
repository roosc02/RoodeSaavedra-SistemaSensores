from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    """Umbrales conservadores para navegar primero en modo simulacion."""

    depth_scale_mm: float = 1.0
    min_valid_zone_pixels: int = 80
    center_near_stop_mm: float = 350.0
    center_slow_mm: float = 700.0
    person_stop_mm: float = 500.0
    person_slow_mm: float = 900.0
    car_stop_mm: float = 800.0
    dog_stop_mm: float = 450.0
    sign_slow_mm: float = 600.0
    turn_clearance_margin_mm: float = 120.0
    forward_speed: float = 0.30
    slow_speed: float = 0.15
    turn_speed: float = 0.20


@dataclass(frozen=True)
class ZoneDepths:
    left_mm: Optional[float]
    center_mm: Optional[float]
    right_mm: Optional[float]
    left_valid_pixels: int
    center_valid_pixels: int
    right_valid_pixels: int


@dataclass(frozen=True)
class NavigationDecision:
    action: str
    speed: float
    turn: float
    reason: str
    zones: ZoneDepths
    target_label: Optional[str] = None
    target_depth_mm: Optional[float] = None

    @property
    def text(self) -> str:
        if self.target_label and self.target_depth_mm is not None:
            return (
                f"{self.action} | v={self.speed:.2f} giro={self.turn:.2f} | "
                f"{self.target_label} {self.target_depth_mm:.0f}mm | {self.reason}"
            )
        return (
            f"{self.action} | v={self.speed:.2f} giro={self.turn:.2f} | "
            f"{self.reason}"
        )


def median_valid_depth_mm(
    depth: np.ndarray,
    scale_mm: float,
    min_valid_pixels: int,
) -> tuple[Optional[float], int]:
    if depth.size == 0:
        return None, 0
    valid = depth[depth > 0]
    count = int(valid.size)
    if count < min_valid_pixels:
        return None, count
    return float(np.median(valid) * scale_mm), count


def summarize_depth_zones(
    aligned_depth: np.ndarray,
    config: NavigationConfig,
) -> ZoneDepths:
    """Divide la profundidad en izquierda, centro y derecha."""

    if aligned_depth.size == 0:
        return ZoneDepths(None, None, None, 0, 0, 0)

    height, width = aligned_depth.shape[:2]
    y1 = int(height * 0.30)
    y2 = int(height * 0.90)
    x_left_end = int(width * 0.38)
    x_center_start = int(width * 0.32)
    x_center_end = int(width * 0.68)
    x_right_start = int(width * 0.62)

    left, left_count = median_valid_depth_mm(
        aligned_depth[y1:y2, 0:x_left_end],
        config.depth_scale_mm,
        config.min_valid_zone_pixels,
    )
    center, center_count = median_valid_depth_mm(
        aligned_depth[y1:y2, x_center_start:x_center_end],
        config.depth_scale_mm,
        config.min_valid_zone_pixels,
    )
    right, right_count = median_valid_depth_mm(
        aligned_depth[y1:y2, x_right_start:width],
        config.depth_scale_mm,
        config.min_valid_zone_pixels,
    )

    return ZoneDepths(
        left_mm=left,
        center_mm=center,
        right_mm=right,
        left_valid_pixels=left_count,
        center_valid_pixels=center_count,
        right_valid_pixels=right_count,
    )


def detection_label(detection: Any) -> Optional[str]:
    label = getattr(detection, "label", None)
    if label is None:
        return None
    normalized = str(label).strip().lower()
    return normalized.replace("\u00f1", "n")


def detection_depth_mm(detection: Any) -> Optional[float]:
    depth = getattr(detection, "depth_mm", None)
    if depth is None:
        return None
    try:
        value = float(depth)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def is_centered_detection(detection: Any, image_width: int) -> bool:
    center = getattr(detection, "center", None)
    if not center or image_width <= 0:
        return False
    x = float(center[0])
    return image_width * 0.35 <= x <= image_width * 0.65


def choose_turn(zones: ZoneDepths, config: NavigationConfig) -> tuple[str, float, str]:
    left = zones.left_mm
    right = zones.right_mm

    if left is None and right is None:
        return "stop", 0.0, "sin profundidad lateral confiable"
    if left is None:
        return "turn_right", 1.0, "derecha es el unico lado con profundidad"
    if right is None:
        return "turn_left", -1.0, "izquierda es el unico lado con profundidad"

    if left > right + config.turn_clearance_margin_mm:
        return "turn_left", -1.0, "izquierda mas libre"
    if right > left + config.turn_clearance_margin_mm:
        return "turn_right", 1.0, "derecha mas libre"

    return "turn_left", -1.0, "lados similares, giro preventivo a izquierda"


def class_risk_action(
    label: Optional[str],
    depth_mm: Optional[float],
    centered: bool,
    config: NavigationConfig,
) -> Optional[tuple[str, float, float, str]]:
    if label is None or depth_mm is None:
        return None

    if label == "persona":
        if depth_mm <= config.person_stop_mm:
            return "stop", 0.0, 0.0, "persona demasiado cerca"
        if centered and depth_mm <= config.person_slow_mm:
            return "slow", config.slow_speed, 0.0, "persona al frente"

    if label == "carro":
        if depth_mm <= config.car_stop_mm:
            return "stop", 0.0, 0.0, "carro cerca"

    if label == "perro":
        if depth_mm <= config.dog_stop_mm:
            return "stop", 0.0, 0.0, "perro cerca"

    if label == "senalamiento_trafico":
        if centered and depth_mm <= config.sign_slow_mm:
            return "slow", config.slow_speed, 0.0, "senalamiento cerca"

    return None


def decide_navigation(
    detections: list[Any],
    aligned_depth: np.ndarray,
    config: NavigationConfig,
) -> NavigationDecision:
    """Devuelve una decision de navegacion sin accionar motores."""

    zones = summarize_depth_zones(aligned_depth, config)
    image_width = int(aligned_depth.shape[1]) if aligned_depth.ndim >= 2 else 0
    measured = [
        item
        for item in detections
        if detection_depth_mm(item) is not None
    ]
    target = min(
        measured,
        key=lambda item: detection_depth_mm(item) or float("inf"),
        default=None,
    )
    target_label = detection_label(target) if target is not None else None
    target_depth = detection_depth_mm(target) if target is not None else None

    if zones.center_mm is None:
        return NavigationDecision(
            action="stop",
            speed=0.0,
            turn=0.0,
            reason="profundidad frontal no confiable",
            zones=zones,
            target_label=target_label,
            target_depth_mm=target_depth,
        )

    centered = bool(target is not None and is_centered_detection(target, image_width))
    risk = class_risk_action(target_label, target_depth, centered, config)
    if risk is not None:
        action, speed, turn, reason = risk
        return NavigationDecision(
            action,
            speed,
            turn,
            reason,
            zones,
            target_label,
            target_depth,
        )

    if zones.center_mm <= config.center_near_stop_mm:
        action, turn, reason = choose_turn(zones, config)
        if action == "stop":
            return NavigationDecision(
                action,
                0.0,
                turn,
                reason,
                zones,
                target_label,
                target_depth,
            )
        return NavigationDecision(
            action=action,
            speed=config.turn_speed,
            turn=turn,
            reason=f"obstaculo frontal a {zones.center_mm:.0f}mm, {reason}",
            zones=zones,
            target_label=target_label,
            target_depth_mm=target_depth,
        )

    if zones.center_mm <= config.center_slow_mm:
        action, turn, reason = choose_turn(zones, config)
        if action == "stop":
            return NavigationDecision(
                "slow",
                config.slow_speed,
                0.0,
                reason,
                zones,
                target_label,
                target_depth,
            )
        return NavigationDecision(
            action=action,
            speed=config.turn_speed,
            turn=turn,
            reason=f"espacio frontal reducido a {zones.center_mm:.0f}mm, {reason}",
            zones=zones,
            target_label=target_label,
            target_depth_mm=target_depth,
        )

    return NavigationDecision(
        action="forward",
        speed=config.forward_speed,
        turn=0.0,
        reason=f"frente libre a {zones.center_mm:.0f}mm",
        zones=zones,
        target_label=target_label,
        target_depth_mm=target_depth,
    )
