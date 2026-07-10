from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MotorPins:
    left_forward: Optional[int]
    left_backward: Optional[int]
    right_forward: Optional[int]
    right_backward: Optional[int]

    @property
    def complete(self) -> bool:
        return all(
            pin is not None
            for pin in (
                self.left_forward,
                self.left_backward,
                self.right_forward,
                self.right_backward,
            )
        )


@dataclass(frozen=True)
class MotorConfig:
    enabled: bool = False
    dry_run: bool = True
    max_speed: float = 0.30
    pins: MotorPins = MotorPins(None, None, None, None)


class MotorController:
    """Control moderado para dos motores DC con puente H y GPIO BCM."""

    def __init__(self, config: MotorConfig):
        self.config = config
        self.status = "motores desactivados"
        self.last_command: tuple[str, float, float] = ("stop", 0.0, 0.0)
        self.left_forward = None
        self.left_backward = None
        self.right_forward = None
        self.right_backward = None

        if not config.enabled:
            return
        if not config.pins.complete:
            self.status = "pines de motores incompletos"
            return
        if config.dry_run:
            self.status = "motores en simulacion"
            return

        try:
            from gpiozero import PWMOutputDevice
        except ImportError:
            self.status = "falta instalar gpiozero"
            return

        self.left_forward = PWMOutputDevice(config.pins.left_forward)
        self.left_backward = PWMOutputDevice(config.pins.left_backward)
        self.right_forward = PWMOutputDevice(config.pins.right_forward)
        self.right_backward = PWMOutputDevice(config.pins.right_backward)
        self.status = "motores activos"

    @property
    def active(self) -> bool:
        return (
            self.config.enabled
            and not self.config.dry_run
            and self.left_forward is not None
            and self.left_backward is not None
            and self.right_forward is not None
            and self.right_backward is not None
        )

    def clamp_speed(self, value: float) -> float:
        limit = max(0.0, min(float(self.config.max_speed), 1.0))
        return max(-limit, min(float(value), limit))

    def set_wheels(self, left: float, right: float) -> None:
        left = self.clamp_speed(left)
        right = self.clamp_speed(right)

        if not self.active:
            return

        self.left_forward.value = max(left, 0.0)
        self.left_backward.value = max(-left, 0.0)
        self.right_forward.value = max(right, 0.0)
        self.right_backward.value = max(-right, 0.0)

    def apply_decision(self, decision) -> None:
        action = getattr(decision, "action", "stop")
        speed = self.clamp_speed(getattr(decision, "speed", 0.0))
        turn = float(getattr(decision, "turn", 0.0))

        if action in {"stop", "hold"}:
            left = 0.0
            right = 0.0
        elif action in {"forward", "slow"}:
            left = speed
            right = speed
        elif action == "turn_left":
            left = -speed * 0.75
            right = speed * 0.75
        elif action == "turn_right":
            left = speed * 0.75
            right = -speed * 0.75
        else:
            left = speed * (1.0 + turn)
            right = speed * (1.0 - turn)

        self.last_command = (action, left, right)
        self.set_wheels(left, right)

    def stop(self) -> None:
        self.last_command = ("stop", 0.0, 0.0)
        self.set_wheels(0.0, 0.0)

    def close(self) -> None:
        self.stop()
        for device in (
            self.left_forward,
            self.left_backward,
            self.right_forward,
            self.right_backward,
        ):
            if device is not None:
                device.close()
