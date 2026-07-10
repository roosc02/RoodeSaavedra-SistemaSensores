from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from primesense import openni2

from orbbec_sencillo import (
    SimpleConfig,
    add_title,
    align_depth_to_rgb_manual,
    depth_rgb_overlay,
    depth_visualization,
    open_rgb_camera,
    prepare_rgb_frame,
)


WINDOW = "Calibrar empalme depth -> RGB"


def load_config(path: Path) -> SimpleConfig:
    if not path.exists():
        config = SimpleConfig()
        path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
        return config

    values = json.loads(path.read_text(encoding="utf-8"))
    allowed = SimpleConfig.__dataclass_fields__.keys()
    return SimpleConfig(**{key: value for key, value in values.items() if key in allowed})


def save_homography(path: Path, config: SimpleConfig, homography: np.ndarray) -> None:
    values = {}
    if path.exists():
        values = json.loads(path.read_text(encoding="utf-8"))
    values.update(asdict(config))
    values["depth_to_rgb_homography"] = homography.astype(float).tolist()
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def save_config(path: Path, config: SimpleConfig) -> None:
    values = {}
    if path.exists():
        values = json.loads(path.read_text(encoding="utf-8"))
    values.update(asdict(config))
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def depth_flip_text(config: SimpleConfig) -> str:
    return "DEPTH FLIP SI" if config.depth_flip_with_rgb else "DEPTH FLIP NO"


def rgb_flip_text(config: SimpleConfig) -> str:
    return "RGB FLIP SI" if config.rgb_flip_horizontal else "RGB FLIP NO"


def reset_calibration_points(
    rgb_points: list[tuple[float, float]],
    depth_points: list[tuple[float, float]],
) -> None:
    rgb_points.clear()
    depth_points.clear()


def draw_status(panel: np.ndarray, text: str) -> None:
    height = panel.shape[0]
    cv2.rectangle(panel, (0, height - 30), (panel.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (10, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def read_rgb(camera: cv2.VideoCapture, config: SimpleConfig) -> np.ndarray:
    ok, rgb = camera.read()
    if not ok or rgb is None:
        raise RuntimeError("No se pudo leer frame RGB.")
    if config.rgb_flip_horizontal:
        rgb = cv2.flip(rgb, 1)
    return prepare_rgb_frame(rgb, config)


def read_depth(depth_stream, rgb_shape: tuple[int, ...], config: SimpleConfig) -> np.ndarray:
    depth_frame = depth_stream.read_frame()
    depth_raw = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
    depth_raw = depth_raw.reshape(depth_frame.height, depth_frame.width).copy()
    depth = cv2.flip(depth_raw, 1) if config.depth_flip_with_rgb else depth_raw
    return align_depth_to_rgb_manual(depth, rgb_shape, config)


def make_depth_panel(depth_base: np.ndarray, config: SimpleConfig) -> np.ndarray:
    return depth_visualization(depth_base, config)


def draw_points(
    rgb: np.ndarray,
    depth_panel: np.ndarray,
    rgb_points: list[tuple[float, float]],
    depth_points: list[tuple[float, float]],
    config: SimpleConfig,
) -> np.ndarray:
    rgb_draw = rgb.copy()
    depth_draw = depth_panel.copy()
    for index, point in enumerate(rgb_points):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(rgb_draw, center, 5, (0, 0, 255), -1)
        cv2.putText(rgb_draw, str(index + 1), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    for index, point in enumerate(depth_points):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(depth_draw, center, 5, (0, 255, 255), -1)
        cv2.putText(depth_draw, str(index + 1), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    add_title(rgb_draw, f"RGB puntos | {rgb_flip_text(config)}")
    add_title(depth_draw, f"Depth puntos | {depth_flip_text(config)}")
    draw_status(rgb_draw, "R cambia espejo RGB")
    draw_status(depth_draw, "F cambia espejo depth")
    return np.hstack((rgb_draw, depth_draw))


def compute_homography(
    rgb_points: list[tuple[float, float]],
    depth_points: list[tuple[float, float]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(rgb_points) < 4 or len(depth_points) < 4 or len(rgb_points) != len(depth_points):
        return None, None
    src_depth = np.asarray(depth_points, dtype=np.float32)
    dst_rgb = np.asarray(rgb_points, dtype=np.float32)
    homography, inliers = cv2.findHomography(src_depth, dst_rgb, cv2.RANSAC, 4.0)
    return homography, inliers


def preview_overlay(
    rgb: np.ndarray,
    depth_base: np.ndarray,
    homography: np.ndarray,
    config: SimpleConfig,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    warped = cv2.warpPerspective(
        depth_base,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    preview = depth_rgb_overlay(rgb, warped, config)
    add_title(preview, "Vista previa calibrada")
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra empalme 2D depth -> RGB con puntos equivalentes.")
    parser.add_argument("--config", type=Path, default=Path("orbbec_sencillo_config.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rgb_points: list[tuple[float, float]] = []
    depth_points: list[tuple[float, float]] = []
    frozen_rgb: Optional[np.ndarray] = None
    frozen_depth: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal homography
        if event != cv2.EVENT_LBUTTONDOWN or frozen_rgb is None:
            return
        height, width = frozen_rgb.shape[:2]
        if y >= height:
            return
        if x < width:
            rgb_points.append((float(x), float(y)))
            print(f"RGB punto {len(rgb_points)}: ({x}, {y})")
        else:
            depth_points.append((float(x - width), float(y)))
            print(f"Depth punto {len(depth_points)}: ({x - width}, {y})")
        homography = None

    openni_initialized = False
    rgb_camera = None
    depth_stream = None
    try:
        rgb_camera = open_rgb_camera(config.rgb_index, config)
        if not rgb_camera.isOpened():
            raise RuntimeError(f"No se pudo abrir RGB en indice {config.rgb_index}.")

        openni2.initialize(config.openni_path)
        openni_initialized = True
        device = openni2.Device.open_any()
        depth_stream = device.create_depth_stream()
        depth_stream.start()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, on_mouse)

        print("Controles:")
        print("  C: capturar/congelar escena")
        print("  F: invertir/desinvertir depth y limpiar homografia anterior")
        print("  R: invertir/desinvertir RGB y limpiar homografia anterior")
        print("  clic normal en panel izquierdo: punto en RGB")
        print("  clic normal en panel derecho: punto correspondiente en depth")
        print("  Z: deshacer ultimo par incompleto/completo")
        print("  P: previsualizar homografia")
        print("  S: guardar homografia en config")
        print("  Q/ESC: salir")

        while True:
            if frozen_rgb is None or frozen_depth is None:
                rgb = read_rgb(rgb_camera, config)
                depth_base = read_depth(depth_stream, rgb.shape, config)
                depth_panel = make_depth_panel(depth_base, config)
                rgb_live = rgb.copy()
                depth_live = depth_panel.copy()
                add_title(rgb_live, f"RGB | {rgb_flip_text(config)}")
                add_title(depth_live, f"Depth | {depth_flip_text(config)}")
                draw_status(rgb_live, "R cambia espejo RGB")
                draw_status(depth_live, "F cambia espejo depth")
                live = np.hstack((rgb_live, depth_live))
                cv2.imshow(WINDOW, live)
            else:
                depth_panel = make_depth_panel(frozen_depth, config)
                view = draw_points(frozen_rgb, depth_panel, rgb_points, depth_points, config)
                if homography is not None:
                    preview = preview_overlay(frozen_rgb, frozen_depth, homography, config)
                    view = np.vstack((view, np.hstack((preview, preview))))
                cv2.imshow(WINDOW, view)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("f"), ord("F")):
                config.depth_flip_with_rgb = not config.depth_flip_with_rgb
                config.depth_to_rgb_homography = None
                frozen_rgb = None
                frozen_depth = None
                reset_calibration_points(rgb_points, depth_points)
                homography = None
                save_config(args.config, config)
                print(f"{depth_flip_text(config)}. Homografia anterior limpiada; vuelve a presionar C.")
            elif key in (ord("r"), ord("R")):
                config.rgb_flip_horizontal = not config.rgb_flip_horizontal
                config.depth_to_rgb_homography = None
                frozen_rgb = None
                frozen_depth = None
                reset_calibration_points(rgb_points, depth_points)
                homography = None
                save_config(args.config, config)
                print(f"{rgb_flip_text(config)}. Homografia anterior limpiada; vuelve a presionar C.")
            elif key in (ord("c"), ord("C")):
                frozen_rgb = read_rgb(rgb_camera, config)
                frozen_depth = read_depth(depth_stream, frozen_rgb.shape, config)
                reset_calibration_points(rgb_points, depth_points)
                homography = None
                print("Escena congelada. Marca minimo 4 pares de puntos.")
            elif key in (ord("z"), ord("Z")):
                if len(rgb_points) > len(depth_points) and rgb_points:
                    rgb_points.pop()
                elif depth_points:
                    depth_points.pop()
                elif rgb_points:
                    rgb_points.pop()
                homography = None
                print(f"Puntos: RGB={len(rgb_points)} depth={len(depth_points)}")
            elif key in (ord("p"), ord("P"), ord("s"), ord("S")):
                if frozen_rgb is None or frozen_depth is None:
                    print("Primero presiona C para congelar una escena.")
                    continue
                homography, inliers = compute_homography(rgb_points, depth_points)
                if homography is None:
                    print("Se necesitan al menos 4 pares completos RGB/depth.")
                    continue
                inlier_count = int(np.count_nonzero(inliers)) if inliers is not None else 0
                print(f"Homografia lista. Puntos usados: {len(rgb_points)} | inliers: {inlier_count}")
                if key in (ord("s"), ord("S")):
                    save_homography(args.config, config, homography)
                    print(f"Homografia guardada en {args.config}")
                    time.sleep(0.4)
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


if __name__ == "__main__":
    main()
