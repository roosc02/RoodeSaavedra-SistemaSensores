from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DATA_DIR = Path("calibration_data")
DEFAULT_CONFIG_PATH = Path("camera_config.json")


@dataclass
class FrameResult:
    path: Path
    corners: np.ndarray
    object_points: np.ndarray
    image_size: tuple[int, int]
    depth_values: np.ndarray
    valid_depth_ratio: float


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def checkerboard_object_points(
    pattern_size: tuple[int, int],
    square_mm: float,
) -> np.ndarray:
    cols, rows = pattern_size
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= float(square_mm)
    return grid


def find_checkerboard_corners(
    rgb_bgr: np.ndarray,
    pattern_size: tuple[int, int],
) -> np.ndarray | None:
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if ok:
            return corners.astype(np.float32)

    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK,
    )
    if not ok:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def sample_depth_at_corners(
    depth_aligned: np.ndarray,
    corners: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, float]:
    values: list[float] = []
    height, width = depth_aligned.shape[:2]
    for corner in corners.reshape(-1, 2):
        x, y = np.rint(corner).astype(int)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        roi = depth_aligned[y0:y1, x0:x1]
        valid = roi[roi > 0]
        values.append(float(np.median(valid)) if valid.size else np.nan)
    depth_values = np.asarray(values, dtype=np.float32)
    valid_ratio = float(np.count_nonzero(np.isfinite(depth_values)) / len(depth_values))
    return depth_values, valid_ratio


def load_calibration_frames(
    data_dir: Path,
    pattern_size: tuple[int, int],
    square_mm: float,
    depth_radius: int,
) -> list[FrameResult]:
    object_points = checkerboard_object_points(pattern_size, square_mm)
    frames: list[FrameResult] = []
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "rgb_bgr" not in data:
                continue
            rgb = data["rgb_bgr"]
            if "depth_aligned_to_rgb" in data:
                depth = data["depth_aligned_to_rgb"]
            elif "depth_raw" in data:
                depth = cv2.resize(
                    data["depth_raw"],
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                continue

        corners = find_checkerboard_corners(rgb, pattern_size)
        if corners is None:
            print(f"[omitida] {path.name}: no se detecto el tablero")
            continue

        depth_values, valid_depth_ratio = sample_depth_at_corners(
            depth,
            corners,
            depth_radius,
        )
        frames.append(
            FrameResult(
                path=path,
                corners=corners,
                object_points=object_points.copy(),
                image_size=(rgb.shape[1], rgb.shape[0]),
                depth_values=depth_values,
                valid_depth_ratio=valid_depth_ratio,
            )
        )
        print(
            f"[ok] {path.name}: esquinas={len(corners)} "
            f"profundidad_valida={valid_depth_ratio * 100:.1f}%"
        )
    return frames


def calibrate_rgb(frames: list[FrameResult]) -> tuple[float, np.ndarray, np.ndarray, Any, Any]:
    image_size = frames[0].image_size
    object_points = [frame.object_points for frame in frames]
    image_points = [frame.corners for frame in frames]
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return float(rms), camera_matrix, dist_coeffs, rvecs, tvecs


def reprojection_error(
    frames: list[FrameResult],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: Any,
    tvecs: Any,
) -> float:
    errors: list[float] = []
    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            frame.object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        diff = projected.reshape(-1, 2) - frame.corners.reshape(-1, 2)
        errors.extend(np.linalg.norm(diff, axis=1).tolist())
    return float(np.mean(errors)) if errors else float("nan")


def estimate_depth_scale_correction(
    frames: list[FrameResult],
    rvecs: Any,
    tvecs: Any,
    current_depth_scale_mm: float,
) -> tuple[float | None, float | None, int]:
    ratios: list[float] = []
    errors_mm: list[float] = []
    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(rvec)
        expected = (rotation @ frame.object_points.T + tvec).T[:, 2]
        measured = frame.depth_values.astype(np.float32) * float(current_depth_scale_mm)
        valid = np.isfinite(measured) & (measured > 0) & (expected > 0)
        if not np.any(valid):
            continue
        ratios.extend((expected[valid] / measured[valid]).tolist())
        errors_mm.extend((measured[valid] - expected[valid]).tolist())

    if not ratios:
        return None, None, 0
    return float(np.median(ratios)), float(np.median(errors_mm)), len(ratios)


def draw_debug_images(
    frames: list[FrameResult],
    output_dir: Path,
    pattern_size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        with np.load(frame.path, allow_pickle=False) as data:
            rgb = data["rgb_bgr"].copy()
        cv2.drawChessboardCorners(rgb, pattern_size, frame.corners, True)
        cv2.imwrite(str(output_dir / f"{frame.path.stem}_corners.png"), rgb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibra RGB y verifica profundidad alineada con capturas NPZ."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cols", type=int, default=9, help="esquinas internas por fila")
    parser.add_argument("--rows", type=int, default=6, help="esquinas internas por columna")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--apply", action="store_true", help="actualiza camera_config.json")
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)
    config = load_config(args.config)
    frames = load_calibration_frames(
        args.data_dir,
        pattern_size,
        args.square_mm,
        args.depth_radius,
    )

    if len(frames) < args.min_frames:
        raise SystemExit(
            f"Se requieren al menos {args.min_frames} capturas validas; "
            f"solo hay {len(frames)}."
        )

    image_sizes = {frame.image_size for frame in frames}
    if len(image_sizes) != 1:
        raise SystemExit(f"Las capturas tienen resoluciones distintas: {sorted(image_sizes)}")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_rgb(frames)
    mean_error = reprojection_error(frames, camera_matrix, dist_coeffs, rvecs, tvecs)
    current_depth_scale = float(config.get("depth_scale_mm", 1.0))
    scale_ratio, median_depth_error, depth_samples = estimate_depth_scale_correction(
        frames,
        rvecs,
        tvecs,
        current_depth_scale,
    )

    report = {
        "pattern_cols": args.cols,
        "pattern_rows": args.rows,
        "square_mm": args.square_mm,
        "frames_used": len(frames),
        "rgb_rms_px": rms,
        "rgb_mean_reprojection_error_px": mean_error,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "depth_samples_used": depth_samples,
        "depth_scale_current_mm": current_depth_scale,
        "depth_scale_suggested_mm": (
            current_depth_scale * scale_ratio if scale_ratio is not None else None
        ),
        "depth_median_error_mm_before_scale_update": median_depth_error,
        "used_files": [str(frame.path) for frame in frames],
    }

    output_path = args.data_dir / "rgb_depth_calibration_report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_debug_images(frames, args.data_dir / "debug_corners", pattern_size)

    print("\n--- Resultado RGB ---")
    print(f"Capturas usadas: {len(frames)}")
    print(f"RMS calibracion: {rms:.4f} px")
    print(f"Error medio reproyeccion: {mean_error:.4f} px")
    print("Matriz RGB:")
    print(camera_matrix)
    print(f"Distorsion RGB: {dist_coeffs.reshape(-1)}")

    print("\n--- Chequeo profundidad alineada ---")
    print(f"Muestras validas: {depth_samples}")
    if scale_ratio is None:
        print("No hubo profundidad suficiente en las esquinas del tablero.")
    else:
        suggested = current_depth_scale * scale_ratio
        print(f"Error mediano actual: {median_depth_error:.2f} mm")
        print(f"depth_scale_mm actual: {current_depth_scale:.6f}")
        print(f"depth_scale_mm sugerido: {suggested:.6f}")

    print(f"\nReporte: {output_path}")
    print(f"Imagenes debug: {args.data_dir / 'debug_corners'}")

    if args.apply:
        config["fx"] = float(camera_matrix[0, 0])
        config["fy"] = float(camera_matrix[1, 1])
        config["cx"] = float(camera_matrix[0, 2])
        config["cy"] = float(camera_matrix[1, 2])
        config["rgb_dist_coeffs"] = dist_coeffs.reshape(-1).astype(float).tolist()
        if scale_ratio is not None and 0.80 <= scale_ratio <= 1.20:
            config["depth_scale_mm"] = float(current_depth_scale * scale_ratio)
        save_config(args.config, config)
        print(f"\nConfiguracion actualizada: {args.config}")
    else:
        print("\nNo se modifico camera_config.json. Usa --apply para aplicar resultados.")


if __name__ == "__main__":
    main()
