from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_DIR = Path("calibration_data")


def load_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            return {}
        raw = data["metadata_json"]
        if hasattr(raw, "item"):
            raw = raw.item()
        return json.loads(str(raw))


def keep_row(
    row: dict[str, Any],
    only_marked: bool,
    require_detection: bool,
    min_area_px: float,
    require_depth: bool,
    label: str | None,
    min_depth_mm: float | None,
    max_depth_mm: float | None,
    min_area_cm2: float | None,
    max_area_cm2: float | None,
) -> bool:
    if only_marked and not row["capture_marked_for_calibration"]:
        return False
    if require_detection and row["detection_area_px"] is None:
        return False
    if label and row["label"] != label:
        return False
    if row["detection_area_px"] is not None and row["detection_area_px"] < min_area_px:
        return False
    if require_depth and row["detection_depth_mm"] is None:
        return False
    if min_depth_mm is not None and (
        row["detection_depth_mm"] is None or row["detection_depth_mm"] < min_depth_mm
    ):
        return False
    if max_depth_mm is not None and (
        row["detection_depth_mm"] is None or row["detection_depth_mm"] > max_depth_mm
    ):
        return False
    if min_area_cm2 is not None and (
        row["detection_area_cm2"] is None or row["detection_area_cm2"] < min_area_cm2
    ):
        return False
    if max_area_cm2 is not None and (
        row["detection_area_cm2"] is None or row["detection_area_cm2"] > max_area_cm2
    ):
        return False
    return True


def format_optional(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revisa capturas NPZ con metadata de deteccion de objeto."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-area-px", type=float, default=800.0)
    parser.add_argument(
        "--label",
        default=None,
        help="filtra por etiqueta detectada, por ejemplo rectangulo, cuadrado, rojo",
    )
    parser.add_argument("--min-depth-mm", type=float, default=None)
    parser.add_argument("--max-depth-mm", type=float, default=None)
    parser.add_argument("--min-area-cm2", type=float, default=None)
    parser.add_argument("--max-area-cm2", type=float, default=None)
    parser.add_argument("--all", action="store_true", help="incluye no marcadas")
    parser.add_argument(
        "--allow-no-detection",
        action="store_true",
        help="incluye capturas sin objeto detectado",
    )
    parser.add_argument(
        "--allow-no-depth",
        action="store_true",
        help="incluye capturas sin distancia del objetivo",
    )
    args = parser.parse_args()

    output = args.output or (args.data_dir / "object_captures_review.csv")
    rows: list[dict[str, Any]] = []
    skipped = 0

    for path in sorted(args.data_dir.glob("*.npz")):
        metadata = load_metadata(path)
        row = {
            "file": str(path),
            "timestamp": metadata.get("timestamp", ""),
            "capture_marked_for_calibration": bool(
                metadata.get("capture_marked_for_calibration", False)
            ),
            "label": metadata.get("detection_label"),
            "detection_area_px": metadata.get("detection_area_px"),
            "detection_area_mm2": metadata.get("detection_area_mm2"),
            "detection_area_cm2": metadata.get("detection_area_cm2"),
            "detection_depth_mm": metadata.get("detection_depth_mm"),
            "detection_width_mm": metadata.get("detection_width_mm"),
            "detection_height_mm": metadata.get("detection_height_mm"),
            "detection_center_rgb": metadata.get("detection_center_rgb"),
            "detection_bbox_rgb": metadata.get("detection_bbox_rgb"),
            "depth_to_rgb_mode": metadata.get("depth_to_rgb_mode", ""),
            "depth_to_rgb_offset_x": metadata.get("depth_to_rgb_offset_x", ""),
            "depth_to_rgb_offset_y": metadata.get("depth_to_rgb_offset_y", ""),
            "depth_to_rgb_scale": metadata.get("depth_to_rgb_scale", ""),
        }
        if keep_row(
            row,
            only_marked=not args.all,
            require_detection=not args.allow_no_detection,
            min_area_px=args.min_area_px,
            require_depth=not args.allow_no_depth,
            label=args.label,
            min_depth_mm=args.min_depth_mm,
            max_depth_mm=args.max_depth_mm,
            min_area_cm2=args.min_area_cm2,
            max_area_cm2=args.max_area_cm2,
        ):
            rows.append(row)
        else:
            skipped += 1

    rows.sort(
        key=lambda item: (
            item["detection_area_px"] is None,
            -(item["detection_area_px"] or 0),
        )
    )

    fieldnames = [
        "file",
        "timestamp",
        "capture_marked_for_calibration",
        "label",
        "detection_area_px",
        "detection_area_mm2",
        "detection_area_cm2",
        "detection_depth_mm",
        "detection_width_mm",
        "detection_height_mm",
        "detection_center_rgb",
        "detection_bbox_rgb",
        "depth_to_rgb_mode",
        "depth_to_rgb_offset_x",
        "depth_to_rgb_offset_y",
        "depth_to_rgb_scale",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- Revision de capturas ---")
    print(f"Carpeta: {args.data_dir}")
    print(f"Capturas utiles: {len(rows)}")
    print(f"Capturas omitidas: {skipped}")
    print(f"CSV: {output}")
    if args.label:
        print(f"Etiqueta filtrada: {args.label}")
    if rows:
        areas = [row["detection_area_px"] for row in rows if row["detection_area_px"] is not None]
        depths = [
            row["detection_depth_mm"]
            for row in rows
            if row["detection_depth_mm"] is not None
        ]
        print(f"Area px min/max: {min(areas):.0f} / {max(areas):.0f}")
        if depths:
            print(f"Distancia mm min/max: {min(depths):.1f} / {max(depths):.1f}")
        real_areas = [
            row["detection_area_cm2"]
            for row in rows
            if row["detection_area_cm2"] is not None
        ]
        if real_areas:
            print(f"Area cm2 min/max: {min(real_areas):.2f} / {max(real_areas):.2f}")
        print("\nTop 5 por area:")
        for row in rows[:5]:
            print(
                f"- {Path(row['file']).name}: "
                f"area={format_optional(row['detection_area_px'], 0)} px2, "
                f"dist={format_optional(row['detection_depth_mm'], 1)} mm, "
                f"area={format_optional(row['detection_area_cm2'], 2)} cm2"
            )
    else:
        print(
            "\nSin capturas utiles con esos filtros. Prueba bajar --min-area-px "
            "o revisar que el modo de deteccion sea color/forma correcto."
        )


if __name__ == "__main__":
    main()
