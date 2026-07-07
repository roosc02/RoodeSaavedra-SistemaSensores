from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


TRAFFIC_CLASSES = ["red octagonal traffic sign"]
DATASET_CLASS_NAME = "senalamiento_trafico"
CSV_FIELDS = [
    "trajectory_id",
    "trajectory_sample",
    "timestamp",
    "dataset_class_id",
    "dataset_class_name",
    "yolo_confidence",
    "track_id",
    "center_x_px",
    "center_y_px",
    "bbox_x_px",
    "bbox_y_px",
    "bbox_width_px",
    "bbox_height_px",
    "area_px",
    "depth_mm",
    "depth_source",
    "depth_detection_valid_pixels",
    "width_mm",
    "height_mm",
    "area_mm2",
    "rgb_fps",
    "depth_fps",
    "ir_fps",
    "depth_valid_pixels",
    "sensor_read_span_ms",
    "depth_scale_mm",
    "depth_to_rgb_mode",
]


def normalize_sensor_u8(image: np.ndarray, ignore_zero: bool = False) -> np.ndarray:
    values = image[image > 0] if ignore_zero else image.reshape(-1)
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image.astype(np.float32) - float(low)) * (255.0 / (high - low))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def add_evidence_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (16, 16, 16), -1)
    cv2.putText(
        image,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

TRAFFIC_KEYWORDS = (
    "traffic light",
    "traffic sign",
    "stop sign",
    "yield",
    "speed limit",
    "no entry",
    "crossing sign",
    "road sign",
    "warning sign",
    "mandatory sign",
    "prohibitory sign",
    "one way",
    "semaforo",
    "señal",
)


@dataclass
class YoloDetection:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area_px: float
    label: str
    class_id: int
    confidence: float
    track_id: Optional[int]


class YoloSingleObjectTracker:
    """YOLO + ByteTrack, limitado a un solo objetivo por trayectoria."""

    def __init__(
        self,
        model_path: str,
        target_class: str = "cualquiera",
        confidence: float = 0.45,
        iou: float = 0.50,
        device: str = "cpu",
        traffic_only: bool = True,
        traffic_classes: Optional[list[str]] = None,
        tracker_config: str = "bytetrack_stop.yaml",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Falta Ultralytics. Ejecuta: python -m pip install ultralytics"
            ) from error

        self.model_path = model_path
        self.target_class = target_class.strip().lower() or "cualquiera"
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self.traffic_only = bool(traffic_only)
        self.traffic_classes = traffic_classes or TRAFFIC_CLASSES
        self.tracker_config = tracker_config
        self.model = YOLO(model_path)
        self.open_vocabulary = self.traffic_only and "world" in model_path.lower()
        if self.open_vocabulary:
            self.model.set_classes(self.traffic_classes)
        self.names = self.model.names
        self.selected_track_id: Optional[int] = None
        self.previous_center: Optional[tuple[int, int]] = None
        self.missing_frames = 0
        self.switch_after_missing_frames = 12
        self.inference_ms = 0.0

    def reset_target(self) -> None:
        self.selected_track_id = None
        self.previous_center = None
        self.missing_frames = 0

    def _class_name(self, class_id: int) -> str:
        if self.traffic_only:
            return DATASET_CLASS_NAME
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        return str(self.names[class_id])

    def _matches_target(self, class_id: int, name: str) -> bool:
        if self.traffic_only:
            if self.open_vocabulary:
                return True
            normalized = name.strip().lower().replace("_", " ").replace("-", " ")
            return any(keyword in normalized for keyword in TRAFFIC_KEYWORDS)
        target = self.target_class
        if target in ("", "cualquiera", "any", "*"):
            return True
        if target.isdigit() and int(target) == class_id:
            return True
        return name.lower() == target

    def update(self, frame_bgr: np.ndarray) -> Optional[YoloDetection]:
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None
        result = results[0]
        self.inference_ms = float(result.speed.get("inference", 0.0))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if self.selected_track_id is not None:
                self.missing_frames += 1
                if self.missing_frames > self.switch_after_missing_frames:
                    self.reset_target()
            return None

        xyxy = boxes.xyxy.int().cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(classes)
        )
        height, width = frame_bgr.shape[:2]
        candidates: list[YoloDetection] = []
        for coords, class_id, confidence, track_id in zip(
            xyxy, classes, confidences, track_ids
        ):
            name = self._class_name(int(class_id))
            if not self._matches_target(int(class_id), name):
                continue
            x1, y1, x2, y2 = [int(value) for value in coords]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
            candidates.append(
                YoloDetection(
                    bbox=(x1, y1, box_width, box_height),
                    center=(x1 + box_width // 2, y1 + box_height // 2),
                    area_px=float(box_width * box_height),
                    label=name,
                    class_id=int(class_id),
                    confidence=float(confidence),
                    track_id=int(track_id) if track_id is not None else None,
                )
            )

        if not candidates:
            if self.selected_track_id is not None:
                self.missing_frames += 1
                if self.missing_frames > self.switch_after_missing_frames:
                    self.reset_target()
            return None

        same_track = [
            item
            for item in candidates
            if self.selected_track_id is not None
            and item.track_id == self.selected_track_id
        ]
        if same_track:
            selected = max(same_track, key=lambda item: item.confidence)
            self.missing_frames = 0
        elif self.selected_track_id is not None:
            self.missing_frames += 1
            if self.missing_frames <= self.switch_after_missing_frames:
                return None
            self.reset_target()
            selected = max(
                candidates,
                key=lambda item: item.confidence * np.sqrt(item.area_px),
            )
        elif self.previous_center is not None:
            selected = min(
                candidates,
                key=lambda item: np.hypot(
                    item.center[0] - self.previous_center[0],
                    item.center[1] - self.previous_center[1],
                )
                - 40.0 * item.confidence,
            )
        else:
            selected = max(
                candidates,
                key=lambda item: item.confidence * np.sqrt(item.area_px),
            )

        self.selected_track_id = selected.track_id
        self.previous_center = selected.center
        self.missing_frames = 0
        return selected


class TrajectoryDatasetRecorder:
    """Guarda N muestras RGB-D-IR coordinadas para una trayectoria."""

    def __init__(
        self,
        root: Path,
        samples_per_trajectory: int = 10,
        interval_seconds: float = 0.30,
    ) -> None:
        self.root = root
        self.trajectories_root = self.root / "trayectorias"
        self.reports_root = self.root / "reportes"
        self.trajectories_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.reports_root / "datos_sensores.csv"
        self.dataset_class_name = DATASET_CLASS_NAME
        (self.root / "clase.txt").write_text(
            "0: senalamiento_trafico\n",
            encoding="utf-8",
        )
        self._ensure_csv_header()
        self.samples_per_trajectory = max(10, int(samples_per_trajectory))
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.active = False
        self.trajectory_id = ""
        self.directory: Optional[Path] = None
        self.sample_count = 0
        self.track_id: Optional[int] = None
        self.last_capture_time = 0.0
        self.target_label = ""
        self.last_completed_directory: Optional[Path] = None
        self.track_points: list[tuple[int, int]] = []

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists():
            with self.csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.reader(csv_file)
                current_header = next(reader, [])
            if current_header == CSV_FIELDS:
                return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            legacy_path = self.csv_path.with_name(
                f"{self.csv_path.stem}_legacy_{stamp}{self.csv_path.suffix}"
            )
            self.csv_path.replace(legacy_path)

        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_FIELDS).writeheader()

    @property
    def status(self) -> str:
        if self.active:
            return f"GRABANDO {self.sample_count}/{self.samples_per_trajectory}"
        return "LISTO"

    def start(self, target_label: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.trajectory_id = f"trajectory_{stamp}"
        self.directory = self.trajectories_root / self.trajectory_id
        for folder in (
            "images",
            "labels",
            "depth_raw",
            "depth_aligned",
            "ir",
            "raw",
            "metadata",
            "evidencias_rgb",
            "depth_visual",
            "ir_visual",
            "evidencias_sensores",
        ):
            (self.directory / folder).mkdir(parents=True, exist_ok=True)
        self.active = True
        self.sample_count = 0
        self.track_id = None
        self.last_capture_time = 0.0
        self.target_label = self.dataset_class_name
        self.track_points = []
        yaml_label = self.dataset_class_name
        (self.directory / "data.yaml").write_text(
            f'path: "{self.directory.resolve().as_posix()}"\n'
            "train: images\n"
            "val: images\n"
            "names:\n"
            f'  0: "{yaml_label}"\n',
            encoding="utf-8",
        )
        return self.directory

    def _append_csv(self, metadata: dict[str, Any]) -> None:
        self._ensure_csv_header()
        center = metadata.get("center_rgb") or (None, None)
        bbox = metadata.get("bbox_rgb") or (None, None, None, None)
        fps = metadata.get("fps") or {}
        row = {
            "trajectory_id": metadata.get("trajectory_id"),
            "trajectory_sample": metadata.get("trajectory_sample"),
            "timestamp": metadata.get("timestamp"),
            "dataset_class_id": 0,
            "dataset_class_name": self.dataset_class_name,
            "yolo_confidence": metadata.get("yolo_confidence"),
            "track_id": metadata.get("track_id"),
            "center_x_px": center[0],
            "center_y_px": center[1],
            "bbox_x_px": bbox[0],
            "bbox_y_px": bbox[1],
            "bbox_width_px": bbox[2],
            "bbox_height_px": bbox[3],
            "area_px": metadata.get("area_px"),
            "depth_mm": metadata.get("depth_mm"),
            "depth_source": metadata.get("depth_source"),
            "depth_detection_valid_pixels": metadata.get("depth_detection_valid_pixels"),
            "width_mm": metadata.get("width_mm"),
            "height_mm": metadata.get("height_mm"),
            "area_mm2": metadata.get("area_mm2"),
            "rgb_fps": fps.get("rgb"),
            "depth_fps": fps.get("depth"),
            "ir_fps": fps.get("ir"),
            "depth_valid_pixels": metadata.get("depth_valid_pixels"),
            "sensor_read_span_ms": metadata.get("sensor_read_span_ms"),
            "depth_scale_mm": metadata.get("depth_scale_mm"),
            "depth_to_rgb_mode": metadata.get("depth_to_rgb_mode"),
        }
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def cancel(self) -> None:
        self.active = False

    def ready_for(self, track_id: Optional[int]) -> bool:
        if not self.active:
            return False
        if self.track_id is None and track_id is not None:
            self.track_id = track_id
        if self.track_id is not None and track_id != self.track_id:
            return False
        return time.perf_counter() - self.last_capture_time >= self.interval_seconds

    def save_sample(
        self,
        rgb: np.ndarray,
        depth_raw: np.ndarray,
        depth_aligned: np.ndarray,
        ir_raw: np.ndarray,
        detection: Any,
        metadata: dict[str, Any],
    ) -> Path:
        if not self.active or self.directory is None:
            raise RuntimeError("No hay una trayectoria activa.")

        index = self.sample_count
        stem = f"frame_{index:03d}"
        height, width = rgb.shape[:2]
        x, y, box_width, box_height = detection.bbox
        center_x = (x + box_width / 2.0) / width
        center_y = (y + box_height / 2.0) / height
        normalized_width = box_width / width
        normalized_height = box_height / height

        cv2.imwrite(str(self.directory / "images" / f"{stem}.jpg"), rgb)
        cv2.imwrite(str(self.directory / "depth_raw" / f"{stem}.png"), depth_raw)
        cv2.imwrite(
            str(self.directory / "depth_aligned" / f"{stem}.png"), depth_aligned
        )
        cv2.imwrite(str(self.directory / "ir" / f"{stem}.png"), ir_raw)
        (self.directory / "labels" / f"{stem}.txt").write_text(
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}\n",
            encoding="utf-8",
        )

        sample_metadata = {
            **metadata,
            "trajectory_id": self.trajectory_id,
            "trajectory_sample": index,
            "dataset_class_id": 0,
            "dataset_class_name": self.dataset_class_name,
            "yolo_source_class_id": detection.class_id,
            "yolo_source_label": detection.label,
            "yolo_confidence": detection.confidence,
            "track_id": detection.track_id,
            "bbox_rgb": detection.bbox,
            "center_rgb": detection.center,
            "area_px": detection.area_px,
            "depth_mm": detection.depth_mm,
            "depth_source": getattr(detection, "depth_source", None),
            "depth_detection_valid_pixels": getattr(detection, "depth_valid_pixels", 0),
            "width_mm": detection.width_mm,
            "height_mm": detection.height_mm,
            "area_mm2": detection.area_mm2,
            "pseudo_label": True,
        }
        self.track_points.append(tuple(detection.center))
        sample_metadata["trajectory_points_rgb"] = self.track_points.copy()

        annotated = rgb.copy()
        if len(self.track_points) >= 2:
            cv2.polylines(
                annotated,
                [np.asarray(self.track_points, dtype=np.int32)],
                False,
                (255, 80, 20),
                3,
                cv2.LINE_AA,
            )
        x, y, box_width, box_height = detection.bbox
        cv2.rectangle(
            annotated,
            (x, y),
            (x + box_width, y + box_height),
            (0, 220, 255),
            2,
        )
        depth_text = (
            f"{detection.depth_mm:.0f}mm"
            if detection.depth_mm is not None
            else "sin profundidad"
        )
        depth_source = getattr(detection, "depth_source", None)
        depth_detail = (
            f"{depth_text} ({depth_source})"
            if depth_source
            else depth_text
        )
        evidence_label = (
            f"{self.dataset_class_name} | {detection.confidence:.1%} | "
            f"ID {detection.track_id} | {depth_text}"
        )
        cv2.putText(
            annotated,
            evidence_label,
            (x, max(24, y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        panel_width = annotated.shape[1]
        data_panel = np.full(
            (annotated.shape[0], panel_width, 3),
            (24, 24, 24),
            dtype=np.uint8,
        )
        fps = sample_metadata.get("fps") or {}
        area_cm2 = (
            detection.area_mm2 / 100.0
            if detection.area_mm2 is not None
            else None
        )
        dimensions = (
            f"{detection.width_mm:.1f} x {detection.height_mm:.1f} mm"
            if detection.width_mm is not None and detection.height_mm is not None
            else "sin datos"
        )
        evidence_lines = [
            ("EVIDENCIA DE TRAYECTORIA", (90, 220, 255)),
            (
                f"Muestra: {index + 1}/{self.samples_per_trajectory}",
                (235, 235, 235),
            ),
            (f"Clase: {self.dataset_class_name}", (120, 230, 170)),
            (f"Confianza YOLO: {detection.confidence:.2%}", (235, 235, 235)),
            (f"Track ID: {detection.track_id}", (235, 235, 235)),
            (f"Centro RGB: {detection.center}", (235, 235, 235)),
            (f"BBox RGB: {detection.bbox}", (235, 235, 235)),
            (f"Distancia: {depth_detail}", (255, 210, 90)),
            (f"Ancho x alto: {dimensions}", (235, 235, 235)),
            (f"Area imagen: {detection.area_px:.0f} px2", (235, 235, 235)),
            (
                f"Area estimada: {area_cm2:.2f} cm2"
                if area_cm2 is not None
                else "Area estimada: sin datos",
                (235, 235, 235),
            ),
            (
                f"FPS R/D/IR: {fps.get('rgb', 0):.1f} / "
                f"{fps.get('depth', 0):.1f} / {fps.get('ir', 0):.1f}",
                (235, 235, 235),
            ),
            (
                f"Prof. validos: {sample_metadata.get('depth_valid_pixels', 0)} px",
                (235, 235, 235),
            ),
            (
                f"Prof. objetivo: {sample_metadata.get('depth_detection_valid_pixels', 0)} px",
                (235, 235, 235),
            ),
            (
                f"Desfase sensores: {sample_metadata.get('sensor_read_span_ms', 0):.2f} ms",
                (235, 235, 235),
            ),
            (
                f"Alineacion: {sample_metadata.get('depth_to_rgb_mode', 'sin datos')}",
                (235, 235, 235),
            ),
            (f"Fecha: {str(sample_metadata.get('timestamp', ''))[:19]}", (170, 170, 170)),
        ]
        y_text = 31
        for line_index, (line, color) in enumerate(evidence_lines):
            cv2.putText(
                data_panel,
                line,
                (16, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54 if line_index == 0 else 0.44,
                color,
                2 if line_index == 0 else 1,
                cv2.LINE_AA,
            )
            y_text += 29
        evidence_image = np.hstack((annotated, data_panel))
        cv2.imwrite(
            str(self.directory / "evidencias_rgb" / f"{stem}.jpg"),
            evidence_image,
        )

        depth_u8 = normalize_sensor_u8(depth_aligned, ignore_zero=True)
        depth_visual = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_visual = cv2.resize(
            depth_visual,
            (annotated.shape[1], annotated.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.rectangle(
            depth_visual,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 255),
            2,
        )
        add_evidence_title(depth_visual, f"PROFUNDIDAD ALINEADA | {depth_text}")
        cv2.imwrite(
            str(self.directory / "depth_visual" / f"{stem}.jpg"),
            depth_visual,
        )

        ir_u8 = normalize_sensor_u8(ir_raw)
        ir_visual = cv2.applyColorMap(ir_u8, cv2.COLORMAP_BONE)
        ir_visual = cv2.resize(
            ir_visual,
            (annotated.shape[1], annotated.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        ix, iy, iw, ih = x, y, box_width, box_height
        cv2.rectangle(ir_visual, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
        add_evidence_title(ir_visual, "INFRARROJO NORMALIZADO | ROI RGB")
        cv2.imwrite(
            str(self.directory / "ir_visual" / f"{stem}.jpg"),
            ir_visual,
        )

        annotated_titled = annotated.copy()
        add_evidence_title(annotated_titled, "RGB | YOLO | TRAYECTORIA")
        sensor_evidence = np.vstack(
            (
                np.hstack((annotated_titled, depth_visual)),
                np.hstack((ir_visual, data_panel)),
            )
        )
        cv2.imwrite(
            str(self.directory / "evidencias_sensores" / f"{stem}.jpg"),
            sensor_evidence,
        )
        metadata_path = self.directory / "metadata" / f"{stem}.json"
        metadata_path.write_text(
            json.dumps(sample_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.savez_compressed(
            self.directory / "raw" / f"{stem}.npz",
            rgb_bgr=rgb,
            depth_raw=depth_raw,
            depth_aligned_to_rgb=depth_aligned,
            ir_raw=ir_raw,
            metadata_json=json.dumps(sample_metadata, ensure_ascii=False),
        )
        with (self.directory / "trajectory.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(sample_metadata, ensure_ascii=False) + "\n")
        self._append_csv(sample_metadata)

        self.sample_count += 1
        self.last_capture_time = time.perf_counter()
        if self.sample_count >= self.samples_per_trajectory:
            self.active = False
            self.last_completed_directory = self.directory
            summary = {
                "trajectory_id": self.trajectory_id,
                "target_label": self.target_label,
                "samples": self.sample_count,
                "track_id": self.track_id,
                "completed": True,
                "labels_are_pseudo_labels": True,
                "trajectory_points_rgb": self.track_points,
            }
            (self.directory / "trajectory_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            cv2.imwrite(
                str(self.directory / "trayectoria_final_rgb.jpg"),
                evidence_image,
            )
            cv2.imwrite(
                str(self.directory / "trayectoria_final_sensores.jpg"),
                sensor_evidence,
            )
        return metadata_path
