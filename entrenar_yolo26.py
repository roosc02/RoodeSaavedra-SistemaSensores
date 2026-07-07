# -*- coding: utf-8 -*-
"""
Entrena un modelo YOLO26 (Ultralytics) para detectar una sola clase,
"senalamiento_trafico", usando el dataset generado por
preparar_dataset_yolo26.py.

Requisito (una sola vez):
    pip install -U ultralytics

Uso basico (usa yolo26n, el mas rapido, bueno para tiempo real / CPU):
    python entrenar_yolo26.py

Con mas control (ej. modelo mas grande y GPU explicita):
    python entrenar_yolo26.py --modelo yolo26s.pt --epocas 250 --dispositivo 0

La primera vez que corras esto, Ultralytics descargara automaticamente
los pesos preentrenados de YOLO26 (necesitas internet esa vez).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def detectar_dispositivo() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dataset_yolo26/data.yaml"))
    parser.add_argument(
        "--modelo", default="yolo26n.pt",
        help="yolo26n.pt (nano, mas rapido) | yolo26s.pt | yolo26m.pt | yolo26l.pt | yolo26x.pt",
    )
    parser.add_argument("--epocas", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Usa -1 para autoajuste (solo con GPU)")
    parser.add_argument("--paciencia", type=int, default=40, help="Epocas sin mejora antes de detener")
    parser.add_argument("--dispositivo", default=None, help="'0' para GPU, 'cpu' para forzar CPU")
    parser.add_argument("--proyecto", default="runs_yolo26")
    parser.add_argument("--nombre", default="senalamiento_trafico")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"No se encontro {args.data}. Corre primero preparar_dataset_yolo26.py."
        )

    device = args.dispositivo or detectar_dispositivo()
    print(f"Modelo base: {args.modelo}")
    print(f"Dataset: {args.data}")
    print(f"Dispositivo: {device}")

    modelo = YOLO(args.modelo)

    modelo.train(
        data=str(args.data),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.paciencia,
        project=args.proyecto,
        name=args.nombre,
        pretrained=True,
        single_cls=True,   # red de seguridad: colapsa todo a la clase 0
        plots=True,
    )

    pesos_finales = Path(args.proyecto) / args.nombre / "weights" / "best.pt"
    print("\nEntrenamiento terminado.")
    print(f"Pesos entrenados: {pesos_finales.resolve()}")
    print(
        "\nPara usar este modelo en tu app en vivo (orbbec_vision.py), en "
        "camera_config.json coloca algo asi:\n"
        f'  "yolo_model": "{pesos_finales.as_posix()}",\n'
        '  "yolo_target_class": "senalamiento_trafico",\n'
        '  "yolo_traffic_only": true,\n'
        '  "yolo_confidence": 0.25\n'
        "\n"
        "El 0.25 es un punto de partida razonable para un modelo YA entrenado "
        "(no zero-shot); ajustalo viendo cuantos falsos positivos/negativos "
        "te da en tus pruebas. Necesitas la version parcheada de "
        "yolo_tracking.py y orbbec_vision.py para que el filtro de deteccion "
        "acepte este modelo propio de una sola clase (ver mensaje del chat)."
    )


if __name__ == "__main__":
    main()
