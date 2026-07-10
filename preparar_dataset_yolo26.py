# -*- coding: utf-8 -*-
"""
Consolida las trayectorias capturadas por orbbec_vision.py (via
TrajectoryDatasetRecorder) en un dataset YOLO listo para entrenar,
con una sola clase: "senalamiento_trafico".

ENTRADA esperada (lo que ya genera tu pipeline):
    evidencias_pruebas/senalamiento_trafico/
        trayectorias/
            trajectory_20260101_120000_000001/
                images/frame_000.jpg ...
                labels/frame_000.txt ...   (formato YOLO: "0 cx cy w h")
            trajectory_20260101_120500_000002/
                ...

SALIDA:
    dataset_yolo26/
        images/train/*.jpg   images/val/*.jpg
        labels/train/*.txt   labels/val/*.txt
        data.yaml

El split train/val se hace POR TRAYECTORIA (no por frame individual),
porque los frames de una misma trayectoria son casi identicos entre si
(se capturan cada ~0.3s). Mezclar frames de la misma trayectoria entre
train y val inflaria artificialmente las metricas de validacion.

Uso basico:
    python preparar_dataset_yolo26.py

Con opciones:
    python preparar_dataset_yolo26.py --origen evidencias_pruebas/senalamiento_trafico ^
        --destino dataset_yolo26 --val-ratio 0.15 --negativos captures

--negativos apunta a una carpeta con imagenes SIN el objetivo (fondo,
calle vacia, etc.). Es opcional pero muy recomendable: tu dataset actual
solo tiene ejemplos POSITIVOS (el recorder solo guarda frames cuando hay
una deteccion activa), y sin negativos el modelo aprende mas facil a
confundir formas u objetos rojos parecidos con la señal.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAME = "senalamiento_trafico"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_trajectories(origen: Path) -> list[Path]:
    trayectorias_dir = origen / "trayectorias"
    if not trayectorias_dir.exists():
        raise SystemExit(
            f"No se encontro la carpeta 'trayectorias' dentro de {origen}. "
            "Revisa la ruta con --origen."
        )
    trayectorias = sorted(p for p in trayectorias_dir.iterdir() if p.is_dir())
    if not trayectorias:
        raise SystemExit(f"No hay carpetas de trayectoria dentro de {trayectorias_dir}.")
    return trayectorias


def collect_pairs(trajectory_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = trajectory_dir / "images"
    labels_dir = trajectory_dir / "labels"
    pairs: list[tuple[Path, Path]] = []
    if not images_dir.exists():
        print(f"  [AVISO] {trajectory_dir.name} no tiene carpeta 'images', se omite.")
        return pairs
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"  [AVISO] {image_path.name}: falta el label, se omite.")
            continue
        if label_path.stat().st_size == 0:
            print(f"  [AVISO] {label_path.name}: label vacio, se omite.")
            continue
        pairs.append((image_path, label_path))
    return pairs


def copy_pairs(pairs: list[tuple[Path, Path]], split: str, destino: Path, prefix: str) -> None:
    images_out = destino / "images" / split
    labels_out = destino / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        stem = f"{prefix}_{image_path.stem}"
        shutil.copy2(image_path, images_out / f"{stem}.jpg")
        shutil.copy2(label_path, labels_out / f"{stem}.txt")


def add_negatives(negativos_dir: Path, destino: Path, split: str) -> int:
    """Copia imagenes sin objeto (fondo) con un label vacio (background)."""
    images_out = destino / "images" / split
    labels_out = destino / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(negativos_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = f"negativo_{image_path.stem}"
        shutil.copy2(image_path, images_out / f"{stem}{image_path.suffix.lower()}")
        (labels_out / f"{stem}.txt").write_text("", encoding="utf-8")
        count += 1
    return count


def split_trajectories(trayectorias: list[Path], val_ratio: float, seed: int):
    trayectorias = list(trayectorias)
    random.Random(seed).shuffle(trayectorias)

    if len(trayectorias) < 2:
        print(
            "[AVISO] Solo hay 1 trayectoria disponible: todo se usara como "
            "entrenamiento y no habra validacion real. Graba mas trayectorias "
            "(distintos angulos/distancias/fondos) antes de entrenar en serio."
        )
        return trayectorias, []

    n_val = max(1, round(len(trayectorias) * val_ratio))
    n_val = min(n_val, len(trayectorias) - 1)  # deja al menos 1 para train
    return trayectorias[n_val:], trayectorias[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", type=Path, default=Path("evidencias_pruebas/senalamiento_trafico"))
    parser.add_argument("--destino", type=Path, default=Path("dataset_yolo26"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negativos", type=Path, default=None,
        help="Carpeta con imagenes SIN el objetivo (ej. captures/) para usarse como fondo.",
    )
    args = parser.parse_args()

    trayectorias = find_trajectories(args.origen)
    print(f"Trayectorias encontradas: {len(trayectorias)}")

    train_trajectories, val_trajectories = split_trajectories(
        trayectorias, args.val_ratio, args.seed
    )

    total_train = total_val = 0
    for split, group in (("train", train_trajectories), ("val", val_trajectories)):
        for trajectory_dir in group:
            pairs = collect_pairs(trajectory_dir)
            copy_pairs(pairs, split, args.destino, prefix=trajectory_dir.name)
            total_train += len(pairs) if split == "train" else 0
            total_val += len(pairs) if split == "val" else 0
            print(f"  [{split}] {trajectory_dir.name}: {len(pairs)} muestras")

    if args.negativos is not None:
        if args.negativos.exists():
            n_neg = add_negatives(args.negativos, args.destino, "train")
            print(f"Negativos (fondo, sin objeto) agregados a train: {n_neg}")
        else:
            print(f"[AVISO] --negativos apunta a {args.negativos}, que no existe. Se omite.")

    val_path = "images/val" if val_trajectories else "images/train"
    data_yaml = args.destino / "data.yaml"
    data_yaml.write_text(
        f'path: "{args.destino.resolve().as_posix()}"\n'
        "train: images/train\n"
        f"val: {val_path}\n"
        "names:\n"
        f'  0: "{CLASS_NAME}"\n',
        encoding="utf-8",
    )

    print("\nListo.")
    print(f"  Imagenes de entrenamiento: {total_train}")
    print(f"  Imagenes de validacion:   {total_val}")
    print(f"  data.yaml: {data_yaml.resolve()}")
    if total_train < 100:
        print(
            "\n[AVISO] Tienes menos de 100 imagenes de entrenamiento. YOLO26 "
            "parte de pesos preentrenados (transfer learning), asi que puede "
            "funcionar con poco, pero para un modelo confiable conviene grabar "
            "mas trayectorias variando angulo, distancia, iluminacion y fondo."
        )


if __name__ == "__main__":
    main()
