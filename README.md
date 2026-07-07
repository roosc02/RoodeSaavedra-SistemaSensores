# Vision con Orbbec Astra+


## Ejecucion

```powershell
python -m pip install -r requirements.txt
python orbbec_vision.py
```

En el primer arranque se crea `camera_config.json`. Verifica:

- `openni_path`: carpeta que contiene el runtime de OpenNI.
- `rgb_index`: indice de la camara RGB. Para la RGB de la Orbbec normalmente es `0`.
- `depth_scale_mm`: milimetros por unidad de profundidad.
- `fx`, `fy`, `cx`, `cy`: parametros intrinsecos RGB; inicialmente pueden ser `null`.

- Las trayectorias se inician automaticamente al estabilizar una señal de ALTO;
  no se necesita pulsar ninguna tecla. Cada una guarda 10 muestras del mismo
  `track_id`.

## YOLO y dataset RGB-D-IR

Configura estas opciones en `camera_config.json` si necesitas cambiarlas:

```json
{
  "yolo_model": "yolov8s-world.pt",
  "yolo_target_class": "senalamiento_trafico",
  "yolo_traffic_only": true,
  "yolo_confidence": 0.02,
  "yolo_iou": 0.5,
  "yolo_device": "cpu",
  "yolo_tracker_config": "bytetrack_stop.yaml",
  "yolo_dataset_dir": "evidencias_pruebas/senalamiento_trafico",
  "trajectory_samples": 10,
  "trajectory_interval_seconds": 0.3,
  "trajectory_auto_capture": true,
  "trajectory_auto_stable_frames": 5
}
```

YOLO se inicia automaticamente buscando una señal de vialidad color rojo y la registra con la
unica clase `senalamiento_trafico`. Todas las etiquetas usan la clase `0`. Si aparecen varios
candidatos, solo muestra y mide uno y mantiene su `track_id`.
Los detectores anteriores por color y forma permanecen aislados en el codigo
como referencia, pero no estan conectados al bucle principal ni tienen teclas
para activarlos.

Para pruebas con una señal mostrada en un telefono, YOLO recibe internamente el
RGB sin espejo aunque la interfaz siga mostrandose reflejada para conservar la
calibracion. El prompt visual es `red octagonal traffic sign` y ByteTrack usa
umbrales bajos adaptados a pantallas; la estabilidad de cinco cuadros limita
falsos disparos.
Tambien puedes indicar la ruta a tu propio `best.pt` en `yolo_model`; en ese caso
sus nombres de clase deben describir señales de transito.

Solo presenta la señal frente a la cámara. Tras cinco detecciones consecutivas,
la aplicación inicia sola una trayectoria de 10 muestras. Cada trayectoria crea
una carpeta independiente con:

- RGB JPG y etiqueta YOLO normalizada;
- profundidad cruda y profundidad alineada a RGB en PNG de 16 bits;
- infrarrojo PNG;
- matrices completas NPZ;
- JSON por muestra con distancia, dimensiones, area, FPS, calibracion,
  estadisticas y tiempos de lectura de los sensores;
- RGB anotado con caja, confianza, Track ID, distancia y linea de trayectoria;
- `trayectoria_final.jpg` con el recorrido completo;
- `data.yaml` y resumen de trayectoria.

La estructura principal se crea al iniciar:

```text
evidencias_pruebas/senalamiento_trafico/
|-- clase.txt
|-- reportes/datos_sensores.csv
`-- trayectorias/trajectory_.../
    |-- images/
    |-- labels/
    |-- depth_raw/
    |-- depth_aligned/
    |-- ir/
    |-- raw/
    |-- metadata/
    |-- evidencias_rgb/
    `-- trayectoria_final.jpg
```

Los archivos de `calibration_data` no contienen bordes, mapas de color ni texto,
por lo que pueden utilizarse posteriormente para calibracion.

## Medicion y registro entre sensores

La distancia se toma como la mediana de profundidad dentro de la region del
objetivo, pero usando `depth_aligned_to_rgb`, es decir, la profundidad ya llevada
al plano RGB.

Si `fx` y `fy` estan configurados, se estiman ancho, alto y area en milimetros.

## Calibracion RGB + profundidad

Usa un tablero de ajedrez plano. Los valores `cols` y `rows` son las esquinas
internas, no los cuadros impresos. Por ejemplo, un tablero de 10x7 cuadros tiene
9x6 esquinas internas.

1. Ejecuta la vision:

   ```powershell
   python orbbec_vision.py
   ```

2. Coloca el tablero en 12 a 25 posiciones diferentes:

   - cerca, medio y lejos;
   - inclinado hacia izquierda/derecha/arriba/abajo;
   - ocupando centro, esquinas y bordes de la imagen.

3. Antes de capturar, ajusta el ensamble visual con:

   - `J` / `L` para izquierda/derecha;
   - `I` / `K` para arriba/abajo;
   - `U` / `O` para escala.

   Cuando se vea bien, presiona `G` para guardar ese ensamble.

4. Presiona `C` en cada pose del tablero. Cada captura guarda:

   - `rgb_bgr`;
   - `depth_raw`;
   - `depth_aligned_to_rgb`;
   - `ir_raw`.

5. Cierra la vision y calibra. Cambia `--square-mm` por el tamano real de cada
   cuadro del tablero:

   ```powershell
   python calibrate_rgb_depth.py --cols 9 --rows 6 --square-mm 25 --apply
   ```

El calibrador actualiza en `camera_config.json`:

- `fx`, `fy`, `cx`, `cy` de RGB;
- `rgb_dist_coeffs`;
- `depth_scale_mm`, solo si la correccion sugerida esta dentro de un rango razonable.

Ademas crea:

- `calibration_data\rgb_depth_calibration_report.json`;
- `calibration_data\debug_corners\*_corners.png`.

Nota: esta calibracion usa `depth_aligned_to_rgb`, por lo que ya sirve para medir
distancias y dimensiones en el plano RGB. Para una extrinseca fisica completa
profundidad->RGB se necesitan tambien intrinsecos reales del sensor de profundidad
y correspondencias visibles en profundidad, o los parametros de registro de fabrica
del dispositivo.
