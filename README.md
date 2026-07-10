# Vision con Orbbec Astra+

La aplicacion muestra una cuadricula 2x2:

1. Bordes Canny de RGB sobre fondo negro.
2. Ensamble RGB + profundidad alineada.
3. Bordes Canny de profundidad, suavizados temporalmente con Kalman.
4. Opciones, telemetria y datos del objetivo.

El panel de datos muestra resolucion, FPS, pixeles activos, pixeles detectados
como borde, estadisticas del sensor, umbrales Canny, estado del objetivo y estado
del ensamble RGB+profundidad.

La aplicacion conserva los filtros temporales Kalman, los bordes Canny y el
ensamble RGB+profundidad. Ademas incluye deteccion automatica y tracking de un
objeto COCO con YOLO26 + ByteTrack y captura coordinada RGB-D para crear
evidencias y datasets.

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

## Navegacion y motores

El proyecto incluye una capa de navegacion en `navegar.py` y un controlador de
motores en `motores.py`. La navegacion usa profundidad alineada a RGB y
detecciones YOLO para decidir:

- `forward`: avanzar;
- `slow`: avanzar lento;
- `stop`: detener;
- `turn_left`: girar a la izquierda;
- `turn_right`: girar a la derecha.

Los motores estan desactivados por defecto. En Raspberry Pi configura
`camera_config.json` con pines GPIO BCM de tu puente H:

```json
{
  "motor_enabled": true,
  "motor_dry_run": true,
  "motor_max_speed": 0.30,
  "motor_left_forward_pin": 17,
  "motor_left_backward_pin": 27,
  "motor_right_forward_pin": 22,
  "motor_right_backward_pin": 23
}
```

Primero deja `motor_dry_run` en `true`: veras la decision en pantalla sin mover
el robot. Cuando confirmes que las decisiones son correctas, cambia
`motor_dry_run` a `false`.

Los valores por defecto estan pensados para un robot mini: velocidad moderada y
frenado cercano. No conectes motores directo a los GPIO. Usa un driver de motor
o puente H, fuente separada para motores y tierra comun entre Raspberry Pi y
driver.

## Controles

- `Q` o `Esc`: salir.
- `C`: guardar matrices originales RGB y profundidad en un archivo `.npz`, ademas de PNG.
- `P`: guardar la cuadricula completa.
- `G`: guardar el ensamble manual RGB+profundidad en `camera_config.json`.
- `1`, `2`: seleccionar ajustes Canny de RGB o profundidad.
- `A` / `Z`: aumentar o reducir el umbral Canny bajo.
- `S` / `X`: aumentar o reducir el umbral Canny alto.
- `I` / `K`: mover la profundidad alineada arriba / abajo.
- `J` / `L`: mover la profundidad alineada izquierda / derecha.
- `U` / `O`: reducir / aumentar la escala de profundidad sobre RGB.
- `R`: reiniciar el ensamble manual.
- Las trayectorias se inician automaticamente al estabilizar un objeto COCO;
  no se necesita pulsar ninguna tecla. Cada una guarda 10 muestras del mismo
  `track_id`.

## YOLO y dataset RGB-D

Configura estas opciones en `camera_config.json` si necesitas cambiarlas:

```json
{
  "yolo_model": "yolo26n.pt",
  "yolo_target_class": "person,car,dog,stop sign",
  "yolo_traffic_only": false,
  "yolo_confidence": 0.25,
  "yolo_iou": 0.5,
  "yolo_device": "cpu",
  "yolo_tracker_config": "bytetrack_stop.yaml",
  "yolo_dataset_dir": "evidencias_pruebas/coco_objetos",
  "trajectory_samples": 10,
  "trajectory_interval_seconds": 0.3,
  "trajectory_auto_capture": true,
  "trajectory_auto_stable_frames": 5
}
```

YOLO26 se inicia automaticamente con pesos COCO. Por ahora se filtran las clases
`person`, `car`, `dog` y `stop sign`, registradas como
`persona`, `carro`, `perro` y `senalamiento_trafico`. No se
selecciona color, forma ni tipo desde la interfaz.
Los detectores anteriores por color y forma permanecen aislados en el codigo
como referencia, pero no estan conectados al bucle principal ni tienen teclas
para activarlos.

La deteccion nace en RGB: YOLO26 obtiene la caja, etiqueta y confianza. Despues
esa caja se consulta sobre `depth_aligned_to_rgb` para estimar distancia,
dimensiones y area usando la calibracion RGB-D. Arboles se dejan fuera porque
COCO no incluye una clase `tree/arbol`.

Presenta el objeto frente a la cámara. Tras cinco detecciones consecutivas,
la aplicación inicia sola una trayectoria de 10 muestras. Cada trayectoria crea
una carpeta independiente con:

- RGB JPG y etiqueta YOLO normalizada;
- profundidad cruda y profundidad alineada a RGB en PNG de 16 bits;
- matrices completas NPZ;
- JSON por muestra con distancia, dimensiones, area, FPS, calibracion,
  estadisticas y tiempos de lectura de los sensores;
- RGB anotado con caja, confianza, Track ID, distancia y linea de trayectoria;
- profundidad normalizada para visualizacion, sin reemplazar los crudos;
- `trayectoria_final_rgb.jpg` y `trayectoria_final_sensores.jpg`;
- `data.yaml` y resumen de trayectoria.

La estructura principal se crea al iniciar:

```text
evidencias_pruebas/coco_objetos/
|-- clase.txt
|-- reportes/datos_sensores.csv
`-- trayectorias/trajectory_.../
    |-- images/
    |-- labels/
    |-- depth_raw/
    |-- depth_aligned/
    |-- raw/
    |-- metadata/
    |-- evidencias_rgb/
    |-- depth_visual/
    |-- evidencias_sensores/
    |-- trayectoria_final_rgb.jpg
    `-- trayectoria_final_sensores.jpg
```

Las cajas generadas por un modelo preentrenado son pseudo-etiquetas: deben
revisarse antes de entrenar o validar un modelo propio. La lectura de RGB y
profundidad se realiza en el mismo ciclo y el JSON registra el desfase
observado; esto no equivale a sincronizacion hardware de ambos sensores.

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
