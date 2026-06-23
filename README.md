# Sistema de Sensores para la Asistencia a la Conducción de un Automóvil

## Módulo de percepción visual mediante cámara RGB-D

Este repositorio contiene los avances individuales del módulo de percepción visual desarrollado durante la estancia de verano en el CIC, dentro del proyecto **Sistema de Sensores para la Asistencia a la Conducción de un Automóvil**.

El trabajo se enfoca en el uso de una cámara **Orbbec Astra+** para capturar información visual del entorno mediante imagen RGB y datos de profundidad. Esta etapa forma parte de la preparación del sistema de percepción para futuras pruebas de procesamiento de imagen, aplicación de filtros, detección de bordes, segmentación por color e identificación inicial de elementos relevantes en el entorno.

---

## Objetivo

Desarrollar y documentar pruebas iniciales de percepción visual mediante cámara RGB-D, obteniendo información de imagen y profundidad para preparar la base del procesamiento visual del sistema.

El módulo busca mejorar la interpretación del entorno mediante técnicas de visión por computadora, reduciendo ruido visual y resaltando características importantes como contornos, colores, formas y posibles objetivos de interés.

---

## Alcance actual del repositorio

Actualmente, este repositorio incluye:

* Código fuente para pruebas de captura con cámara.
* Captura de imagen RGB.
* Captura de información de profundidad.
* Preparación para procesamiento en escala de grises.
* Preparación para aplicación de filtros visuales.
* Planeación de detección de bordes.
* Planeación de segmentación por color.
* Planeación de identificación de objetivos por color o figura.
* Documentación del avance individual del módulo de cámara.
* Organización inicial del repositorio para futuras pruebas de procesamiento visual.

---

## Tecnologías utilizadas

| Tecnología           | Uso dentro del proyecto                                             |
| -------------------- | ------------------------------------------------------------------- |
| Python               | Desarrollo de scripts de captura y procesamiento                    |
| OpenCV               | Visualización, manejo de imágenes, filtros y visión por computadora |
| Cámara Orbbec Astra+ | Obtención de imagen RGB y profundidad                               |
| Git                  | Control de versiones del proyecto                                   |
| GitHub               | Respaldo, documentación y seguimiento de avances                    |

---

## Estructura del proyecto

```text
RoodeSaavedra-SistemaSensores/
│
├── src/                              # Scripts principales del proyecto
│   └── captura_rgb_profundidad.py    # Captura RGB y profundidad
│
├── tests/                            # Pruebas realizadas
├── evidencias/                       # Capturas o resultados de pruebas
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Documentación general
```

---

## Script principal

### `captura_rgb_profundidad.py`

El script `captura_rgb_profundidad.py` permite realizar pruebas iniciales de captura visual utilizando la cámara Orbbec Astra+.

Su propósito es obtener y visualizar información en formato RGB y profundidad, con el fin de comprobar el funcionamiento básico de la cámara y preparar el entorno para las siguientes etapas del proyecto.

Este script representa el primer avance práctico del módulo de percepción visual, ya que permite obtener datos visuales que posteriormente podrán utilizarse para aplicar filtros, detectar bordes, analizar profundidad y realizar segmentación visual.

---

## Procesamiento visual, filtros y detección inicial

Una de las etapas principales del proyecto consiste en aplicar filtros y técnicas básicas de procesamiento de imagen para mejorar la calidad de la información obtenida por la cámara.

La aplicación de filtros permitirá reducir ruido visual, resaltar elementos importantes de la imagen y preparar los datos para futuras tareas de detección de objetos o análisis del entorno.

Entre las pruebas consideradas se encuentran:

* Conversión de imagen RGB a escala de grises.
* Suavizado de imagen para reducción de ruido.
* Aplicación de filtros de desenfoque.
* Detección de bordes para resaltar contornos.
* Segmentación por color.
* Ajuste de umbrales.
* Identificación de un objetivo por color específico.
* Búsqueda de figuras o formas dentro de la imagen.
* Análisis básico de profundidad.
* Preparación de imágenes para detección de elementos relevantes.

Además de la captura RGB y de profundidad, se contempla el uso de técnicas de visión por computadora para enfocar la percepción en un objetivo determinado. Este objetivo puede definirse por características visuales como color, forma o contraste respecto al fondo.

Por ejemplo, el sistema podrá orientarse a la búsqueda de un objeto de color específico, una figura determinada o una región de interés dentro de la escena. Esto permitirá filtrar información no relevante del entorno y concentrar el análisis en elementos útiles para la asistencia a la conducción.

Esta etapa se encuentra en preparación y será integrada progresivamente al repositorio conforme se realicen las pruebas correspondientes.

---

## Flujo de trabajo propuesto

```text
Captura RGB y profundidad
        ↓
Conversión y preprocesamiento de imagen
        ↓
Aplicación de filtros
        ↓
Reducción de ruido visual
        ↓
Detección de bordes y contornos
        ↓
Segmentación por color o figura
        ↓
Identificación de posibles elementos relevantes
        ↓
Documentación de resultados
```

---

## Instalación de dependencias

Para instalar las dependencias necesarias del proyecto, se puede utilizar:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Los scripts principales se encuentran dentro de la carpeta `src/`.

Para ejecutar el script de captura RGB y profundidad:

```bash
python src/captura_rgb_profundidad.py
```

---

## Avances realizados

### 1. Configuración inicial del repositorio

Se creó la estructura base del repositorio para organizar el código fuente, pruebas, evidencias y documentación general del proyecto.

### 2. Documentación inicial del proyecto

Se agregó una descripción general del proyecto, su objetivo, alcance, tecnologías utilizadas y estructura del repositorio.

### 3. Captura RGB y profundidad

Se agregó el script `captura_rgb_profundidad.py`, orientado a realizar pruebas iniciales de captura con la cámara Orbbec Astra+.

Este avance permite iniciar la obtención de datos visuales del entorno, tanto en imagen RGB como en profundidad.

### 4. Preparación para filtros y detección inicial

Se documentó la etapa de procesamiento visual, en la cual se aplicarán filtros para mejorar la calidad de imagen, reducir ruido, detectar bordes y preparar los datos para futuras pruebas de segmentación por color, identificación de figuras y detección de objetos relevantes.

---

## Estado actual

| Área                                  | Estado         |
| ------------------------------------- | -------------- |
| Repositorio base                      | Completado     |
| Documentación inicial                 | Completado     |
| Captura RGB                           | En pruebas     |
| Captura de profundidad                | En pruebas     |
| Aplicación de filtros                 | En preparación |
| Escala de grises                      | En preparación |
| Detección de bordes                   | En preparación |
| Segmentación por color                | En preparación |
| Identificación de objetivos por color | En preparación |
| Identificación de figuras o formas    | En preparación |
| Evidencias visuales                   | Pendiente      |
| Detección de objetos                  | Pendiente      |

---

## Próximos pasos

* Verificar el funcionamiento estable de la captura RGB.
* Verificar la lectura de datos de profundidad.
* Implementar pruebas en escala de grises.
* Aplicar filtros básicos de procesamiento de imagen.
* Probar filtros para reducción de ruido visual.
* Implementar detección de bordes.
* Realizar pruebas de segmentación por color.
* Probar identificación de un objetivo por color específico.
* Explorar detección de figuras o formas simples.
* Documentar pruebas con capturas reales.
* Agregar evidencias visuales obtenidas durante la ejecución.
* Preparar futuras pruebas de detección de objetos.

---

## Notas del desarrollo

Este repositorio corresponde al avance individual del módulo de cámara y percepción visual. La parte relacionada con LiDAR corresponde a otro integrante del proyecto, mientras que la calibración o integración entre cámara y LiDAR se trabajará posteriormente de forma conjunta.

---

## Autor

**Roode Saavedra Carrera**
Estancia de verano — CIC
Módulo individual: percepción visual mediante cámara
