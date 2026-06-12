Sistema de Sensores para la Asistencia a la Conducción de un Automóvil

Módulo de percepción visual

Este repositorio contiene el desarrollo correspondiente al módulo de percepción visual del proyecto “Sistema de Sensores para la Asistencia a la Conducción de un Automóvil”, realizado como parte de la estancia de Verano Delfín en el Centro de Investigación en Computación del Instituto Politécnico Nacional.

El trabajo se enfoca en la caracterización de una cámara Orbbec Astra, la adquisición de imágenes RGB, infrarrojas y de profundidad, la aplicación de filtros, el desarrollo de algoritmos de detección y orientación hacia objetos, y la preparación para pruebas físicas de evasión de obstáculos con un robot móvil referencial.

Actividades principales

* Caracterización de la cámara.
* Especificación de sensores.
* Aplicación de filtros de procesamiento de imagen.
* Implementación de modelos de reconocimiento de objetos.
* Generación de algoritmos de percepción.
* Pruebas físicas de evasión de obstáculos.


Estado actual del proyecto

Hasta el momento se han realizado pruebas de reconocimiento y comunicación con la cámara Orbbec Astra en Windows. El sistema operativo identifica correctamente el dispositivo y OpenNI permite acceder a diferentes flujos de datos.

Funcionalidades comprobadas

* Detección de la cámara Orbbec por medio del sistema operativo Windows.
* Reconocimiento del dispositivo mediante OpenNI.
* Adquisición del flujo infrarrojo.
* Visualización de imagen infrarroja en Python.
* Visualización del flujo RGB mediante OpenCV.
* Aplicación de normalización y mapas de color.
* Generación de indicaciones básicas de orientación hacia un objeto.


Este proyecto es probado con Python 3.10 en el SO Windows. Se recomienta crear un entorno virtual antes de instalar las dependencias requeridas:

py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
pip install -r requirements.txt

además es necesario instalar el controlador de la cámara Orbbec y disponer del SDK OpenNI compatible con su dispositivo.