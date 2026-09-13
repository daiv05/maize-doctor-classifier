# Análisis de Sesgos, Equidad y Ética

Un diagnóstico erróneo en el campo tiene consecuencias directas sobre la economía de subsistencia de las familias agricultoras. Por ello, la auditoría ética del modelo evaluó tres posibles fuentes de sesgo algorítmico: la dependencia del fondo de la fotografía frente a la lesión foliar, la disparidad de rendimiento entre entornos de laboratorio y campo real, y la consistencia de los mapas de atención visual.

---

## La prueba de ablación espacial y el control nulo

Para medir qué tanto influye el contexto exterior de la foto frente a los síntomas centrales de la hoja, realizamos una prueba de oclusión espacial sobre las 5,015 imágenes del conjunto de prueba, comparando los resultados contra un predictor nulo (que adivina siempre la clase mayoritaria):

![Ablación con control nulo](/resultados/ablacion_control_nulo.png)

| Condición de la prueba | Qué parte de la imagen ve el modelo | Exactitud (Accuracy) | Exceso sobre el azar nulo | Confianza retenida |
|---|---|:---:|:---:|:---:|
| **Imagen completa original** | Toda la escena (hoja y fondo) | **97.91 %** | +71.77 pp | 100.0 % |
| **Oclusión central (60 %)** | **Solo el contorno y el fondo** (hoja tapada) | **79.46 %** | **+53.32 pp** | **69.16 %** |
| **Oclusión periférica (40 %)** | **Solo el centro** (fondo y bordes tapados) | **42.25 %** | +16.11 pp | 42.66 % |
| *Línea base de control nulo* | *Sin ver la imagen (clase más frecuente)* | *26.14 %* | 0.0 pp | — |

El hallazgo es inequívoco: **el fondo por sí solo predice mejor que el centro de la hoja por sí solo (79.46 % frente a 42.25 %)**. 

Con el centro tapado, el modelo aún acierta ocho de cada diez veces y conserva casi el 70 % de su confianza, mientras que al tapar los bordes y dejar visible únicamente la lesión central, el acierto cae a la mitad. 

Esto demuestra que las redes convolucionales son muy sensibles a la composición global de la toma (el color del suelo, la vegetación circundante o la luz ambiental). De ahí la importancia de guiar al agricultor mediante el marco en pantalla para asegurar que el encuadre se concentre en la lámina foliar.

---

## Equidad entre laboratorio y campo real

Auditamos si el modelo discrimina o pierde precisión cuando se le evalúa en entornos naturales de cultivo frente a fotos en condiciones controladas de laboratorio:

| Subgrupo evaluado | Muestras ($N$) | Exactitud (Accuracy) | Macro $F_1$ sobre clases evaluables |
|---|:---:|:---:|:---:|
| **Campo real (`real`)** | 4,483 | **98.04 %** | **0.9215** |
| **Laboratorio (`lab`)** | 532 | **96.80 %** | **0.9423** |

La disparidad en exactitud es de apenas **1.24 puntos porcentuales** y el ratio de impacto dispar ($DIR = 0.9778$) supera holgadamente el criterio del 80 %, confirmando que el clasificador mantiene un rendimiento parejo en ambos dominios.

![Matrices de confusión desagregadas](/fairness/disaggregated_confusion_matrices.png)

Como se detalló en la fase de evaluación, el subgrupo de laboratorio solo contiene 3 clases en los datasets públicos (*roya*, *mancha gris* y *tizón*); sobre esas patologías evaluables, el comportamiento del modelo es altamente consistente y no genera falsos positivos hacia enfermedades de campo.

---

## Desagregación por fuente de procedencia

![Rendimiento por procedencia con su control nulo](/resultados/equidad_por_fuente.png)

Al auditar los resultados a través de los 14 repositorios del corpus, encontramos un fenómeno ilustrativo:
- Cuatro fuentes alcanzan un 100 % de exactitud perfecta, pero lo hacen de forma trivial: **contienen una sola clase**, por lo que cualquier modelo constante acertaría el 100 % sin aprender nada.
- En cambio, repositorios multi-clase como `corn_leaf_roboflow` (seis clases, 92.5 % de acierto y +65.8 pp sobre el azar) o `maize_africa` (+56.4 pp sobre el azar) son los que verdaderamente demuestran que el sistema aprende patrones biológicos complejos.

---

## Atención visual con Grad-CAM y atribución SHAP

![Panel Grad-CAM](/fairness/gradcam_samples.png)

Los mapas de activación visual con Grad-CAM confirman cualitativamente que, en condiciones normales de campo, la red dirige sus mayores gradientes hacia las manchas foliares, las pústulas fúngicas y las decoloraciones típicas de deficiencia. 

En paralelo, los análisis cuantitativos con SHAP corroboran que la fracción de atribución positiva que recae sobre el tejido foliar supera significativamente la cobertura que daría una atención aleatoria.

La combinación de estas auditorías —conocer el peso del fondo, verificar la equidad entre entornos y blindar el sistema con el marco de captura y el detector OOD— garantiza que la tecnología se despliegue con responsabilidad y rigor ético en beneficio de los productores salvadoreños.
