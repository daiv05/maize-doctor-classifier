# Análisis de Sesgos, Equidad y Ética

Un diagnóstico erróneo en el campo tiene consecuencias directas sobre la economía de subsistencia de las familias agricultoras. Por ello, la auditoría ética del modelo evaluó tres posibles fuentes de sesgo algorítmico: la dependencia del fondo de la fotografía frente a la lesión foliar, la disparidad de rendimiento entre entornos de laboratorio y campo real, y la consistencia de los mapas de atención visual.

---

## La prueba de ablación espacial y el control nulo

Para medir qué tanto influye el contexto exterior de la foto frente a los síntomas centrales de la hoja, realizamos una prueba de oclusión espacial sobre las 5,015 imágenes del conjunto de prueba, comparando los resultados contra un predictor nulo (que adivina siempre la clase mayoritaria). La máscara es un rectángulo fijo que abarca del 20 % al 80 % de cada eje: ocluir el centro tapa el 36.3 % del área y deja visible un anillo periférico del 63.7 %; ocluir la periferia tapa ese 63.7 % y deja visible la caja central del 36.3 %.

![Ablación con control nulo](/resultados/ablacion_control_nulo.png)

| Condición de la prueba | Qué parte de la imagen ve el modelo | Área visible | Exactitud (Accuracy) | Exceso sobre el azar nulo | Confianza retenida |
|---|---|:---:|:---:|:---:|:---:|
| **Imagen completa original** | Toda la escena (hoja y fondo) | 100.0 % | **97.91 %** | +71.77 pp | 100.0 % |
| **Oclusión central** | **Anillo periférico** (centro tapado) | 63.7 % | **79.46 %** | **+53.32 pp** | **69.16 %** |
| **Oclusión periférica** | **Caja central** (fondo y bordes tapados) | 36.3 % | **42.25 %** | +16.11 pp | 42.66 % |
| *Línea base de control nulo* | *Sin ver la imagen (clase más frecuente)* | 0.0 % | *26.14 %* | 0.0 pp | — |

**El anillo periférico basta para acertar en ocho de cada diez imágenes** (79.46 %, +53.32 pp sobre el azar) conservando casi el 70 % de la confianza original, pese a que el centro de la escena —donde se encuentra la lámina foliar en la práctica totalidad del corpus— está tapado en negro.

Las dos condiciones de oclusión **no están igualadas en área**: la que deja visible el anillo muestra 1.75 veces más píxeles que la que deja visible la caja central, así que la distancia entre 79.46 % y 42.25 % no es atribuible únicamente a la región observada. Lo que sí queda establecido es la comparación de cada condición contra el control nulo: el anillo supera al predictor constante en +53.32 pp y la caja central en +16.11 pp.

La lectura de fondo es que las redes convolucionales son muy sensibles a la composición global de la toma (el color del suelo, la vegetación circundante o la luz ambiental). De ahí la importancia de guiar al agricultor mediante el marco en pantalla para asegurar que el encuadre se concentre en la lámina foliar.

---

## Equidad entre laboratorio y campo real

Auditamos si el modelo discrimina o pierde precisión cuando se le evalúa en entornos naturales de cultivo frente a fotos en condiciones controladas de laboratorio:

| Subgrupo evaluado | Muestras ($N$) | Exactitud (Accuracy) | Macro $F_1$ | Clases con soporte |
|---|:---:|:---:|:---:|:---:|
| **Campo real (`real`)** | 4,483 | **98.04 %** | **0.9215** | 9 de 9 |
| **Laboratorio (`lab`)** | 532 | **96.80 %** | **0.9423** | 3 de 9 |

El subgrupo de laboratorio solo contiene 3 de las 9 clases en los datasets públicos (*roya*, *mancha gris* y *tizón*), de modo que su macro $F_1$ promedia sobre tres categorías y el de campo real sobre nueve. Las dos cifras no son directamente comparables entre sí, y el cociente entre ellas no mide disparidad sino la distinta composición de clases de cada promedio. La comparación válida restringe ambos subgrupos a las tres clases que comparten:

| Subgrupo evaluado | Muestras ($N$) | Exactitud (Accuracy) | Macro $F_1$ (3 clases) |
|---|:---:|:---:|:---:|
| **Campo real (`real`)** | 1,121 | **97.77 %** | **0.9050** |
| **Laboratorio (`lab`)** | 532 | **96.80 %** | **0.9423** |
| *Ratio de impacto dispar* | — | *$DIR = 0.9901$* | *$DIR = 0.9604$* |

Sobre esa base comparable, la disparidad en exactitud es de **0.97 puntos porcentuales** y ambos ratios de impacto dispar superan holgadamente el criterio del 80 %. El sentido de la disparidad se invierte según la métrica: campo real acierta más en conjunto, y laboratorio reparte mejor el acierto entre las tres clases.

![Matrices de confusión desagregadas](/fairness/disaggregated_confusion_matrices.png)

La desagregación por clase dentro de cada entorno descubre una asimetría que las métricas globales no revelan:

| Clase | $FNR$ laboratorio | $N$ | $FNR$ campo real | $N$ |
|---|:---:|:---:|:---:|:---:|
| **Roya común** | 0.31 % | 322 | **31.25 %** | 16 |
| Mancha gris (GLS) | 15.58 % | 77 | 5.16 % | 213 |
| Tizón foliar (NCLB) | 3.01 % | 133 | 1.01 % | 892 |

La **roya común** es prácticamente infalible en laboratorio (1 fallo sobre 322 imágenes) y **falla en casi un tercio de los casos en campo real** (5 fallos sobre 16). Es el reflejo directo de la composición del corpus: el 95 % de las imágenes de roya del conjunto de prueba proceden de repositorios de laboratorio, así que el modelo aprendió la patología bajo fondo controlado y no la reconoce con fiabilidad sobre vegetación. La mancha gris exhibe el patrón inverso y más favorable. La lectura de la roya en campo está limitada por el tamaño de su muestra ($N = 16$): señala la dirección del problema, no su magnitud exacta.

---

## Desagregación por fuente de procedencia

![Rendimiento por procedencia con su control nulo](/resultados/equidad_por_fuente.png)

Al auditar los resultados a través de los 14 repositorios del corpus, encontramos un fenómeno ilustrativo:
- Cuatro fuentes alcanzan un 100 % de exactitud perfecta, pero lo hacen de forma trivial: **contienen una sola clase**, por lo que cualquier modelo constante acertaría el 100 % sin aprender nada.
- En cambio, repositorios multi-clase como `corn_leaf_roboflow` (seis clases, 92.5 % de acierto y +65.8 pp sobre el azar) o `maize_africa` (+56.4 pp sobre el azar) son los que verdaderamente demuestran que el sistema aprende patrones biológicos complejos.

---

## Atención visual con Grad-CAM y atribución SHAP

![Panel Grad-CAM](/fairness/gradcam_samples.png)

Los mapas de activación visual con Grad-CAM sitúan, en condiciones normales de campo, los mayores gradientes sobre las manchas foliares, las pústulas fúngicas y las decoloraciones típicas de deficiencia. Esa lectura visual no basta como evidencia: la lámina foliar ocupa el centro y la mayor parte del cuadro en prácticamente todo el corpus, así que una atribución repartida al azar también recaería mayoritariamente sobre la hoja.

Para separar ambos efectos se mide el **exceso de atribución**: la fracción de atribución positiva que cae sobre el tejido foliar menos la cobertura de la máscara de hoja, que es el nivel de azar correspondiente a cada imagen.

| Clase | $N$ | Atribución en hoja | Cobertura (azar) | Exceso |
|---|:---:|:---:|:---:|:---:|
| Roya común | 29 | 0.897 | 0.703 | **+0.193** |
| Deficiencia de fósforo | 28 | 0.474 | 0.378 | +0.096 |
| Deficiencia de nitrógeno | 23 | 0.352 | 0.302 | +0.051 |
| Tizón foliar (NCLB) | 30 | 0.575 | 0.584 | −0.009 |
| Planta sana | 30 | 0.538 | 0.578 | −0.040 |
| Mancha gris (GLS) | 26 | 0.570 | 0.639 | −0.069 |
| **Media de las seis clases** | — | — | — | **+0.037** |

El perfil se calculó sobre 270 imágenes (30 por clase), empleando como máscara de hoja la segmentación producida por el modelo del repositorio `maize-doctor-segmenter` en lugar de un umbral cromático. Las tres clases restantes —*gusano cogollero*, *necrosis letal* y *deficiencia de potasio*— quedan fuera de la tabla porque su máscara alcanza coberturas de entre 0.54 y 0.86 y el control de calidad rechaza entre el 37 % y el 57 % de sus imágenes, así que su ratio no es interpretable. Los datos brutos están en [`xai_global_summary.csv`](/es/resultados/evidencia/) y su desglose por imagen en `xai_global_per_image.csv`.

**El exceso medio es de +0.037 y su signo se reparte: tres clases por encima del azar y tres por debajo.** La atribución sobre la hoja no supera de forma consistente la que produciría una atención repartida en proporción al área; únicamente la roya común exhibe una concentración foliar netamente superior a su nivel de azar.

Este resultado no contradice la prueba de oclusión, porque ambas técnicas responden a preguntas distintas. Grad-CAM y SHAP son métodos de atribución *local*: miden la contribución de cada región **mientras el resto de la imagen sigue presente**, y premian la evidencia concentrada y de alto contraste, como es una lesión. La ablación mide *suficiencia*: cuánto acierto sobrevive cuando todo lo demás se elimina. La firma de procedencia de un repositorio —paleta cromática, iluminación, ruido de sensor, textura del suelo— es un estadístico de baja frecuencia repartido sobre decenas de miles de píxeles: cada píxel aporta poco, así que nunca aparece como punto caliente en un mapa de atribución, pero el agregado carga información de clase suficiente para sostener el 79.46 % de acierto de la ablación. A ello se suma que la última capa convolucional produce un mapa de 7×7 celdas cuyo campo receptivo abarca casi la escena completa, de modo que evidencia recogida del fondo puede quedar representada sobre una coordenada central tras la interpolación.

La combinación de estas auditorías —conocer el peso del fondo, verificar la equidad entre entornos y blindar el sistema con el marco de captura y el detector OOD— permite desplegar la tecnología declarando con precisión qué está medido y qué no.
