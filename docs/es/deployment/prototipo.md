# Prototipo en dispositivo

Llevar una red neuronal de un cuaderno de investigación a un teléfono físico siempre cambia la perspectiva del proyecto. Los números de precisión teórica dejan de importar por un momento y lo que cuenta es la experiencia real: si la aplicación abre rápido, si la cámara responde sin trabarse y si el diagnóstico aparece en pantalla antes de que el usuario pierda la paciencia.

Probamos el prototipo funcional de DoctorMaiz sobre un teléfono real de gama media —un **Motorola edge 40 neo** con procesador MediaTek Dimensity 7030 y Android 15—, ejecutando el modelo `efficientnet_lite0` cuantizado a Int8 (3.74 MB) de forma 100% offline.

---

## La experiencia de uso en la app

La aplicación fue pensada desde el inicio para trabajar en condiciones agrícolas cotidianas, donde la sencillez y la claridad visual mandan.

![Pantalla de inicio y cámara guiada](/app/home.png)

Al abrir la aplicación, el usuario encuentra un acceso directo al escáner y a su historial de análisis previos. 

Uno de los mayores retos al evaluar fotos en campo es cómo la gente sostiene el teléfono. En los datasets de laboratorio o de entrenamiento, las fotos son casi siempre primeros planos nítidos de la hoja. Pero en una milpa real, un agricultor tiende a tomar la foto desde un metro de distancia, capturando un 80% de cielo, tierra o maleza y apenas una pequeña hoja en el centro. 

![Cámara con el marco guía](/app/camara-marco-guia.png)

Para resolver esto sin obligar al usuario a recortar fotos manualmente, la cámara incorpora un **marco guía con silueta de hoja**. El visor le pide al productor acercarse a unos 20 o 30 centímetros y hacer que la lámina foliar llene el área marcada. La app recorta automáticamente esa región central antes de procesarla, asegurando que la red reciba tejido vegetal y no el suelo de la parcela.

![Pantalla de resultado con diagnóstico](/app/resultado-mancha-gris.png)

Una vez capturada la imagen, la pantalla de resultados muestra el diagnóstico identificado, el nivel de confianza, la foto procesada y recomendaciones prácticas de manejo agronómico recomendadas por técnicos agrícolas.

---

## Tiempos de respuesta y rendimiento real

Tiempos medidos en el Motorola edge 40 neo sobre 143 imágenes (95 fotos de campo externas y 48 del corpus). La meta es que preprocesado e inferencia sumen menos de 300 ms:

| Etapa | Tiempo típico (mediana) | Qué ocurre en ese lapso |
|---|:---:|---|
| **Preprocesado** | 98 ms (p95 279 ms) | Decodificación, corrección de orientación, reescalado con antialiasing y armado del tensor. |
| **Inferencia** | **23 ms** | El modelo cuantizado ejecuta sus capas en la CPU del teléfono. |

La inferencia toma unos 23 milisegundos y el preprocesado unos 98: el cálculo del modelo no es el cuello de botella. El resto del tiempo hasta la pantalla de resultados corresponde a interfaz, persistencia en SQLite local y renderizado.

### Fidelidad frente al servidor

La primera medición en el teléfono coincidía con el computador solo en el 84 % de las predicciones: las imágenes de entre 224 y 896 px por lado se escalaban en un solo paso bilineal sin prefiltro, mientras que el entrenamiento usa el redimensionado con antialias de Pillow. Al pasar el tensor del propio teléfono por el int8 del servidor se reprodujeron sus predicciones, así que el intérprete no era la causa. La app ahora porta el remuestreo de Pillow (`src/ml/resample.ts`, verificado byte a byte contra su salida) y el teléfono coincide con el int8 del servidor en las 143 imágenes.

| Lote | Aciertos en el teléfono | Rechazo OOD en el teléfono |
|---|---:|---:|
| Corpus (48) | 45 (93.8 %) | 8.3 % |
| Fotos de campo externas (95) | 38 (40.0 %) | 52.6 % |

Las cifras de campo coinciden con las del computador y confirman la limitación descrita en [resultados](../resultados/index.md).

---

## Comportamiento del diagnóstico y la red de seguridad OOD

Sometimos el prototipo a un conjunto de pruebas con fotografías reales del conjunto de evaluación independiente, abarcando las nueve clases del cultivo:

![Resultados y rechazo de imágenes fuera de dominio](/app/resultado-potasio.png)

En el teléfono, el intérprete int8 reproduce las predicciones del int8 del servidor en 143 de 143 imágenes. Las diferencias entre teléfono y servidor sobre una misma foto provienen del escalado de la imagen y no del modelo; la paridad de muestra de 30 imágenes del int8 no se aprobó por una sola imagen límite (max |Δprob| 0.198 frente al umbral 0.15).

Lo más valioso de las pruebas en dispositivo fue comprobar cómo actúa el **detector de imágenes fuera de dominio (OOD)**. En lugar de forzar una etiqueta cuando una foto resulta confusa o no corresponde a maíz, el sistema activa su salvaguarda:

![Imagen rechazada por el detector](/app/resultado-no-reconocida.png)

El detector del modelo oficial usa un umbral de 106.37 (percentil 95 de validación): detecta 5 de 5 imágenes sintéticas (negro, blanco, gris, ruido y azul) y rechaza el 3.8 % de 500 imágenes de test. Evidencia de la exportación en [`export_movil_20260921_204608/`](../reproducibilidad/evidencia/export_movil_20260921_204608/export_summary_int8.json).

Cuando la distancia matemática supera el umbral de seguridad, la app prefiere decir con honestidad que la imagen no fue reconocida y sugiere repetir la captura con mejor iluminación o enfoque. Para una herramienta de campo, saber dudar es tan importante como acertar.

---

## Por qué la aplicación pide aportes comunitarios

Las pruebas en el teléfono dejaron en evidencia una verdad inevitable de los datos públicos: de las 11 fuentes (`source_id`) que integran el corpus, casi todas provienen de África, India o laboratorios estadounidenses. **Ninguna fue capturada en El Salvador.**

![Pantalla para contribuir al dataset](/app/contribuir.png)

Las variedades locales de maíz, el tipo de suelo volcánico salvadoreño, las malezas locales y hasta la luz tropical tienen sutilezas que ningún ajuste de hiperparámetros puede inventar. La distancia entre acertar sobre fotos de datasets públicos y acertar en una milpa de Chalatenango o San Vicente solo se cierra de una forma: con datos propios de la región.

Por esa razón la aplicación incluye un módulo de **contribución ciudadana**. Cuando el usuario cuenta con conexión a internet, puede compartir fotos con su diagnóstico para nutrir un banco de datos nacional. El camino para perfeccionar la herramienta en el futuro no está en inventar modelos más pesados, sino en enriquecer la base de conocimiento con la realidad del campo salvadoreño.
