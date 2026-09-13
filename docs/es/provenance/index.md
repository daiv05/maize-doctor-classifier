# Procedencia y Fuga de Información

Uno de los mayores aprendizajes de este proyecto no vino de afinar capas ni de inventar funciones de pérdida complicadas, sino de hacernos una pregunta incómoda: **¿el modelo está aprendiendo a reconocer enfermedades foliares o solo está reconociendo de qué cámara y de qué experimento vino cada fotografía?**

En visión por computadora aplicada a la agricultura, casi todo el mundo entrena con mezclas de datasets públicos disponibles en internet. Nosotros reunimos más de 33,400 imágenes provenientes de 14 fuentes distintas. Sobre el papel, el modelo alcanzaba cifras espectaculares en las particiones estándar: más del 97 % de exactitud y 0.94 de Macro F1. 

Sin embargo, al auditar el origen de cada archivo descubrimos un sesgo estructural severo que ponía en duda la honestidad de esas métricas.

---

## La anatomía del problema

Cuando se combinan datasets públicos, las fotos no vienen repartidas de manera homogénea. Cada repositorio fue capturado por investigadores diferentes, con teléfonos distintos, bajo la luz de su propio país y con fondos particulares (el piso de un invernadero en Uganda, una cartulina blanca en Estados Unidos o tierra rojiza en la India).

Al mapear las 14 fuentes contra las 9 clases del cultivo encontramos un desbalance crítico:
- **Cinco de las catorce fuentes aportan una sola clase.**
- Más del **80 % de todas las fotos de hojas sanas (`healthy`)** provienen de solo tres repositorios que no contienen ninguna otra enfermedad.
- Cada patología está fuertemente dominada por una única fuente que concentra entre el 30 % y el 60 % de sus imágenes.

Esto abre la puerta a un atajo tramposo (*shortcut learning*): la red neuronal, que siempre busca el camino más fácil para minimizar el error, no necesita fijarse en las pústulas microscópicas de un hongo. Le basta con identificar la temperatura de color de la cámara, el tipo de suelo o la resolución del sensor para saber a qué dataset pertenece la foto y, en consecuencia, adivinar la clase con enorme facilidad.

---

## La prueba del anillo: diagnosticar sin ver la hoja

Para comprobar si este atajo estaba ocurriendo de verdad, diseñamos un experimento de control negativo: tomamos las imágenes de prueba y les tapamos completamente el 90 % central, dejando visible **únicamente un anillo exterior del 10 % del borde**. 

En ese anillo perimetral no había hoja, no había tejido vegetal y no había ninguna lesión patológica; solo se veía el fondo, el cielo o el marco de la fotografía.

El resultado fue contundente: **un modelo entrenado únicamente con ese marco exterior acertó el 78.3 % de las imágenes de prueba**, frente al 11.1 % que daría el azar. 

El modelo era capaz de clasificar casi ocho de cada diez fotos sin haber visto jamás la hoja. Esto demostró que cuando la partición de datos reparte las mismas fuentes entre entrenamiento y prueba de forma aleatoria, la red simplemente memoriza el contexto fotográfico. Por tanto, reportar un 95 % de F1 bajo esa partición no era una medida honesta de diagnóstico agronómico.

---

## La partición honesta (Leave-One-Source-Out)

Para medir la capacidad real de generalización, cambiamos radicalmente la forma de evaluar: implementamos una validación dejando fuentes completas fuera (*Leave-One-Source-Out*). 

En cada iteración, el modelo se entrena con varias fuentes y se evalúa sobre una fuente retenida que jamás vio durante el entrenamiento. Es la prueba definitiva de cómo se comportaría la aplicación si la llevamos a una parcela con condiciones que el sistema nunca ha conocido.

| Protocolo de evaluación | Macro $F_1$-Score | Exactitud (Accuracy) | Qué mide en realidad |
|---|:---:|:---:|---|
| **Partición aleatoria estándar** | **0.8411** | **91.15 %** | Rendimiento memorizando la mezcla de fuentes conocidas. |
| **Evaluación fuera de fuente (honesta)** | **`0.5573`** | **`0.6884`** | **Generalización real frente a cámaras y campos nuevos.** |

La caída de casi 28 puntos en Macro F1 no es un error de programación: es la medida exacta de la **brecha de dominio** en la agricultura digital.

Las patologías con muchas fotos repartidas en varias fuentes (como hojas sanas, necrosis letal o roya común) lograron retener más del 80 % de su rendimiento. Pero las clases con menos imágenes y concentradas en una o dos fuentes (como las deficiencias nutricionales y la mancha gris) sufrieron caídas pronunciadas al ser evaluadas en un entorno desconocido.

---

## El techo de los algoritmos y el camino hacia adelante

Durante semanas exploramos si este sesgo podía solucionarse mediante técnicas algorítmicas: probamos balanceo de fuentes, sustitución artificial de fondos (*BackMix*), ecualizaciones agresivas y recorte de contornos. Ninguna intervención de software logró cerrar la brecha de manera significativa.

La razón es simple y contundente: **ninguno de los 14 datasets públicos disponibles fue capturado en El Salvador ni en Centroamérica**. 

No existe ningún truco matemático que pueda reemplazar la diversidad biológica real. La única solución honesta para que DoctorMaiz sea infalible en el campo salvadoreño es alimentar el modelo con fotografías tomadas en las milpas locales, bajo el sol local y con las variedades locales de maíz. 

Por eso este hallazgo no representó un fracaso, sino el pilar conceptual más valioso de la Etapa 2: justificó el desarrollo del marco guía en la cámara móvil para neutralizar los fondos engañosos y dio sentido al módulo de contribución comunitaria de la aplicación.
