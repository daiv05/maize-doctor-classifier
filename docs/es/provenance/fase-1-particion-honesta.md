# Fase 1 — La Partición Honesta (Leave-One-Source-Out)

La partición aleatoria convencional crea una falsa ilusión de seguridad: al mezclar las mismas fuentes en el entrenamiento y en la prueba, el modelo rinde con un 95 % de acierto porque examina imágenes tomadas bajo las mismas condiciones que ya aprendió.

Para medir qué tan bien generaliza el sistema frente a condiciones nunca antes vistas, implementamos una **validación dejando fuentes completas fuera (*Leave-One-Source-Out*, LOSO)**. Evaluamos 11 pliegues de procedencia sobre más de 33,200 imágenes, de modo que cada fotografía fue evaluada exactamente una vez, en el pliegue donde su repositorio de origen quedó completamente excluido del entrenamiento.

---

## Resultados globales: la caída que revela la realidad

La comparación entre la partición aleatoria estándar y la evaluación fuera de fuente puso en evidencia el verdadero impacto de la brecha de dominio:

| Métrica | Partición aleatoria estándar | Evaluación fuera de fuente (honesta) | Diferencia ($\Delta$) |
|---|:---:|:---:|:---:|
| **Macro $F_1$-Score** | **0.8411** | **`0.5573`** | **−0.2838** |
| **Exactitud (Accuracy)** | **91.15 %** | **`68.84 %`** | **−22.31 pp** |
| **Muestras evaluadas** | 5,015 | **33,268** | — |

La caída de casi 28 puntos en Macro F1 no es un retroceso del modelo, sino la medida rigurosa de qué tanto depende una red convolucional de los fondos y cámaras que ya conoce.

---

## Comportamiento desglosado por clase

No todas las patologías sufrieron la misma caída. Al examinar el rendimiento clase por clase, se observa con claridad qué factores protegen al modelo frente al cambio de entorno:

| Patología / Clase | $F_1$ en partición aleatoria | $F_1$ fuera de fuente | Retención de rendimiento | Diagnóstico de estabilidad |
|---|:---:|:---:|:---:|---|
| **Planta sana (`healthy`)** | 0.9468 | **0.8092** | 85.5 % | **Sostenida:** alta diversidad de fuentes amortigua el cambio. |
| **Necrosis letal (MLN)** | 0.9795 | **0.7912** | 80.8 % | **Sostenida:** síntomas severos y patrón visual muy marcado. |
| **Roya común** | 0.9682 | **0.7804** | 80.6 % | **Sostenida:** pústulas anaranjadas inconfundibles. |
| **Tizón foliar (NCLB)** | 0.8947 | **0.7171** | 80.2 % | **Sostenida:** lesiones elípticas grandes bien reconocibles. |
| **Gusano cogollero** | 0.9054 | **0.6194** | 68.4 % | **Moderada:** daño masticador variable según la edad foliar. |
| **Deficiencia de nitrógeno** | 0.7154 | **0.4190** | 58.6 % | **Sensible:** clorosis difusa dependiente de la iluminación. |
| **Mancha gris (GLS)** | 0.8376 | **0.3972** | 47.4 % | **Vulnerable:** lesiones rectangulares finas que se confunden sin contexto. |
| **Deficiencia de fósforo** | 0.7390 | **0.2861** | 38.7 % | **Vulnerable:** tonos violáceos muy sensibles al sensor de la cámara. |
| **Deficiencia de potasio** | 0.5832 | **0.1958** | 33.6 % | **Crítica:** apenas 620 fotos repartidas en orígenes dispares. |

---

## Conclusiones de la evaluación fuera de fuente

El patrón que emerge de esta tabla es contundente:

1. **Las clases abundantes y multi-fuente resisten:** Las cuatro enfermedades con mayor soporte y repartidas en más de tres repositorios independientes retienen más del 80 % de su efectividad cuando se les presenta una cámara nueva.
2. **Las deficiencias nutricionales son las más vulnerables:** Al contar con pocas fotografías y depender de sutiles variaciones de coloración (amarillamientos y bordes violáceos), cambiar de cámara altera los histogramas de color y desorienta al clasificador.
3. **El camino hacia adelante es la recolección local:** Estos números demostraron que la única forma de garantizar diagnósticos confiables en El Salvador es incorporar fotografías tomadas en parcelas locales, alimentando directamente a las clases más vulnerables con imágenes de campo real.
