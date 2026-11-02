# Ensamble multimodelo por voto suave

El ensamble promedia las distribuciones softmax de `EfficientNet-Lite0`, `EfficientNet-B0` y `ShuffleNet-V2-x1.0` y emite la clase de mayor probabilidad promedio. Los pesos son uniformes y no se reajustan con datos de evaluación.

$$P_{\text{ensamble}}(y \mid x) = \frac{1}{3} \sum_{m=1}^{3} \text{softmax}(z_m(x))$$

Las tres arquitecturas difieren en diseño (atención de canales en B0, convoluciones simplificadas y cuantizables en Lite0, barajado de canales en ShuffleNet), de modo que sus errores no están completamente correlacionados.

## Resultados

Sobre el test estratificado de la etapa 2 el ensamble alcanza Macro-F1 0.9567 y accuracy 98.29 %, frente a 0.9483 del mejor miembro individual (B0). Fuera de dominio la ventaja se reduce y no es estable: sobre `multicrop-disease-maiz` sube +0.0045 de Macro-F1 en promedio (positiva en dos de tres semillas) y duplica el ECE. Las cifras y su evidencia están en [Resultados](/es/resultados/) y [Generalización entre fuentes](/es/resultados/generalizacion).

## Uso en el proyecto

La app despliega un único modelo, `EfficientNet-Lite0` cuantizado a int8, por tamaño (3.74 MB) y latencia. El ensamble queda como evaluación de servidor y se reconstruye desde los checkpoints respaldados.

```bash
make evaluate-ensemble MODELS=efficientnet_lite0,efficientnet_b0,shufflenet_v2_x1_0
```
