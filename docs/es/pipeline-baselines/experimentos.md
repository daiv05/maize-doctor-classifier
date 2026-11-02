# Experimentos

Elegir las tres arquitecturas del baseline canónico no fue una decisión a ciegas. Antes de comprometer cómputo en el entrenamiento completo de 9 clases, se hizo una **exploración de arquitecturas** más barata sobre un subconjunto reducido de **4 clases** (`healthy`, `common_rust`, `fall_armyworm`, `nitrogen_deficiency`), probada en dos regímenes de datos: un **cap de 500 imágenes por clase**, pensado como comparación rápida y barata y a la vez como sonda de eficiencia muestral, y una corrida **sin límite por clase**, con las mismas 4 clases pero usando todas las imágenes disponibles.

El objetivo fue comparar el comportamiento de cada arquitectura (convergencia, colapso de clases, sensibilidad al volumen de datos) a bajo costo, no medir el rendimiento final.

## Arquitecturas exploradas

Con ese objetivo se probó un abanico amplio antes de reducirlo a los tres definitivos: hasta **8 arquitecturas ligeras**, `efficientnet_b0`, `efficientnet_b4`, `efficientnet_lite0`, `fastvit_t8`, `ghostnetv2_100`, `mobilenet_v3_large`, `mobilenet_v3_small` y `shufflenet_v2_x1_0`.

De esta exploración se adoptaron como baselines definitivos `efficientnet_b0`, `shufflenet_v2_x1_0` y `efficientnet_lite0` (ver [Baselines](../baselines/)).

## Resultados completos

Para no perder el detalle de estas corridas exploratorias sin sobrecargar esta página, los outputs completos (métricas, matrices de confusión, reportes LIME) se archivan por separado.

Resultados de experimentos (4 clases): [ZIP en Google Drive](https://drive.google.com/drive/folders/1lHWjZ04V3__OROMNqs38fPD5sd3t8laI?usp=sharing)
