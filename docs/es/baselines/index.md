# Baselines

Antes de invertir tiempo y cómputo en el entrenamiento completo, conviene saber qué tan bien funcionan las arquitecturas candidatas mediante experimentos. Para eso existen los baselines de este proyecto.

Funcionan como demo de modelos candidatos, ya que antes de comprometer el entrenamiento completo permiten observar el comportamiento inicial de cada arquitectura sobre una fracción representativa del dataset y detectar problemas (colapso de clases, overfitting temprano, incompatibilidad con el pipeline) a bajo costo.

Los tres modelos elegidos, **EfficientNet-B0**, **ShuffleNetV2-x1.0** y **EfficientNet-Lite0**, son redes convolucionales ligeras pre-entrenadas en ImageNet que cubren el eje precisión - eficiencia y tiene compatibilidad para convertirse a TFLite para el despliegue móvil offline que es nuestra meta final.

## Dataset a utilizar

Los baselines se entrenan sobre el **perfil `baseline`** de la configuración del pipeline: las **9 clases** del dataset actual con un tope de **1 500 imágenes por clase**, solo se recortan las clases mayoritarias, las minoritarias quedan intactas. Esto permite entrenar los modelos candidatos con un **dataset reducido (10,020 imágenes)** que conserva el desbalance natural de clases.

| Split | Imágenes |
|---|---:|
| Entrenamiento (`train.csv`, 70 %) | 7 014 |
| Validación (`val.csv`, 15 %) | 1 503 |
| Prueba (`test.csv`, 15 %) | 1 503 |
| **Total** | **10 020** |

Se conservan completas las clases minoritarias (potasio 266, nitrógeno 523, fósforo 612) y
limita solo las mayoritarias (healthy, tizones, gusano cogollero). El cap es configurable para permitir experimentar con diferentes tamaños de dataset, pero el valor por defecto es 1 500 imágenes por clase.

::: warning Corridas de la primera etapa
Las cifras y métricas de esta sección corresponden a corridas sobre **31 622 imágenes**, previas a la ampliación. En la materialización vigente de 33 429 elegibles, potasio 621, nitrógeno 846 y fósforo 938 siguen por debajo del tope de 1 500; el total y las métricas cambiarían si se regenerara `splits-baseline`.
:::

## Modelos seleccionados

A continuación se describe cada uno de los tres modelos a evaluar: qué los distingue, por qué se eligieron y cómo se comportan en este dataset. El orden va de menor a mayor tamaño, empezando por el más ligero del grupo.

### ShuffleNetV2-x1.0

**ShuffleNetV2-x1.0** es una CNN diseñada por Megvii (Face++) en 2018 explícitamente para
inferencia eficiente en dispositivos móviles <sup>[[17]](#ref-17)</sup>. Su contribución es un
conjunto de guías prácticas de diseño, que no se limitan a minimizar FLOPs sino que también
atienden al costo real de memoria y acceso, materializadas en dos operaciones. La primera es
**channel split + channel shuffle**, que divide los canales en dos ramas y, tras procesarlas,
los baraja para que la información fluya entre grupos sin convoluciones densas costosas. La
segunda es evitar las convoluciones agrupadas 1x1, esquivando así el cuello de botella de acceso
a memoria (MAC) que penalizaba a ShuffleNetV1 y priorizando velocidad real sobre FLOPs teóricos.

En ImageNet-1K alcanza ~69 % de Top-1 con solo ~2.3 M de parámetros, siendo uno de los modelos
más pequeños del grupo (~5 MB serializado).

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | Inferior a EfficientNet-B0 en ImageNet, pero competitiva tras fine-tuning en el dataset de maíz (macro-F1 0.9030 en 9 clases) |
| Velocidad / tamaño | El más ligero del grupo (~5 MB, ~2.3 M params); ideal para inferencia en gama baja |
| Transfer learning | Pre-entrenado con `ShuffleNet_V2_X1_0_Weights.DEFAULT`; se reemplaza la capa `fc` final |
| Despliegue | Operaciones (channel shuffle, depthwise) soportadas por TFLite; convierte y cuantiza sin ops exóticas |
| Riesgo | Capacidad limitada en clases visualmente ambiguas (deficiencias N/P/K), donde todos los baselines sufren |

Se construye con `torchvision` reemplazando `model.fc` por una `nn.Linear(in_features, 9)`.

### EfficientNet-B0

Si ShuffleNetV2 prioriza la eficiencia por encima de todo, EfficientNet-B0 busca el otro extremo del equilibrio: la mejor precisión posible sin disparar el costo computacional. **EfficientNet-B0** es la red base de la familia EfficientNet, propuesta por Google Brain en 2019. Su contribución central es el *compound scaling*: en lugar de escalar solo la profundidad, el ancho o la resolución de entrada de forma independiente (como hacía la práctica anterior), EfficientNet escala los tres simultáneamente con un coeficiente compuesto $\phi$ determinado por búsqueda de arquitectura (NAS) <sup>[[8]](#ref-8)</sup>.

La versión B0 es el punto de partida de la familia: la arquitectura base encontrada por NAS antes de aplicar cualquier escala adicional. Usa bloques **MBConv** (Mobile Inverted Bottleneck) <sup>[[6]](#ref-6)</sup>, que combinan conexiones residuales, expansión de canales seguida de proyección, y un módulo de Squeeze-and-Excitation integrado en cada bloque <sup>[[9]](#ref-9)</sup>.

En ImageNet-1K alcanza ~77.1 % de Top-1 con 5.3 M de parámetros: más del doble que ShuffleNetV2-x1.0, pero con mayor precisión.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | Superior a ShuffleNetV2-x1.0 en ImageNet (~77 % vs. ~69 %); también lidera tras fine-tuning (macro-F1 0.9146 vs. 0.9030) |
| Velocidad de inferencia | Más lento que ShuffleNetV2-x1.0 en hardware móvil por las operaciones SE en cada bloque y el mayor número de parámetros |
| Tamaño del modelo | ~16 MB serializado; más pesado que ShuffleNetV2-x1.0 (~5 MB) |
| Transfer learning | Pre-entrenado en ImageNet con `EfficientNet_B0_Weights.DEFAULT` (IMAGENET1K_V1); se reemplaza `model.classifier[1]` |
| Regularización implícita | Los bloques MBConv con dropout estructural hacen a EfficientNet-B0 más robusto al overfitting con datasets pequeños |

### EfficientNet-Lite0

El tercer baseline es un intento de quedarse con lo mejor de los dos mundos anteriores: la precisión de EfficientNet, pero adaptada para que funcione bien una vez cuantizada, algo crítico para el despliegue final del proyecto. **EfficientNet-Lite0** es una variante de EfficientNet-B0 optimizada específicamente para dispositivos de borde con aceleradores de inferencia (Coral Edge TPU, microcontroladores ARM con CMSIS-NN) <sup>[[13]](#ref-13)</sup>.

Las diferencias con B0 son tres. Primero, se eliminan los bloques Squeeze-and-Excitation, porque su operación de reducción global no se mapea eficientemente en aceleradores de inferencia cuantizados. Segundo, se usa ReLU6 en lugar de Swish, más compatible con cuantización INT8, ya que Swish introduce errores de representación no triviales que hacen caer la precisión de ~75 % a ~46 % si no se sustituye <sup>[[13]](#ref-13)</sup>. Tercero, la red evita el stem strided convolution, esquivando así algunas operaciones que rompen la compatibilidad con ciertos compiladores de modelos (TFLite, ONNX para Edge TPU).

Se construye con `timm` (`timm.create_model("efficientnet_lite0", pretrained=True, num_classes=9)`) porque `torchvision` no incluye esta variante.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | ~1–2 pp inferior a EfficientNet-B0 en ImageNet (~74–75 % Top-1); compensado por facilidad de despliegue |
| Velocidad de inferencia | Similar o superior a ShuffleNetV2-x1.0 en aceleradores compatibles; sin ventaja clara en GPU estándar |
| Cuantización | Diseñada para cuantizarse a INT8 sin degradación significativa; punto fuerte para despliegue en campo |
| Dependencia adicional | Requiere `timm` (no incluida en `torchvision`); añade una dependencia al entorno |
| Uso en proyecto | Representa el extremo del trade-off "máxima eficiencia en edge" para comparar contra el extremo "máxima precisión" de EfficientNet-B0 |

## Comparación de los tres modelos

Con los tres modelos ya descritos individualmente, esta tabla los pone lado a lado para facilitar la comparación directa de tamaño y precisión en ImageNet:

| Modelo | Parámetros | Top-1 ImageNet | Tamaño (~) | Apto para TFLite/edge |
|---|---:|---:|---:|---|
| `efficientnet_b0` | 5.3 M | ~77.1 % | 16 MB | Sí (float16; INT8 aceptable) |
| `shufflenet_v2_x1_0` | 2.3 M | ~69.4 % | 5 MB | Sí (mobile-native) |
| `efficientnet_lite0` | 4.7 M | ~74.9 % | 14 MB | Sí (diseñado para INT8) |

Los tres parten de pesos pre-entrenados en ImageNet <sup>[[16]](#ref-16)</sup> y se ajustan sobre el dataset de maíz con el mismo pipeline de preprocesado, estratificación y data augmentation. La diferencia entre ellos es la arquitectura y el tamaño del modelo.

---

## Referencias

<a id="ref-6"></a>[6] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 4510–4520.

<a id="ref-8"></a>[8] M. Tan y Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 6105–6114.

<a id="ref-9"></a>[9] J. Hu, L. Shen, y G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 7132–7141.

<a id="ref-12"></a>[12] T.-Y. Lin, P. Goyal, R. Girshick, K. He, y P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, 2017, pp. 2980–2988.

<a id="ref-13"></a>[13] Google Brain / TensorFlow Team, "Higher Accuracy on Vision Models with EfficientNet-Lite," *TensorFlow Blog*, Mar. 2020. [Online]. Available: https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html

<a id="ref-16"></a>[16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, y L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Miami, FL, USA, 2009, pp. 248–255.

<a id="ref-17"></a>[17] N. Ma, X. Zhang, H.-T. Zheng, y J. Sun, "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Munich, Germany, 2018, pp. 116–131.
