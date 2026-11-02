# Deep Learning

Esta sección presenta los fundamentos teóricos del **aprendizaje profundo** aplicados a la clasificación de imágenes. El énfasis está en cómo una red neuronal aprende representaciones visuales a partir de píxeles, cómo las CNN extraen patrones espaciales y por qué arquitecturas como LeNet, AlexNet, VGG, ResNet, MobileNet y EfficientNet marcaron etapas importantes en la evolución del área.

## Aprendizaje Profundo y Redes Neuronales

El **aprendizaje profundo** (*deep learning*) es una rama del aprendizaje automático basada en redes neuronales con múltiples capas. Forma parte del campo más amplio de **Machine Learning**, donde el objetivo es aprender patrones a partir de datos en lugar de programar reglas explícitas. Su diferencia principal frente a enfoques más tradicionales es que puede aprender representaciones jerárquicas directamente desde los datos: las capas tempranas capturan patrones simples y las capas posteriores combinan esos patrones en conceptos más complejos <sup>[[1]](#ref-1)</sup>.

Una **red neuronal artificial** está formada por unidades conectadas entre sí. Cada unidad recibe valores de entrada, los combina mediante **pesos** y un **sesgo**, y aplica una función de activación:

$$z = w^\top x + b, \quad a = \phi(z)$$

En esta ecuación, $x$ representa las entradas de la neurona, $w$ sus pesos, $b$ el sesgo, $z$ la combinación lineal y $a$ la activación resultante. Los pesos determinan qué señales son relevantes, el sesgo desplaza la respuesta de la unidad y la activación $\phi$ introduce no linealidad. Sin activaciones no lineales, muchas capas apiladas se comportarían como una sola transformación lineal y no podrían modelar relaciones visuales complejas.

Las neuronas se organizan en **capas**. La **capa de entrada** recibe los datos originales, como los valores de píxeles de una imagen. Las **capas ocultas** transforman progresivamente esa información y aprenden representaciones internas. La **capa de salida** produce los valores finales usados para clasificar o estimar una respuesta. En una tarea de imágenes, una capa inicial puede responder a cambios bruscos de intensidad, como bordes de una hoja, capas intermedias pueden responder a texturas, nervaduras o manchas, y capas profundas pueden combinar esas señales para separar clases visualmente parecidas.

### Propagación, Pérdida y Optimización

La **propagación hacia adelante** (*forward propagation*) es el proceso mediante el cual la entrada atraviesa la red capa por capa hasta producir una salida. Cada capa aplica sus pesos, sesgos y activaciones sobre la representación recibida de la capa anterior.

Durante el entrenamiento, la red produce una predicción y se compara con la etiqueta real mediante una **función de pérdida**. En clasificación multiclase se usa con frecuencia la entropía cruzada:

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{p}_c)$$

Aquí, $\mathcal{L}$ es la pérdida, $C$ representa el total de categorías consideradas por la tarea, $y_c$ indica si la clase $c$ es la correcta y $\hat{p}_c$ es la probabilidad predicha para esa clase. Si la red asigna baja probabilidad a la clase correcta, la pérdida aumenta.

La **retropropagación** (*backpropagation*) calcula cómo cambia la pérdida respecto a cada peso de la red, aplicando la regla de la cadena desde la salida hacia las capas iniciales. Luego, un optimizador actualiza los pesos mediante variantes del **descenso de gradiente**:

$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$$

En esta expresión, $\eta$ es la tasa de aprendizaje y $\frac{\partial \mathcal{L}}{\partial w}$ indica la dirección en la que el peso aumenta la pérdida. Restar ese término ajusta el peso en la dirección que tiende a reducir el error. Así, la red modifica gradualmente sus filtros y combinaciones internas para producir predicciones más consistentes <sup>[[2]](#ref-2)</sup>.

## Imágenes como Tensores

Una imagen digital puede verse como una cuadrícula de **píxeles** con **alto** y **ancho**. Si una imagen tiene alto $H$ y ancho $W$, puede representarse como una matriz de $H \times W$ valores cuando tiene un solo canal. En imágenes RGB, cada píxel tiene tres canales: rojo (*red*), verde (*green*) y azul (*blue*). Por eso, una imagen a color se representa como un tensor de forma:

$$H \times W \times 3$$

Esta notación indica que existen tres matrices, una por canal de color, alineadas espacialmente. Cada canal contiene intensidades numéricas. Al combinarse, estas intensidades describen colores, bordes, sombras y texturas. En hojas de maíz, por ejemplo, los píxeles pueden codificar variaciones de verde, amarillamiento, bordes secos, manchas oscuras o lesiones alargadas. Para una red neuronal, todos esos elementos son patrones numéricos distribuidos en el tensor de entrada.

La estructura espacial importa, porque píxeles vecinos suelen estar relacionados. Una mancha foliar no se reconoce por un píxel aislado, sino por una región con color, textura, borde y forma.

## Redes Neuronales Convolucionales (CNN)

Una **red neuronal convolucional** (*Convolutional Neural Network*, CNN) es una arquitectura de aprendizaje profundo diseñada para procesar datos con estructura de cuadrícula, como imágenes <sup>[[3]](#ref-3)</sup>. A diferencia de una red completamente conectada, una CNN no conecta cada píxel con cada neurona desde el inicio, en su lugar, usa filtros pequeños que recorren la imagen y detectan patrones locales.

Dentro de las CNNs, un **filtro** es una pequeña matriz de pesos aprendibles, por ejemplo de $3 \times 3$ o $5 \times 5$. Al aplicarse sobre distintas regiones de la imagen mediante una **convolución**, produce un **mapa de características** (*feature map*) que indica dónde aparece el patrón aprendido.

Una CNN aprende una **jerarquía visual**. Las primeras capas suelen responder a **bordes**, cambios de orientación y colores básicos, luego las capas intermedias combinan esas señales para detectar **texturas**, patrones repetidos, manchas o transiciones de color. Las capas más profundas integran formas, partes de objetos y patrones visuales complejos. En una hoja de maíz, esta progresión puede ir desde bordes de la lámina foliar, variaciones de verde o amarillo y textura de nervaduras, hasta regiones lesionadas con forma y color característicos.

La convolución tiene dos ventajas:

- **Conectividad local:** cada filtro observa regiones pequeñas, adecuadas para patrones visuales cercanos.
- **Compartición de pesos:** el mismo filtro se aplica en toda la imagen, lo que reduce parámetros y permite detectar el mismo patrón en distintas posiciones.

El **campo receptivo** de una neurona es la región de la imagen original que puede influir en su activación. En las primeras capas, ese campo suele ser pequeño, al apilar convoluciones y operaciones de reducción espacial, las capas profundas integran información de regiones cada vez más amplias. Esto permite pasar de señales locales, como un borde oscuro, a patrones más complejos, como una lesión formada por borde, color y textura.

### ReLU y Pooling

Después de una convolución se suele aplicar una activación como **ReLU** (*Rectified Linear Unit*):

$$\text{ReLU}(x) = \max(0, x)$$

ReLU mantiene las activaciones positivas y anula las negativas, lo que ayuda a entrenar redes profundas de forma eficiente y favorece representaciones dispersas <sup>[[15]](#ref-15)</sup>.

Las capas de **pooling** reducen la resolución espacial de los mapas de características. El caso más común, *max pooling*, toma el valor máximo dentro de una pequeña ventana, esto disminuye el costo computacional y aporta cierta tolerancia a desplazamientos pequeños: una lesión o textura puede seguir siendo reconocible aunque aparezca ligeramente movida dentro de la imagen.

## Clasificación: Logits y Softmax

Después de las capas convolucionales, muchas arquitecturas usan **capas completamente conectadas** (*fully connected*) o una capa lineal final para combinar las características extraídas. Estas capas ya no observan solo una región local: reciben una representación resumida de la imagen y la transforman en valores asociados a las clases posibles.

Esos valores se llaman **logits**. Cada logit corresponde a una clase posible, pero todavía no es una probabilidad. Para convertirlos en una distribución sobre clases se aplica **softmax**:

$$\hat{p}_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

donde $z_c$ es el logit de la clase $c$ y el denominador suma la evidencia de todas las clases. La clase predicha suele ser la que obtiene la probabilidad más alta. Esta salida permite interpretar la decisión como una competencia entre clases: la red asigna mayor peso a la clase cuyas características visuales resultan más compatibles con la imagen.

La interpretación de por qué una red asigna una clase a una imagen se aborda en la página complementaria de [interpretabilidad y explicabilidad](interpretability.md), sin duplicar aquí sus métodos específicos.

## Sobreajuste, Regularización y Transferencia

El **sobreajuste** (*overfitting*) ocurre cuando una red aprende demasiado bien los ejemplos de entrenamiento, pero no generaliza igual de bien a imágenes nuevas. En visión por computadora esto puede pasar si el modelo memoriza fondos, condiciones de iluminación, encuadres o detalles accidentales en lugar de aprender patrones visuales relevantes.

La **regularización** agrupa técnicas que buscan reducir ese riesgo y mejorar la generalización del modelo <sup>[[2]](#ref-2)</sup>. Algunas estrategias habituales son:

- **Dropout:** durante el entrenamiento, desactiva aleatoriamente una fracción de unidades para evitar que la red dependa demasiado de combinaciones específicas de neuronas <sup>[[4]](#ref-4)</sup>.
- **Batch normalization:** normaliza activaciones dentro de mini-lotes y aprende parámetros de escala y desplazamiento. Esto ayuda a estabilizar el entrenamiento de redes profundas y puede tener un efecto regularizador <sup>[[14]](#ref-14)</sup>.
- **Data augmentation:** genera variaciones de las imágenes de entrenamiento mediante transformaciones como recortes, rotaciones moderadas, cambios de brillo o volteos cuando son coherentes con el dominio. Esto expone al modelo a más variabilidad visual sin cambiar la etiqueta <sup>[[2]](#ref-2)</sup>.
- **Transfer learning:** reutiliza pesos de un modelo preentrenado en una tarea fuente y los adapta a una tarea objetivo. En imágenes, los modelos preentrenados suelen aprender filtros tempranos útiles para bordes, colores y texturas, que pueden transferirse a dominios visuales relacionados <sup>[[5]](#ref-5)</sup>.

El aprendizaje por transferencia es especialmente útil cuando no se dispone de millones de imágenes etiquetadas. La idea no es copiar la tarea original, sino aprovechar representaciones visuales generales y ajustarlas al nuevo problema.

## Evolución de Arquitecturas CNN

### LeNet

**LeNet-5**, propuesta por LeCun et al. en 1998, fue una de las primeras CNN exitosas para reconocimiento de dígitos y documentos <sup>[[6]](#ref-6)</sup>. Combinaba convoluciones, *pooling* y capas finales de clasificación. Aunque era pequeña comparada con arquitecturas modernas, estableció una plantilla fundamental para procesar imágenes mediante extracción jerárquica de características.

### AlexNet

**AlexNet** marcó un punto de inflexión en 2012 al ganar ImageNet con una CNN profunda entrenada en GPU <sup>[[7]](#ref-7)</sup>. Popularizó el uso de ReLU, *dropout* y aumento de datos en redes convolucionales grandes. Su éxito mostró que las CNN podían escalar a bases de imágenes naturales mucho más complejas que los dígitos manuscritos.

### VGG

**VGG** mostró que aumentar la profundidad usando bloques simples de convoluciones pequeñas de $3 \times 3$ podía producir representaciones visuales muy efectivas <sup>[[8]](#ref-8)</sup>. Su diseño era regular y fácil de entender, aunque con un costo alto en parámetros y memoria.

### ResNet

**ResNet** introdujo conexiones residuales o *skip connections*, que suman la entrada de un bloque con su salida <sup>[[9]](#ref-9)</sup>:

$$x_{l+1} = \mathcal{F}(x_l, \{W_l\}) + x_l$$

Esta formulación facilita el flujo de gradientes y permite entrenar redes mucho más profundas sin que el aumento de capas degrade el entrenamiento. Las conexiones residuales se volvieron una idea central en muchas arquitecturas posteriores.

### MobileNet

**MobileNet** fue diseñada para redes eficientes en dispositivos con recursos limitados <sup>[[10]](#ref-10)</sup>. Su pieza clave son las **convoluciones depthwise-separable**, que factorizan una convolución estándar en dos pasos: primero una convolución independiente por canal (*depthwise*) y luego una convolución $1 \times 1$ que combina canales (*pointwise*). Esta separación reduce mucho el costo computacional (alrededor de 8 a 9 veces menos operaciones con filtros de $3 \times 3$) manteniendo la capacidad para aprender patrones visuales útiles. Por eso es un buen punto de partida cuando la meta es correr el modelo en un teléfono.

La familia fue evolucionando versión a versión, cada una con una idea central:

| Versión | Año | Idea principal |
|---|---|---|
| MobileNetV1 <sup>[[10]](#ref-10)</sup> | 2017 | Convoluciones depthwise-separable |
| MobileNetV2 <sup>[[11]](#ref-11)</sup> | 2018 | Bloques con residuos invertidos y cuellos de botella lineales |
| MobileNetV3 <sup>[[12]](#ref-12)</sup> | 2019 | Búsqueda de arquitectura, bloques de atención y activaciones más eficientes |

La idea de los **residuos invertidos** es expandir primero los canales, aplicar la convolución depthwise en ese espacio más amplio y luego proyectar de vuelta a un espacio pequeño, todo sin perder información por el camino. Esto se hereda en otras arquitecturas ligeras.

### EfficientNet

**EfficientNet** propuso una idea simple pero potente: en lugar de agrandar la red por un solo lado (más capas, o más canales, o imágenes más grandes), conviene escalar las tres dimensiones a la vez y de forma equilibrada. A esto lo llamaron *compound scaling* <sup>[[13]](#ref-13)</sup>:

$$\text{depth} \propto \alpha^\phi, \quad \text{width} \propto \beta^\phi, \quad \text{resolution} \propto \gamma^\phi$$

Aquí $\phi$ es un único coeficiente que controla cuánto crece la red, y los valores $\alpha$, $\beta$ y $\gamma$ reparten ese crecimiento entre profundidad, ancho y resolución. Partiendo de un modelo base pequeño (EfficientNet-B0, con unos 5.3 millones de parámetros y cerca de 77 % de precisión en ImageNet) y subiendo $\phi$, se obtiene toda la familia B1 a B7.

Lo interesante es que esta receta consigue una muy buena relación entre precisión y costo, que es justo lo que buscamos para un despliegue ligero.

## Manejo del Desbalance de Clases

En muchos conjuntos de datos reales, algunas clases tienen muchas más imágenes que otras. En nuestro caso el desbalance es fuerte: la clase más frecuente supera por decenas de veces a la menos frecuente. El riesgo es que, sin ninguna intervención, la red aprenda el atajo fácil de predecir casi siempre las clases mayoritarias. Así lograría una *accuracy* alta pero fallaría justo donde más importa: en las clases minoritarias, que suelen ser las de diagnóstico más delicado.

Hay tres herramientas habituales para lidiar con esto, y se pueden combinar entre sí.

### Función de pérdida con pesos de clase

La idea es hacer que los errores en las clases poco frecuentes "cuesten" más. La **entropía cruzada ponderada** asigna a cada clase $c$ un peso $w_c$ inversamente proporcional a su frecuencia:

$$w_c = \frac{N}{C \cdot n_c}$$

donde $N$ es el total de imágenes, $C$ el número de clases y $n_c$ cuántas imágenes tiene la clase $c$. Con esos pesos, la pérdida queda:

$$\mathcal{L} = -\sum_{c=1}^{C} w_c \, y_c \log \hat{p}_c$$

Así, equivocarse en una clase minoritaria pesa más en el gradiente, y la red se ve empujada a aprender también sus patrones.

### Muestreo ponderado

La pérdida ponderada actúa sobre el gradiente, pero también se puede intervenir antes, al armar cada lote de imágenes. Un **muestreador ponderado** elige los ejemplos de cada mini-lote con una probabilidad mayor para las clases poco frecuentes. Puesto de forma intuitiva: si una clase representa apenas el 2 % del conjunto, el muestreador la seleccionará mucho más seguido, de modo que aparezca en casi todos los lotes y la red reciba suficiente señal para aprenderla. Lo bueno es que consigue este equilibrio sin duplicar imágenes en memoria.

### Pérdida focal

Cuando la ponderación por frecuencia no alcanza para las clases más difíciles, la **pérdida focal** (*focal loss*) ofrece otra vía <sup>[[16]](#ref-16)</sup>. Reescala la entropía cruzada para restarle peso a los ejemplos que la red ya clasifica bien, y concentrar el aprendizaje en los ejemplos difíciles o mal clasificados, que suelen ser justamente los de las clases minoritarias.

## Métricas de Evaluación

Una vez que aplicamos técnicas para el desbalance, queda la pregunta clave: ¿cómo sabemos si el modelo de verdad aprendió a distinguir las clases minoritarias y no solo las mayoritarias? Para eso conviene mirar más allá de la *accuracy*.

### Precisión, recall y F1 por clase

Para cada clase $c$ se calculan tres métricas a partir de sus aciertos y errores:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

La **precisión** mide qué tan confiables son las predicciones de esa clase, el **recall** mide cuántos casos reales de la clase logró detectar, y el **F1** combina ambas en un solo número.

### F1-macro frente a F1-weighted

Cuando queremos un número global, hay dos formas de promediar el F1 de todas las clases. El **F1-macro** las trata a todas por igual, sin importar cuántas imágenes tenga cada una:

$$F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

El **F1-weighted**, en cambio, promedia dando más peso a las clases con más imágenes, por lo que tiende a reflejar sobre todo el rendimiento en las mayoritarias.

Con un desbalance fuerte como el nuestro, el **F1-macro** es la métrica más informativa, porque penaliza por igual el fallo en cualquier clase, incluidas las minoritarias más difíciles de diagnosticar. Por eso el proyecto fija un umbral mínimo sobre el F1-macro del conjunto de prueba como criterio de viabilidad, en lugar de usar la *accuracy*: así exigimos un buen rendimiento en todas las clases y no un promedio inflado por las mayoritarias.

## Referencias

<a id="ref-1"></a>[1] Y. LeCun, Y. Bengio, y G. Hinton, "Deep Learning," *Nature*, vol. 521, pp. 436-444, 2015.

<a id="ref-2"></a>[2] I. Goodfellow, Y. Bengio, y A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.

<a id="ref-3"></a>[3] R. Yamashita, M. Nishio, R. K. G. Do, y K. Togashi, "Convolutional Neural Networks: An Overview and Application in Radiology," *Insights Imaging*, vol. 9, no. 4, pp. 611-629, 2018.

<a id="ref-4"></a>[4] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, y R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 1929-1958, 2014.

<a id="ref-5"></a>[5] S. J. Pan y Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345-1359, 2010.

<a id="ref-6"></a>[6] Y. LeCun, L. Bottou, Y. Bengio, y P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278-2324, 1998.

<a id="ref-7"></a>[7] A. Krizhevsky, I. Sutskever, y G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, 2012, pp. 1097-1105.

<a id="ref-8"></a>[8] K. Simonyan y A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in *International Conference on Learning Representations (ICLR)*, 2015.

<a id="ref-9"></a>[9] K. He, X. Zhang, S. Ren, y J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.

<a id="ref-10"></a>[10] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, y H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," *arXiv preprint arXiv:1704.04861*, 2017.

<a id="ref-11"></a>[11] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510-4520.

<a id="ref-12"></a>[12] A. Howard et al., "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2019, pp. 1314-1324.

<a id="ref-13"></a>[13] M. Tan y Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. 36th Int. Conf. Machine Learning (ICML)*, 2019, pp. 6105-6114.

<a id="ref-14"></a>[14] S. Ioffe y C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, 2015, pp. 448-456.

<a id="ref-15"></a>[15] V. Nair y G. E. Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines," in *Proc. 27th Int. Conf. Machine Learning (ICML)*, 2010, pp. 807-814.

<a id="ref-16"></a>[16] T.-Y. Lin, P. Goyal, R. Girshick, K. He, y P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2980-2988.
