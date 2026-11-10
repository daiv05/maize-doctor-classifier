# Ensamble Multimodelo por Soft Voting

> [!NOTE] Ámbito metodológico de este diseño Este componente describe la combinación de las tres arquitecturas sobre el conjunto de prueba estratificado de la etapa 2 (`seed_42`, 33 433 imágenes), con las corridas `20260812_221429` (Lite0), `20260910_170120` (B0) y `20260910_184521` (ShuffleNet-V2). Para la auditoría de robustez del ensamble ante fuentes no vistas (exclusión completa LOSO en 3 semillas), consulte [Evaluación del Ensamble Fuera de Dominio](/es/resultados/generalizacion#ensamble-bajo-loso).

Cuando tres expertos agrícolas observan una misma hoja con síntomas dudosos, es probable que se fijen en detalles diferentes: uno prestará más atención a las manchas cloróticas centrales, otro a las quemaduras del borde y el tercero a la textura de la lámina. Si combinamos sus opiniones, el veredicto final suele ser mucho más acertado que el de cualquiera de ellos por separado.

En aprendizaje profundo ocurre exactamente lo mismo. Para llevar la capacidad predictiva al máximo nivel, diseñamos un sistema de ensamble por votación suave (*Soft Voting*) que combina las tres arquitecturas entrenadas: **`EfficientNet-Lite0`**, **`EfficientNet-B0`** y **`ShuffleNet-V2-x1.0`**.

---

## El principio de diversidad

Un ensamble solo funciona de verdad si los modelos que lo componen cometen errores en cosas distintas. Si combináramos tres modelos idénticos o de la misma familia exacta, todos compartirían las mismas debilidades y el ensamble solo repetiría sus equivocaciones.

Las tres redes difieren en arquitectura:
- **`EfficientNet-B0`** aporta alta capacidad de extracción visual gracias a sus bloques de atención selectiva (Squeeze & Excitation).
- **`EfficientNet-Lite0`** no tiene bloques Squeeze & Excitation y usa ReLU6 en lugar de swish, para facilitar la cuantización.
- **`ShuffleNet-V2`** opera sobre división y barajado de canales.

Cada red toma la imagen de 224 × 224 píxeles y calcula su propia distribución de probabilidades sobre las 9 clases. El ensamble (`SoftVotingEnsemble`) aplica softmax a los logits de cada red, promedia las tres distribuciones con pesos iguales de 1/3 y emite la clase con mayor probabilidad promedio.

---

## Ganancias en el conjunto de prueba

Al evaluar el ensamble sobre las 5,015 imágenes del conjunto de prueba retenido, la combinación superó a todos los modelos individuales en cada una de las métricas clave:

![Comparativa del Ensamble](/ensemble/ensemble_comparison_bar.png)

| Modelo / Ensamble | Exactitud (Accuracy) | Macro $F_1$-Score | Macro Precision | Macro Recall |
|---|:---:|:---:|:---:|:---:|
| **ShuffleNet-V2-x1.0** | 97.31 % | 0.9330 | 0.9388 | 0.9298 |
| **EfficientNet-Lite0** | 97.91 % | 0.9468 | 0.9533 | 0.9413 |
| **EfficientNet-B0** | 97.97 % | 0.9483 | 0.9589 | 0.9400 |
| **Ensamble Soft Voting** 🏆 | **`98.29 %`** | **`0.9567`** | **`0.9642`** | **`0.9506`** |

El Macro Recall del ensamble es 95.06 %, frente a 94.13 % (Lite0), 94.00 % (B0) y 92.98 % (ShuffleNet-V2). En patología vegetal un falso negativo equivale a declarar sana una planta enferma.

Los errores de los tres modelos se solapan solo en parte: 206 imágenes son mal clasificadas por al menos un modelo y 44 por los tres; el ensamble falla en 86 (esas 44 y otras 42 en las que algún modelo acertaba). La correlación phi entre los indicadores de error de cada par de modelos va de 0.49 a 0.53.

![Matriz de Confusión Ensamble](/ensemble/confusion_matrix_ensemble.png)

En patologías críticas como la **necrosis letal del maíz (MLN)**, el ensamble alcanzó un recall perfecto (1.0000 sobre 963 casos de prueba, sin un solo caso no detectado). En plantas sanas y roya común superó el 99 % de precisión. El $F_1$ del ensamble en nitrógeno (0.9057), fósforo (0.9632) y potasio (0.8475) supera al de cada modelo individual.

---

## Estrategia de despliegue en dos niveles

La existencia de este ensamble abre una arquitectura muy práctica para el despliegue del sistema:

1. **Nivel local en el teléfono (Edge AI):** Para el trabajo diario en la parcela sin cobertura de red, la app ejecuta **`EfficientNet-Lite0` cuantizado a Int8** (3.74 MB), con inferencia de ~23 ms en el teléfono. El modelo desplegado es el oficial `20260921_204608`: su int8 obtiene Macro-F1 0.9489 sobre el test completo (PyTorch 0.9480).
2. **Nivel en la nube (API para técnicos y extensión):** Cuando el productor cuenta con conexión o cuando técnicos agrícolas del CENTA/MAG consultan el sistema desde una computadora, el backend puede ejecutar el **ensamble completo**, con el 95.67 % de Macro F1 y el 98.3 % de exactitud medidos en el test de la etapa 2 (con el Lite0 `20260812_221429`, no con el oficial).
