# Ensamble Multimodelo por Soft Voting

Se diseñó e implementó un sistema de ensamble heterogéneo basado en **Soft Voting (votación por promedio de probabilidades)** que combina las tres arquitecturas del proyecto: **`EfficientNet-Lite0`**, **`EfficientNet-B0`** y **`ShuffleNet-V2-x1.0`**.

---

## 1. Fundamento Teórico y Diversidad de Arquitecturas

En el aprendizaje profundo, un ensamble solo produce ganancias significativas si las redes individuales cometen **errores descorrelacionados** (*Principio de Diversidad de Dietterich*). Si combináramos dos modelos de la misma familia, ambos compartirían los mismos sesgos de representación.

Por esta razón se seleccionaron arquitecturas con **sesgos inductivos (*inductive bias*) dispares**. `EfficientNet-Lite0` aporta una tercera fuente de diversidad: sustituye los bloques Squeeze & Excitation y las activaciones *swish* de `B0` por operaciones cuantizables, de modo que su representación interna difiere de la de `B0` pese a compartir la familia de escalado compuesto.

```
                    ┌────────────────────────────────────┐
                    │ Imagen de Entrada (224 x 224 x 3)  │
                    └─────────────────┬──────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
 ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
 │ EfficientNet-B0  │      │EfficientNet-Lite0│      │ShuffleNet-V2-x1.0│
 │ • MBConv + SE    │      │ • MBConv sin SE  │      │ • Channel Split  │
 │ • Activación     │      │ • ReLU6 en vez   │      │   & Shuffle      │
 │   swish          │      │   de swish       │      │ • Sin atención   │
 │ • 4.02 M param.  │      │ • 3.38 M param.  │      │ • 1.26 M param.  │
 └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
          │                         │                         │
          ▼                         ▼                         ▼
        P_1                       P_2                       P_3
   [p_1 ... p_9]             [q_1 ... q_9]             [r_1 ... r_9]
          │                         │                         │
          └───────────────────────────┼───────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │        Agregación por Soft Voting       │
                 │  P_ens = (P_1 + P_2 + P_3) / 3          │
                 └────────────────────┬────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
                 │ Predicción Final: y_hat = argmax(P_ens) │
                 └─────────────────────────────────────────┘
```

### Algoritmo de Inferencia:
Dada una imagen foliar $x$, cada modelo $m \in \{1, \dots, M\}$ genera una distribución de probabilidad posterior sobre las $C=9$ clases mediante la función Softmax:
$$P_m(y = c \mid x) = \frac{e^{z_{m, c}}}{\sum_{j=1}^C e^{z_{m, j}}}$$

El ensamble por Soft Voting ponderado calcula la distribución conjunta:
$$P_{\text{ensemble}}(y = c \mid x) = \sum_{m=1}^M w_m \cdot P_m(y = c \mid x), \quad \text{donde } \sum_{m=1}^M w_m = 1$$

En la implementación los pesos son uniformes: `src/models/ensemble.py` asigna $w_m = 1/M$ cuando no se especifican, es decir $1/3$ a cada una de las tres redes. No se exploró ponderar por rendimiento individual.

Los recuentos de parámetros del diagrama son los del modelo con la cabeza de 9 clases de este proyecto, no los de la publicación original con 1000 clases.

---

## 2. Comparativa de Rendimiento en Test (5,015 Imágenes)

El ensamble combina las **tres** arquitecturas, cada una con su mejor checkpoint: `efficientnet_lite0/20260812_221429`, `efficientnet_b0/20260910_170120` y `shufflenet_v2_x1_0/20260910_184521`. Las cuatro cifras se recomputan de forma exacta desde `ensamble_predicciones.csv`.

Al evaluar el ensamble en el conjunto de prueba independiente (`test.csv`), los resultados superaron a los modelos individuales en todas las métricas globales clave:

![Comparativa del Ensamble](/ensemble/ensemble_comparison_bar.png)

### Tabla de Resultados Cuantitativos:

| Modelo / Ensamble | Macro $F_1$-Score | Exactitud (Accuracy) | Macro Precision | Macro Recall | Delta vs Mejor Individual |
|---|:---:|:---:|:---:|:---:|:---:|
| **ShuffleNet-V2-x1.0** | 0.9330 (93.30%) | 0.9731 (97.31%) | 0.9388 | 0.9298 | -1.53 pp |
| **EfficientNet-Lite0** | 0.9468 (94.68%) | 0.9791 (97.91%) | 0.9533 | 0.9413 | -0.15 pp |
| **EfficientNet-B0** | 0.9483 (94.83%) | 0.9797 (97.97%) | 0.9589 | 0.9400 | (Base individual) |
| **Soft Voting Ensemble** 🏆 | **`0.9567` (95.67%)** | **`0.9829` (98.29%)** | **`0.9642`** | **`0.9506`** | **+0.84 pp netos** 🚀 |

::: tip Máxima sensibilidad agronómica (Recall: 95.06%)
El mayor beneficio del ensamble está en el **Macro Recall**, que asciende a **0.9506**, por encima de las tres redes individuales: `EfficientNet-Lite0` 0.9413, `EfficientNet-B0` 0.9400 y `ShuffleNet-V2` 0.9298. En patología vegetal el Recall es la métrica de bioseguridad más crítica: un falso negativo —no detectar una roya o una deficiencia incipiente— cuesta la pérdida potencial del cultivo.
:::

::: warning La ganancia no está repartida
El +0.84 pp es agregado. Desglosado por procedencia, el ensamble **mejora en 4 de las 14 fuentes del corpus y empeora en 5**: aporta donde el mejor modelo individual es débil (`corn_leaf_roboflow` +0.0144) y resta donde ya rozaba el techo (`cropdg` −0.0154). El desglose completo está en [Modelos avanzados y ensamble](/es/resultados/ensamble).
:::

---

## 3. Matriz de Confusión del Ensamble

La matriz de confusión normalizada calculada sobre las 5,015 imágenes del conjunto de prueba ilustra la alta fidelidad diagnóstica alcanzada:

![Matriz de Confusión Ensamble](/ensemble/confusion_matrix_ensemble.png)

### Hallazgos Diagnósticos Clave:

1. **Patologías críticas:**
   - **Lethal Necrosis (MLN):** Recall de **1.0000** — 963 de 963 imágenes, sin falsos negativos. Relevante por tratarse de una enfermedad cuarentenaria y destructiva.
   - **Healthy (planta sana):** precisión **0.9939** y recall 0.9977. Rara vez clasifica una planta enferma como sana, lo que limita los tratamientos fitosanitarios innecesarios.
   - **Common Rust (roya):** recall **0.9882** sobre 338 muestras de campo y laboratorio.

2. **Casos de frontera (deficiencias nutricionales):**
   - Las tres deficiencias son el reto más severo, por bajo soporte y por la similitud visual de los patrones cloróticos.
   - $F_1$ de **0.9057** en nitrógeno y **0.9632** en fósforo. En **potasio** —la clase de menor soporte, 93 muestras— el $F_1$ es **0.8475**, con precisión 0.8929 y recall 0.8065.
   - Las tres son también las clases con más falsos negativos del sistema, lo que motiva la advertencia de confianza descrita en [Evaluación](./evaluacion).

---

## 4. Estrategia de Despliegue Dual

La existencia de este ensamble multimodelo permite una arquitectura de despliegue en dos niveles:

1. **Nivel 1: Despliegue en la Nube / API (Servicio Completo):**
   - Para productores o técnicos del CENTA/MAG con acceso a Internet, el backend FastAPI puede ejecutar el **Soft Voting Ensemble**, entregando la máxima precisión medida ($F_1 = 0.9567$) a cambio de triplicar el cómputo de inferencia.
2. **Nivel 2: Despliegue Móvil Desconectado (Edge):**
   - Para zonas rurales sin cobertura, la aplicación móvil ejecuta **`EfficientNet-Lite0` cuantizado a INT8** (3.6 MiB), con $F_1$ de 0.9468 en el modelo de precisión completa.
   - El `manifest.json` de la aplicación registra la corrida exacta que empaqueta: `20260812_221429`, con su SHA-256.

---

## 5. Reproducibilidad de la Evaluación

Para reproducir la evaluación del ensamble en cualquier momento:

Los checkpoints se pasan explícitos. El autodescubrimiento por `latest.json` selecciona para `efficientnet_lite0` la corrida `20260907_163546`, que es la entrenada sobre imágenes segmentadas y rinde 0.7191:

```bash
# En GPU remota (Modal):
modal run --detach scripts/modal/train.py::evaluate_ensemble_modal   --models "efficientnet_lite0 efficientnet_b0 shufflenet_v2_x1_0"   --checkpoints "/outputs/main/efficientnet_lite0/20260812_221429/best.pth /outputs/main/efficientnet_b0/20260910_170120/best.pth /outputs/main/shufflenet_v2_x1_0/20260910_184521/best.pth"   --num-workers 32 --output-dir /outputs/ensemble_mejores

# En entorno local:
python scripts/pipeline/evaluate_ensemble.py   --models efficientnet_lite0 efficientnet_b0 shufflenet_v2_x1_0   --checkpoints <ruta_lite0> <ruta_b0> <ruta_shufflenet>   --num-workers 8
```

La corrida serializa `ensemble_summary.json`, `ensemble_comparison.csv` y
`ensemble_predictions.csv`, este último con una fila por imagen que incluye la procedencia y la
predicción de cada modelo individual. Toda cifra de esta página se recomputa desde él.
