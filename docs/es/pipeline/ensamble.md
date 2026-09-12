# Ensamble Multimodelo por Soft Voting

Se diseñó e implementó un sistema de ensamble heterogéneo basado en **Soft Voting (votación por promedio de probabilidades)** que combina las fortalezas complementarias de **`EfficientNet-B0`** y **`ShuffleNet-V2-x1.0`**.

---

## 1. Fundamento Teórico y Diversidad de Arquitecturas

En el aprendizaje profundo, un ensamble solo produce ganancias significativas si las redes individuales cometen **errores descorrelacionados** (*Principio de Diversidad de Dietterich*). Si combináramos dos modelos de la misma familia, ambos compartirían los mismos sesgos de representación.

Por esta razón se seleccionaron intencionalmente dos arquitecturas con **sesgos inductivos (*inductive bias*) radicalmente dispares**:

```
                         ┌────────────────────────────────────┐
                         │ Imagen de Entrada (224 x 224 x 3)  │
                         └─────────────────┬──────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │      EfficientNet-B0        │               │      ShuffleNet-V2-x1.0     │
     │ • Bloques MBConv Invertidos │               │ • Channel Split & Shuffle   │
     │ • Atención Squeeze & Excite │               │ • Sin atención de canales   │
     │ • Compounding Scaling (NAS) │               │ • Optimización de memoria   │
     │ • Parámetros: 5.3 M         │               │ • Parámetros: 2.3 M         │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
                    ▼                                             ▼
          Vector Probabilidades P_1                     Vector Probabilidades P_2
          [p_1, p_2, ..., p_9]                          [q_1, q_2, ..., q_9]
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │       Agregación por Soft Voting            │
                    │       P_ens = 0.5 * P_1 + 0.5 * P_2         │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │ Predicción Final: y_hat = argmax(P_ens)     │
                    └─────────────────────────────────────────────┘
```

### Algoritmo de Inferencia:
Dada una imagen foliar $x$, cada modelo $m \in \{1, \dots, M\}$ genera una distribución de probabilidad posterior sobre las $C=9$ clases mediante la función Softmax:
$$P_m(y = c \mid x) = \frac{e^{z_{m, c}}}{\sum_{j=1}^C e^{z_{m, j}}}$$

El ensamble por Soft Voting ponderado calcula la distribución conjunta:
$$P_{\text{ensemble}}(y = c \mid x) = \sum_{m=1}^M w_m \cdot P_m(y = c \mid x), \quad \text{donde } \sum_{m=1}^M w_m = 1$$

En nuestra implementación, los pesos se configuraron en paridad equilibrada ($w_1 = 0.5$, $w_2 = 0.5$), permitiendo que la confianza probabilística de ambas redes filtre la incertidumbre individual.

---

## 2. Comparativa de Rendimiento en Test (5,015 Imágenes)

Al evaluar el ensamble en el conjunto de prueba independiente (`test.csv`), los resultados superaron a los modelos individuales en todas las métricas globales clave:

![Comparativa del Ensamble](/ensemble/ensemble_comparison_bar.png)

### Tabla de Resultados Cuantitativos:

| Modelo / Ensamble | Macro $F_1$-Score | Exactitud (Accuracy) | Macro Precision | Macro Recall | Delta vs Mejor Individual |
|---|:---:|:---:|:---:|:---:|:---:|
| **ShuffleNet-V2-x1.0** | 0.9330 (93.30%) | 0.9731 (97.31%) | 0.9388 | 0.9298 | -1.53 pp |
| **EfficientNet-B0** | 0.9483 (94.83%) | 0.9797 (97.97%) | 0.9589 | 0.9400 | (Base individual) |
| **Soft Voting Ensemble** 🏆 | **`0.9507` (95.07%)** | **`0.9799` (97.99%)** | **`0.9582`** | **`0.9445`** | **+0.24 pp netos** 🚀 |

::: tip Máxima Sensibilidad Agronómica (Recall: 94.45%)
El mayor beneficio del ensamble se observa en el **Macro Recall**, que asciende a **0.9445 (94.45%)**, superior tanto a EfficientNet (0.9400) como a ShuffleNet (0.9298). En patología vegetal, el Recall es la métrica de bioseguridad más crítica: un falso negativo (no detectar una roya o una deficiencia foliar incipiente) cuesta la pérdida potencial del cultivo.
:::

---

## 3. Matriz de Confusión del Ensamble

La matriz de confusión normalizada calculada sobre las 5,015 imágenes del conjunto de prueba ilustra la alta fidelidad diagnóstica alcanzada:

![Matriz de Confusión Ensamble](/ensemble/confusion_matrix_ensemble.png)

### Hallazgos Diagnósticos Clave:

1. **Rendimiento Perfecto en Patologías Críticas:**
   - **Lethal Necrosis (MLN):** 100% de Recall (963 de 963 imágenes clasificadas correctamente, 0 falsos negativos). Esto es vital dado que la necrosis letal del maíz es cuarentenaria y destructiva.
   - **Healthy (Planta Sana):** 99.7% de precisión. La red prácticamente no confunde plantas sanas con enfermas, eliminando tratamientos fitosanitarios innecesarios.
   - **Common Rust (Roya):** 98.5% de sensibilidad sobre 338 muestras de campo y laboratorio.

2. **Resolución de Casos de Frontera (Deficiencias Nutricionales):**
   - Las deficiencias nutricionales (*Nitrogen*, *Phosphorus*, *Potassium*) representan el reto más severo debido al bajo número de muestras y la similitud visual de los patrones cloróticos.
   - El ensamble logró un $F_1$ del **89.5% en Nitrógeno** y **95.3% en Fósforo**. En **Potasio** (clase más minoritaria con solo 93 muestras en test), el Soft Voting amortiguó los falsos positivos de ShuffleNet, alcanzando una precisión del **89%**.

---

## 4. Estrategia de Despliegue Dual

La existencia de este ensamble multimodelo permite una arquitectura de despliegue en dos niveles:

1. **Nivel 1: Despliegue en la Nube / API (Servicio Completo):**
   - Para productores o técnicos del CENTA/MAG con acceso a Internet, el backend FastAPI ejecuta el **Soft Voting Ensemble**, entregando la máxima precisión posible ($F_1 = 0.9507$) y calibración de confianza.
2. **Nivel 2: Despliegue Móvil Desconectado (Edge):**
   - Para zonas rurales sin cobertura 4G, la aplicación móvil ejecuta individualmente **`ShuffleNet-V2` cuantizado a INT8** (apenas 2.3M parámetros y ~2.5 MB de peso), manteniendo un sólido 93.3% de $F_1$ con inferencia instantánea en el dispositivo.

---

## 5. Reproducibilidad de la Evaluación

Para reproducir la evaluación del ensamble en cualquier momento:

```bash
# En GPU remota (Modal):
make modal-ensemble MODELS="efficientnet_b0 shufflenet_v2_x1_0"

# O en entorno local:
make evaluate-ensemble MODELS="efficientnet_b0 shufflenet_v2_x1_0"
```
*(Los reportes se serializan automáticamente en `outputs/ensemble/ensemble_summary.json` y `outputs/ensemble/ensemble_comparison.csv`).*
