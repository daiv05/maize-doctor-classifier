# Informe de Equidad Algorítmica, Auditoría de Sesgos y Ética (Fairness Report)
**Proyecto:** Sistema de Detección y Diagnóstico de Patologías Foliares en Maíz (*Zea mays*)  
**Etapa 2 - Criterio 4:** Análisis de sesgos y ética (15 Puntos)  
**Módulos del Pipeline:** `src/analysis/fairness.py` | `scripts/pipeline/evaluate_fairness.py` | `src/explainability/gradcam.py`

---

## 1. Resumen Ejecutivo de Equidad Algorítmica

El despliegue de modelos de visión por computadora en contextos agrícolas de pequeños productores enfrenta riesgos críticos de **sesgo de dominio** y **aprendizaje de atajos (*Shortcut Learning / Efecto Clever Hans*)**. Si una red neuronal convolucional aprende a clasificar imágenes basándose en el fondo (e.g., mesas de laboratorio, suelo, dedos del agricultor) o el sensor de la cámara en lugar de la morfología fitopatológica de la lesión foliar, el sistema fallará catastróficamente al ser utilizado en campo real.

Esta auditoría evalúa formalmente la equidad matemática del modelo principal (`EfficientNet-B0` y ensambles) a través de tres dimensiones fundamentales:
1. **Disparidad de Dominio / Entorno:** Rendimiento en condiciones de laboratorio controlado (`lab`) versus campo abierto real (`real`).
2. **Control Negativo de Atajos Visuales (*Clever Hans Check*):** Test de oclusión foliar para verificar que la red no retenga predicciones espurias sin tejido vegetal lesionado.
3. **Equidad por Clase y Costo Asimétrico del Error:** Análisis de la tasa de falsos negativos ($FNR = 1 - \text{Recall}$) en clases mayoritarias frente a patologías o deficiencias nutricionales minoritarias.

---

## 2. Métricas Formales de Equidad Cuantitativa

### 2.1. Rendimiento Desagregado por Entorno de Captura (5,015 Muestras de Test)

| Entorno de Captura | Muestras ($N$) | Macro $F_1$ | Accuracy | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Campo Agrícola Real (`real`)** | **4,483** | **0.9298** | **0.9842** | 0.9466 | 0.9164 |
| **Laboratorio Controlado (`lab`)** | **532** | **0.2955\*** | **0.9417** | 0.3093 | 0.2904 |
| **Global (Test Set Independiente)** | **5,015** | **0.9483** | **0.9797** | 0.9589 | 0.9400 |

\* *Nota Demográfica:* La reducción aritmética en el Macro $F_1$ de laboratorio se debe a que el dataset de prueba únicamente contiene 3 patologías en banco de trabajo (*common_rust*, *gray_leaf_spot*, *northern_corn_leaf_blight*); las otras 6 patologías tienen 0 muestras, asignando $0.0$ al promedio. En las clases presentes, el modelo alcanza un Recall del 100% en Roya y 97.75% en Tizón, con una exactitud global en laboratorio del 94.17%.

### 2.2. Brechas de Disparidad e Impacto Dispar

* **Disparidad de Exactitud ($\Delta \text{Acc}$):**
  $$\Delta \text{Acc} = |\text{Acc}_{\text{real}} - \text{Acc}_{\text{lab}}| = |0.9842 - 0.9417| = \mathbf{0.0424} \quad (4.24\% \implies \text{Rango Seguro})$$
* **Disparate Impact Ratio en Exactitud ($DIR_{\text{Acc}}$):**
  $$DIR_{\text{Acc}} = \frac{\min(\text{Acc}_{\text{real}}, \text{Acc}_{\text{lab}})}{\max(\text{Acc}_{\text{real}}, \text{Acc}_{\text{lab}})} = \frac{0.9417}{0.9842} = \mathbf{0.9568} \ge 0.80 \quad (\text{Cumple Regla 80\%})$$

---

## 3. Auditoría de Atajos Visuales (*Shortcut Learning / Clever Hans Effect*)

### 3.1. Metodología de la Auditoría de Ablación Dual
Para certificar si la red aprende la morfología fitopatológica genuina o memoriza correlaciones espurias de fondo (suelo, sombras, reflectancia o sensor), se ejecutó un protocolo dual de ablación:
1. **Control Negativo (Oclusión Central 60%):** Se enmascara la lesión foliar central. Si la red dependiera de la patología biológica, su confianza debería colapsar drásticamente.
2. **Control Inverso (Oclusión Periférica 40%):** Se enmascara el fondo y bordes perimetrales, dejando **únicamente la lesión foliar central pura** sin pistas de entorno. Si la red aprende la patología genuina, debe mantener alta exactitud diagnóstica.

### 3.2. Resultados Cuantitativos de la Auditoría de Atajos

```
[Inferencia Original]:
  • Confianza Media:        0.8429 (84.29%)
  • Exactitud (Accuracy):   0.9797 (97.97%)

[Control Negativo - Oclusión Central 60% (Sin Lesión)]:
  • Confianza con Oclusión: 0.7795 (77.95%)
  • Caída de Confianza:     -0.0634 (-6.34 pp)
  • Ratio de Retención:     92.47% de certeza anclada al fondo
  • Exactitud sin Lesión:   0.8400 (84.00%)  [Atajo Espurio Activo]

[Control Inverso - Oclusión Periférica 40% (Solo Lesión Pura)]:
  • Confianza solo Centro:  0.6964 (69.64%)
  • Caída de Confianza:     -0.1465 (-14.65 pp)
  • Exactitud solo Centro:  0.7000 (70.00%)
  • Desplome de Exactitud:  -27.97 pp (Colapso del rendimiento)
  • Tasa de Inestabilidad:  27.84% de predicciones correctas mutan a error
```

> [!CAUTION]
> **Vulnerabilidad Crítica Confirmada (Clever Hans):** La red retiene el 92.47% de su confianza y acierta el 84% de las veces sin ver el centro de la hoja. Al aislar la lesión pura sin entorno de captura, la exactitud colapsa 28 puntos. Esto demuestra empíricamente una dependencia severa de artefactos perimetrales de fondo y sensor. Se prohíbe el despliegue directo en campo sin una etapa previa de desacople de fondo.

### 3.3. Evidencia Visual con Grad-CAM
Mediante `src/explainability/gradcam.py` sobre la última capa convolucional (`features.-1` en EfficientNet-B0), se analizó el comportamiento cualitativo:
- **En Campo Real (`real`):** El mapa térmico concentra activaciones sobre las pústulas visibles, pero parte de las neuronas del cuello de botella captan la firma de contraste entre el contorno foliar y el fondo terroso.
- **En Laboratorio (`lab`):** La red enfoca la lámina, pero exhibe sensibilidad al blanco uniforme del banco de trabajo.

---

## 4. Análisis Ético y Costo Asimétrico del Error en el Agro

En el diagnóstico fitopatológico asistido por IA, los errores de clasificación no tienen el mismo impacto social, económico ni ecológico:

```mermaid
graph TD
    A[Predicción Errónea del Modelo] --> B[Falso Negativo: Patógeno Crítico clasificado como Sano]
    A --> C[Falso Positivo: Sano o Nutricional clasificado como Hongo]
    
    B --> B1[Pérdida de la Cosecha: 60-100%]
    B --> B2[Ruina económica para pequeños agricultores de subsistencia]
    B --> B3[Propagación comunitaria de esporas fúngicas]
    
    C --> C1[Gasto innecesario en fungicidas químicos: ~$45/ha]
    C --> C2[Contaminación de mantos acuíferos y suelos]
    C --> C3[Generación de resistencia en fitopatógenos]
```

### 4.1. Tasa de Falsos Negativos ($FNR$) por Patología (Test Set Oficial)

| Patología / Condición | FNR Campo Real | FNR Laboratorio | Impacto Agronómico Crítico |
| :--- | :---: | :---: | :--- |
| **Tizón Foliar (*northern_corn_leaf_blight*)** | **0.0056 (0.56%)** | 0.0226 (2.26%) | **Severo:** Necrosis rápida del tejido fotosintético; riesgo de pérdida total de espiga. |
| **Roya Común (*common_rust*)** | **0.3125 (31.25%)** | **0.0000 (0.00%)** | **Alto:** Esporulación masiva por viento; requiere fungicida temprano. |
| **Mancha Gris (*gray_leaf_spot*)** | **0.0516 (5.16%)** | 0.3636 (36.36%) | **Moderado/Alto:** Lesiones rectangulares; difícil diferenciación en fases iniciales. |
| **Deficiencia de Potasio (*potassium*)** | **0.2366 (23.66%)** | N/A (0 muestras) | **Nutricional:** Clorosis marginal; mayor tasa de falso negativo en campo. |
| **Hojas Sanas (*healthy*)** | **0.0031 (0.31%)** | N/A (0 muestras) | **Ecológico/Financiero:** Falso positivo induce aplicación innecesaria de agroquímicos. |

---

## 5. Medidas de Mitigación y Salvaguardas para Despliegue

Para neutralizar los riesgos de sesgo demográfico y la **vulnerabilidad crítica de atajos visuales (*Clever Hans*)**, se establece una estrategia en dos niveles:

### 5.1. Mitigaciones Implementadas en el Pipeline Base
1. **Estratificación Jerárquica Dual (`src/data/splitter.py`):**  
   Particionamiento estratificado simultáneamente por **clase fitopatológica** y **entorno de captura (`lab`/`real`)**, garantizando idéntica proporción de datos de campo en entrenamiento, validación y prueba.
2. **Pérdida Ponderada `sqrt_inverse` (`src/training/losses.py`):**  
   Compensación matemática del desbalance de clases sin sobre-penalizar las clases mayoritarias, reduciendo el $FNR$ en clases minoritarias.
3. **Data Augmentation Fotométrico y de Contraste (`src/data/transforms.py`):**  
   Inclusión de `ColorJitter` (variación de brillo, contraste, saturación) y ecualización adaptativa `CLAHE`, simulando variaciones de luz solar intensa y cámaras de teléfonos de gama baja.
4. **Detección Fuera de Distribución (OOD) por Distancia de Mahalanobis:**  
   Rechazo proactivo de imágenes no foliares o corruptas antes de emitir un diagnóstico clínico.

### 5.2. Mitigaciones Obligatorias contra Atajos Visuales (Clever Hans)
1. **Desacople Mandatorio de Fondo vía Segmentación Semántica:**  
   Integración estricta de `maize-doctor-segmenter` como paso frontal en el móvil/API: recortar la lámina foliar y descartar el 100% de los píxeles perimetrales para impedir que el clasificador use el suelo o entorno como atajo.
2. **Entrenamiento Basado en Parches (*Patch-Based Training*):**  
   Entrenar sobre parches interiores de la hoja ($128 \times 128$ o $224 \times 224$), desacoplando el diagnóstico de la silueta completa y el entorno de captura.
3. **Data Augmentation Destructivo Perimetral y Swapping:**  
   Borrado perimetral aleatorio (*Random Perimeter Erasing*), síntesis de fondos alternantes y adición agresiva de ruido de sensor fotométrico y compresión JPEG.

---

## 6. Gobernanza de Datos y Privacidad de los Agricultores

* **Soberanía y No-Geocodificación Forzada:** La inferencia en la aplicación móvil `doctor-maiz-app` opera de forma **100% local (Edge Computing con TFLite)** sin transmitir coordenadas GPS de parcelas a servidores centrales, protegiendo al pequeño productor contra posibles especulaciones de mercado de granos o fiscalizaciones indebidas.
* **Consentimiento Informado:** Los metadatos de imágenes recopiladas con fines de re-entrenamiento son anonimizados mediante hash SHA-256 sin almacenar identificadores personales.

---

## 7. Comandos de Reproducibilidad

Para re-ejecutar la auditoría de equidad completa y regenerar todos los artefactos:

```bash
# Vía Makefile:
make fairness-report

# O directamente vía CLI:
python scripts/pipeline/evaluate_fairness.py \
    --model efficientnet_b0 \
    --run-gradcam \
    --run-shortcut-test
```

**Artefactos Generados en `outputs/fairness/`:**
- `fairness_metrics.json`: Registro estructurado de métricas y ratios de impacto dispar.
- `fairness_disparity.csv`: Tabla desagregada por subgrupos y clases.
- `fairness_disparity.png`: Gráfico comparativo de barras de rendimiento.
- `disaggregated_confusion_matrices.png`: Matrices de confusión normalizadas (Campo vs Laboratorio).
- `gradcam_samples.png`: Panel visual de validación de atención fotográfica.
