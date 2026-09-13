# Prototipo en dispositivo

Pruebas del prototipo funcional de DoctorMaiz corriendo sobre un teléfono real, con el modelo
del run `20260812_221429`. La página reporta qué hace la app de punta a punta, cuánto tarda cada
etapa del pipeline, qué predice sobre imágenes del corpus limpio, y dos límites que las pruebas
dejaron a la vista: uno de implementación, que se corrigió, y otro de datos, que no se puede
corregir con más ingeniería.

## Qué se midió y con qué

| | |
|---|---|
| Dispositivo | Motorola edge 40 neo — MediaTek Dimensity 7030 (`MT6879`), Android 15 |
| Pantalla | 1080 × 2400 px a 400 dpi, es decir 432 × 960 dp |
| APK | `com.doctormaiz.app` 1.0.0, `minSdk` 24, `targetSdk` 36, ABIs `arm64-v8a` + `armeabi-v7a` |
| Modelo embarcado | `efficientnet_lite0`, run `20260812_221429`, `model_int8.tflite` |
| sha256 del modelo | `0a77ad7937f73e24b795288c68741d14dd00226ffe546d5ddf5b99125fbef24d` |
| Detector OOD | `ood_stats.json` esquema 4 — `feature_dim` 1280, `pca_dim` 186, umbral 31,468311 (percentil 95 sobre validación) |
| Referencia del servidor | 5 015 imágenes de prueba: accuracy 0,9785, macro-F1 0,9474 |

Las imágenes de prueba salen del split de test de `data/clean`, tres por clase, elegidas al azar
con semilla 42 y repartidas entre fuentes distintas cuando la clase tiene más de una. Entran a la
app por la galería, no por la cámara, porque es la única vía que permite controlar exactamente qué
píxeles ve el modelo.

::: tip El selector fuerza un recorte 1:1
`handlePickFromGallery` llama a `launchImageLibraryAsync` con `allowsEditing: true` y
`aspect: [1, 1]`. Para que ese editor no altere la imagen, todas las de prueba se recortaron antes
a un cuadrado centrado. Sin eso, la comparación contra el servidor no sería sobre los mismos
píxeles.
:::

**Cada resultado se valida antes de aceptarlo.** El motor de la app volcó por `adb logcat` la suma
del tensor de entrada, y el arnés compara esa suma con la calculada en el PC para la imagen que
creía estar enviando. Sin esa comprobación el experimento habría quedado inservible: el selector de
fotos de Android ordena por fecha y devolvía siempre la misma imagen, así que las primeras tandas
midieron cuarenta y cinco veces la misma foto sin que nada lo delatara.

## Lo que hace la app

![Pantalla de inicio](/app/home.png)

El inicio ofrece el acceso al escáner y la entrada al aporte de datos. **Las cuatro tarjetas
ambientales (temperatura, humedad, humedad de suelo, viento) y los escaneos recientes son datos
fijos de maqueta**, no lecturas reales: están escritos en `HomeScreen.tsx` y en `mockData.ts`.

![Cámara con el marco guía](/app/camara-marco-guia.png)

La cámara superpone un marco con forma de hoja, una línea de barrido animada y dos instrucciones
explícitas sobre la distancia de captura. Es la respuesta de la app al problema del encuadre, que
se detalla más abajo.

![Aporte al dataset nacional](/app/contribuir.png)

La pantalla de aporte pide foto y etiqueta de diagnóstico, y encola el envío al backend. Es el
mecanismo con el que el proyecto pretende salir del techo de datos descrito al final de esta
página.

## Un defecto que solo aparece en el teléfono

La primera tanda de escaneos devolvía siempre **100 % de confianza**, incluso en imágenes que el
servidor clasificaba con 0,78. Esa saturación no era un detalle cosmético: era el síntoma de que el
artefacto embarcado calculaba en el teléfono algo distinto de lo que calculaba en el servidor.

### Cómo se aisló

El motor de inferencia se instrumentó temporalmente para volcar por `adb logcat` la suma del tensor
de entrada, la suma del vector de features, los logits crudos y la distancia RMD. Se le pasó un PNG
sintético de gradiente —determinista, sin recompresión JPEG de por medio— y se comparó contra el
mismo artefacto corrido en CPU con el preprocesado del pipeline.

| Magnitud | Servidor | Teléfono |
|---|---:|---:|
| Suma del tensor de entrada | 32 806,73 | 32 806,72 |
| Primer elemento del tensor | −2,1179 | −2,1179 |
| Suma del vector de features (`output_1`) | 23,91 | 23,41 |
| RMD del detector OOD | 56,59 | 55,53 |
| Logit máximo (`output_0`) | 2,854 | **922,95** |
| Confianza softmax | 0,7430 | **1,0000** |

El tensor de entrada es idéntico y las features coinciden. Solo divergen los logits, y lo hacen por
un factor de unas 321×, conservando casi entero el orden de clases (R² = 0,999 en un ajuste lineal).
Eso descarta el preprocesado, el decodificador de imagen, el orden de canales y el layout del
tensor: el problema estaba en la última capa.

### La causa

La cola del grafo exportado es:

```
Op#64 TRANSPOSE(...)                   -> T#171   [1, 1280, 7, 7]
Op#65 SUM(T#171, ejes[3,2])            -> T#172   suma sobre 7×7
Op#66 MUL(T#172, 1/49)                 -> T#173   output_1 (features)
Op#67 FULLY_CONNECTED(T#172, W, b)     -> T#174   output_0 (logits)
```

`W` era int8 con **nueve escalas, una por clase de salida**, porque el export cuantizaba con
`get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)`.

El kernel integrado de TFLite para `FULLY_CONNECTED` con entrada dinámica aplica **una sola escala
por tensor**. Con escalas por canal devuelve `W_int8 · x + b` en lugar de `W_int8 · escala_c · x + b`,
sin error ni aviso. El delegado XNNPACK sí implementa el caso per-channel; el runtime que embarca
`react-native-fast-tflite` no lo activa.

La reconstrucción confirma el mecanismo: `W_int8 · sum_pool + b` reproduce los logits del teléfono
con una desviación relativa mediana del **1,5 %** (el resto lo explica la cuantización dinámica de
las activaciones, que no se modela en el cálculo a mano).

| Comprobación | Resultado |
|---|---|
| Intérprete Python **sin delegados**, artefacto int8 | reproduce los logits del teléfono |
| Intérprete Python **con XNNPACK**, artefacto int8 | logits correctos |
| Artefacto **FP32**, con y sin delegado | máxima diferencia de probabilidad **0,0000** |

### Qué tan grave era

Medido sobre 540 imágenes del split de test, 60 por clase, comparando el intérprete con XNNPACK
(el del servidor) contra el intérprete sin delegados (el del teléfono):

| | Servidor | Teléfono |
|---|---:|---:|
| Acuerdo de clase predicha | — | **99,81 %** |
| Accuracy | 0,9519 | 0,9500 |
| macro-F1 | 0,9517 | 0,9497 |
| Confianza media | 0,8589 | **0,9995** |
| Predicciones con confianza > 0,999 | — | **99,81 %** |

El diagnóstico casi no cambiaba: una imagen de 540 cambió de clase. **Lo que se perdió por completo
fue la capacidad de dudar.** Con el softmax saturado, la compuerta `MIN_CONFIDENCE = 0,6` /
`MIN_MARGIN = 0,15` no podía dispararse nunca, y la confianza que mostraba la pantalla de resultado
no significaba nada. Para una herramienta de campo esa es la pérdida que importa: el camino de baja
confianza es justo el que debería derivar los casos ambiguos a una segunda opinión.

El detector OOD por Mahalanobis siguió funcionando durante todo el episodio, porque consume
`output_1`, que era correcto.

### Por qué no lo vio la verificación existente

`eval_tflite_int8.json` y el *parity gate* de `export_summary_int8.json` construyen el intérprete
con `Interpreter(model_path=...)`, que activa XNNPACK por defecto. Ambos medían el único camino que
aplica bien las escalas. Las cifras publicadas del artefacto int8 describían el servidor; no
describían lo que calculaba el teléfono.

Y el acuerdo top-1 entre los dos intérpretes sobre las 27 imágenes de prueba era **1,0000**: una
compuerta que solo mirase la clase predicha habría dejado pasar el fallo. Lo que lo delata es la
diferencia de probabilidad, que llega a **0,4312**.

### Qué se cambió

`src/export/tflite_export.py` cuantiza ahora el backbone per-channel y **solo la capa lineal final
per-tensor**:

```python
per_tensor = get_symmetric_quantization_config(is_per_channel=False, is_dynamic=True)
quantizer = PT2EQuantizer().set_global(
    get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
)
for linear_key in (torch.nn.Linear, "torch.nn.modules.linear.Linear"):
    quantizer.set_module_type(linear_key, per_tensor)
```

Poner *todo* per-tensor no era alternativa: probado sobre `efficientnet_lite0`, el acuerdo top-1
contra PyTorch cae a **0,30**, porque las convoluciones depthwise de EfficientNet tienen rangos de
peso muy distintos entre canales y una sola escala las destruye. Lo detuvo la compuerta de paridad
que ya existía.

Las dos claves del `set_module_type` no son redundancia decorativa: `_get_module_type_filter`
compara contra `nn_module_stack`, y `torch.export` guarda ahí el tipo como cadena en unas versiones
y como clase en otras. Con solo la clase, la anotación no casaba con nada y el artefacto salía byte
a byte idéntico al anterior.

| Artefacto | sha256 | `FULLY_CONNECTED` con escalas distintas | Confianza sin delegado | max\|Δprob\| entre intérpretes |
|---|---|---:|---:|---:|
| Anterior | `3a0623cd…` | 1 | 1,0000 | 0,4312 |
| Corregido | `0a77ad79…` | **0** | **0,7563** | **0,0016** |

Sobre las 5 015 imágenes de prueba el artefacto corregido rinde igual que el anterior en el
servidor: accuracy 0,9785 (antes 0,9785) y macro-F1 0,9474 (antes 0,9477). El arreglo no cuesta
precisión; solo hace que el teléfono calcule lo mismo que el servidor.

### La compuerta que lo habría detectado

`validate_tflite_parity` ahora hace dos comprobaciones más, y ambas reprueban el export si fallan.

La primera es **estructural** y no depende de qué runtime esté instalado: cuenta los
`FULLY_CONNECTED` cuyo tensor de pesos lleva escalas distintas entre canales. Lo que importa es que
difieran, no cuántas haya — el exportador emite la cuantización per-tensor como un tensor per-axis
con las nueve escalas iguales, y ese caso el kernel integrado sí lo resuelve bien.

La segunda corre el artefacto **con y sin delegado** y compara probabilidades con tolerancia 0,05.

La comprobación estructural es la que sostiene la garantía, porque la comparación entre intérpretes
resultó depender del paquete instalado: sobre el artefacto roto, `tensorflow` 2.20 sin delegados
reproduce el teléfono, pero `ai_edge_litert` 2.2.0 —el único disponible en la imagen de Modal donde
corre el export— aplica bien las escalas por canal y habría dado un **falso aprobado**
(`max|Δprob|` = 0,0068). Una compuerta cuyo veredicto cambia según qué tenga instalado la máquina no
protege de nada.

## Tiempos por etapa

<!-- MEDICIONES_TIEMPOS -->

## Inferencias sobre el corpus limpio

<!-- MEDICIONES_INFERENCIA -->

## El problema del encuadre

<!-- MEDICIONES_ENCUADRE -->

## El techo: catorce fuentes, ninguna centroamericana

El corpus limpio tiene **33 437 imágenes**, y salen de **catorce fuentes públicas**. Esa cifra, no
el número de imágenes, es la que manda sobre lo que el modelo puede aprender.

::: details Censo completo por fuente y por clase

| fuente | imágenes | % del corpus | clases que aporta |
|---|---:|---:|---:|
| `maize_africa` | 9 830 | 29.4 % | 3 |
| `multi_desease` | 5 816 | 17.4 % | 3 |
| `maize_desease_v1.1` | 4 741 | 14.2 % | 1 |
| `corn_leaf_roboflow` | 2 919 | 8.7 % | 6 |
| `cropdg` | 2 562 | 7.7 % | 3 |
| `maize_desease` | 2 248 | 6.7 % | 2 |
| `maize_africa_v1.2` | 1 854 | 5.5 % | 1 |
| `maize_field` | 852 | 2.5 % | 3 |
| `maize_2_roboflow` | 846 | 2.5 % | 3 |
| `corn_leaf_diseases_classification_roboflow` | 480 | 1.4 % | 1 |
| `maize_africa_v1` | 460 | 1.4 % | 1 |
| `maize_nutrient` | 340 | 1.0 % | 4 |
| `maize_leaf_roboflow` | 331 | 1.0 % | 1 |
| `maize_deficiency_scanner_roboflow` | 158 | 0.5 % | 3 |

| clase | imágenes | fuentes | fuente dominante | % de la clase |
|---|---:|---:|---|---:|
| `common_rust` | 2 256 | 3 | `maize_desease` | 52.8 % |
| `fall_armyworm` | 4 857 | 3 | `maize_africa` | 49.9 % |
| `gray_leaf_spot` | 1 930 | 4 | `maize_field` | 31.4 % |
| `healthy` | 8 744 | 6 | `maize_desease_v1.1` | 54.2 % |
| `lethal_necrosis` | 6 415 | 2 | `multi_desease` | 50.4 % |
| `nitrogen_deficiency` | 846 | 4 | `corn_leaf_roboflow` | 50.1 % |
| `northern_corn_leaf_blight` | 6 830 | 5 | `maize_africa` | 61.8 % |
| `phosphorus_deficiency` | 938 | 4 | `corn_leaf_roboflow` | 53.2 % |
| `potassium_deficiency` | 621 | 4 | `maize_2_roboflow` | 49.6 % |

:::


**Cinco de las catorce fuentes aportan una sola clase**, y cada clase está dominada por una sola
fuente, que aporta entre el 31 % y el 62 % de sus imágenes. Con esa estructura el modelo puede
acertar la clase reconociendo de qué dataset viene la foto, sin mirar la lesión. El proyecto ya
midió que eso ocurre: un anillo del 10 % del borde, **sin hoja ni lesión dentro**, clasifica el
78,3 % del test. Todo el análisis está en [Procedencia y fuga](/es/provenance/).

La consecuencia práctica se ve al evaluar honestamente:

| | macro-F1 |
| --- | ---: |
| Modelo desplegado, prueba retenida | 0,9468 |
| Generalización a una fuente no vista | **0,6026 ± 0,1240** |

Esos 0,34 de caída son el techo de esta etapa. Y no es un techo que se levante con más arquitectura,
más épocas ni mejor cuantización: es la distancia entre acertar sobre fotos parecidas a las que ya
vio y acertar sobre fotos nuevas.

El origen geográfico de las fuentes explica por qué. Los datasets de maíz disponibles vienen de
Sudáfrica, Burkina Faso, Uganda, Namibia, Tanzania e India, más colecciones de laboratorio tipo
PlantVillage y lotes de Roboflow sin procedencia declarada. **Ninguna de las catorce documenta
capturas en El Salvador ni en Centroamérica.** Las variedades de maíz, las prácticas de manejo, la
luz, el suelo de fondo y hasta la cámara típica son distintos, y esa diferencia es exactamente la
que el experimento de fuente no vista mide como una caída de 0,34.

### Por eso la app pide fotos

La pantalla de aporte no es un extra: es la única salida disponible a ese techo. Una fuente nueva,
capturada en campo salvadoreño, con etiqueta del agricultor y con la variabilidad real de
dispositivos y condiciones, ataca el problema por donde está —los datos— en lugar de seguir
optimizando sobre un corpus cuya estructura ya se midió como el factor limitante.

El arreglo descrito en esta página es un ejemplo de la diferencia entre los dos tipos de límite. El
defecto del `FULLY_CONNECTED` era de implementación: se localizó, se midió y se corrigió en unas
horas, sin coste en precisión. El techo de las catorce fuentes no se corrige con código.
