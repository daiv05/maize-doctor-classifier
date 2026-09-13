# Prototipo en dispositivo

Pruebas del prototipo funcional de DoctorMaiz corriendo sobre un teléfono real, con el modelo
del run `20260812_221429`. La página reporta qué hace la app de punta a punta, cuánto tarda cada
etapa del pipeline, qué predice sobre imágenes del corpus limpio, y lo que las pruebas dejaron a la
vista: dos defectos de implementación, que se corrigieron, y un techo de datos, que no se corrige
con más ingeniería.

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

**Cada resultado se valida antes de aceptarlo.** El motor de la app volcó por `adb logcat` dos
magnitudes del tensor de entrada —su suma y su primer elemento—, y el arnés las compara con las
calculadas en el PC para la imagen que creía estar enviando; hacen falta las dos porque la suma sola
colisiona entre algunas imágenes del corpus. Sin esa comprobación el experimento habría quedado
inservible: el selector de fotos de Android ordena por fecha y devolvía siempre la misma imagen, así
que las primeras tandas midieron cuarenta y cinco veces la misma foto sin que nada lo delatara.

## Lo que hace la app

![Pantalla de inicio](/app/home.png)

El inicio ofrece el acceso al escáner y la entrada al aporte de datos. **Las cuatro tarjetas
ambientales (temperatura, humedad, humedad de suelo, viento) y los escaneos recientes son datos
fijos de maqueta**, no lecturas reales: están escritos en `HomeScreen.tsx` y en `mockData.ts`.

![Cámara con el marco guía](/app/camara-marco-guia.png)

La cámara superpone un marco con forma de hoja, una línea de barrido animada y dos instrucciones
explícitas sobre la distancia de captura. Es la respuesta de la app al problema del encuadre, que
se detalla más abajo.

![Resultado con diagnóstico de mancha gris](/app/resultado-mancha-gris.png)

La pantalla de resultado da el diagnóstico, la confianza, la imagen analizada y las recomendaciones
de manejo. Los campos de temperatura y humedad aparecen como `N/D`: el escaneo no los captura.

![Aporte al dataset nacional](/app/contribuir.png)

La pantalla de aporte pide foto y etiqueta de diagnóstico, y encola el envío al backend. Es el
mecanismo con el que el proyecto pretende salir del techo de datos descrito al final de esta
página.

## Primer defecto: la cuantización del head

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

## Segundo defecto: el escalado de la imagen

Con la cuantización ya corregida, las predicciones sobre el corpus limpio seguían sin cuadrar con
las del servidor, y **más de la mitad de las hojas legítimas salían como «no reconocida»**. Esta
vez el tensor de entrada tampoco coincidía: difería en torno al 0,5 %, lo bastante poco para pasar
por ruido y lo bastante para cambiar el diagnóstico.

`preprocessImageSkia.ts` bajaba la foto al cuadrado de 224 en un solo paso:

```ts
canvas.drawImageRectOptions(
  image,
  Skia.XYWHRect(0, 0, image.width(), image.height()),
  Skia.XYWHRect(0, 0, size, size),
  FilterMode.Linear,
  MipmapMode.None,
);
```

`FilterMode.Linear` con `MipmapMode.None` muestrea **cuatro téxeles por píxel de salida**. Bajando
de 3840 px a 224 eso descarta más del 99 % de la imagen y produce aliasing severo. El pipeline de
entrenamiento escala con un filtro cuyo soporte crece con el factor de reducción, que promedia
todos los píxeles.

Reproducir ese muestreo en CPU —bilineal sobre la rejilla de salida, sin prefiltro— reproduce al
teléfono:

| Imagen | px | Referencia con antialias | Sin prefiltro (simulado) | Dispositivo |
|---|---|---|---|---|
| md00a | 3840² | common_rust 0,925 · RMD 20,8 | common_rust 0,209 · RMD 674 | common_rust 0,205 · RMD **625** |
| md02a | 3456² | common_rust 0,930 · RMD 8,2 | **healthy** 0,266 · RMD 600 | **healthy** 0,278 · RMD **567** |
| md13a | 1390² | lethal_necrosis 0,834 · RMD −1,2 | **healthy** 0,426 · RMD 451 | **healthy** 0,565 · RMD **475** |
| md24a | 3000² | potassium 0,980 · RMD −7,5 | potassium 0,983 · RMD −15,8 | potassium 0,982 · RMD −17,7 |

El daño es doble y depende del contenido de alta frecuencia, no solo del tamaño: md24a apenas se
mueve, md02a y md13a **cambian de clase**, y el RMD se multiplica por cincuenta o más. Ese segundo
efecto es el que llenaba la pantalla de «no reconocida»: el detector OOD estaba haciendo bien su
trabajo sobre un tensor que ya no era el de una hoja.

**El arreglo** reduce a la mitad por pasos antes del escalado final. Cada paso a la mitad con filtro
lineal equivale a promediar bloques de 2×2 —el mismo cálculo que un nivel de mipmap—, así que
encadenarlos reconstruye el promedio que faltaba. Validado en CPU antes de tocar la app, y después
en el teléfono:

| | Antes | Después | Referencia |
|---|---:|---:|---:|
| md00a — confianza | 0,205 | **0,901** | 0,925 |
| md00a — RMD | 625,1 | **20,2** | 20,8 |

## Tiempos por etapa

El motor de la app mide tres etapas y las emite por `adb logcat`: `preprocess` (decodificar,
escalar y construir el tensor), `inference` (solo la llamada al modelo, sin IO) y `pipeline` (de la
imagen al resultado guardado y navegado). Estas son las 27 fotos del corpus entrando por galería,
en caliente, con el modelo y el escalado ya corregidos:

| etapa | mediana | media | mín | máx | p95 |
|---|---:|---:|---:|---:|---:|
| `preprocess` | 41 ms | 157 ms | 19 ms | 675 ms | 515 ms |
| `inference` | 60 ms | 68 ms | 34 ms | 141 ms | 110 ms |
| `pipeline` | 473 ms | 548 ms | 336 ms | 977 ms | 897 ms |

El primer escaneo de cada sesión paga la carga del modelo. Medido sobre el build final, tras
`force-stop`: `preprocess` 386 ms, `inference` 58 ms, `pipeline` 810 ms. El sobrecoste está en el
`pipeline`, no en la inferencia: el modelo ya está cargado cuando `runSync` arranca su cronómetro.

**La inferencia no es el cuello de botella.** El modelo tarda 60 ms medianos; el preprocesado varía
veinte veces entre la foto más pequeña y la más grande del lote, porque el trabajo es proporcional
a los píxeles que hay que promediar. Y el `pipeline` completo es casi un orden de magnitud mayor que
la suma de los dos, porque incluye crear el registro en la base local, navegar a la pantalla de
resultado y persistir el escaneo.

Una captura real con la cámara del teléfono, que es el caso de uso de verdad:

| | |
|---|---|
| Resolución de la foto | 3072 × 4096 (12,6 MP) |
| `preprocess` | 75 ms |
| `inference` | 61 ms |
| `pipeline` | 289 ms |

El preprocesado de una foto de 12,6 MP cuesta 75 ms **porque se recorta al marco guía antes de
escalar**: el recorte entrega 3,02 MP al escalador en vez de 12,58.

::: warning El recorte no está dentro de ninguna de las tres medidas
`handleCapture` recorta la foto y solo después llama a `persistScan`, que es quien abre la medición
`pipeline`. El recorte reabre el JPEG, lo recodifica y lo escribe, así que la espera real del usuario
tras pulsar el disparador es mayor que los 289 ms de la tabla. Medirla requiere instrumentar
`handleCapture`, cosa que este ensayo no hizo.
:::

## Inferencias sobre el corpus limpio

Las 27 fotos del split de test, tres por clase, entrando por la galería tal cual. Cada resultado se
verificó contra la firma del tensor para confirmar que el teléfono procesó la imagen que se le
envió.

| clase | n | acierta | muestra «no reconocida» | confianza mediana |
|---|---:|---:|---:|---:|
| `common_rust` | 3 | 3 | 1 | 0,908 |
| `fall_armyworm` | 3 | 2 | 1 | 0,874 |
| `gray_leaf_spot` | 3 | 3 | 0 | 0,939 |
| `healthy` | 3 | 3 | 1 | 0,806 |
| `lethal_necrosis` | 3 | 3 | 0 | 0,840 |
| `nitrogen_deficiency` | 3 | 2 | 1 | 0,866 |
| `northern_corn_leaf_blight` | 3 | 3 | 1 | 0,863 |
| `phosphorus_deficiency` | 3 | 3 | 2 | 0,907 |
| `potassium_deficiency` | 3 | 3 | 0 | 0,978 |

**25 de 27 con la clase correcta**, el mismo número que obtiene el artefacto corrido en CPU sobre
las mismas imágenes. La confianza media es 0,874 y ninguna predicción satura.

Lo más útil del reparto no es el acierto sino dónde caen los errores. Las dos fallidas son:

| | clase real | predicción | confianza | RMD | qué mostró la app |
|---|---|---|---:|---:|---|
| md04a | gusano cogollero | roya común | 0,600 | 480,9 | no reconocida |
| md16a | deficiencia de nitrógeno | necrosis letal | 0,777 | 169,5 | no reconocida |

**Las dos las atrapa el detector OOD antes de llegar a la pantalla.** De las 27 fotos, la app afirmó
un diagnóstico en 20 y acertó las 20; en las otras 7 se negó a diagnosticar. El coste de esa
prudencia son las 5 fotos que habría acertado y rechazó igual: el umbral está en el percentil 95 de
validación, así que rechazar en torno a una de cada veinte imágenes de dominio es el precio de
diseño, y aquí sale más caro que eso.

![Diagnóstico de deficiencia de potasio](/app/resultado-potasio.png)

![Imagen que el detector rechaza](/app/resultado-no-reconocida.png)

## El problema del encuadre

El modelo se entrenó estirando cada imagen a 224 × 224 **sin preservar el aspecto y sin recortar**.
Lo que entra en el encuadre es, literalmente, lo que el modelo ve. Y el corpus de entrenamiento son
sobre todo primeros planos de una hoja: de una muestra de 24 fotos de campo real del split de test
revisadas una a una, solo 5 muestran la planta pequeña dentro de la escena; el resto son la hoja
llenando el cuadro. Si el usuario dispara desde lejos, la mayor parte del tensor pasa a ser suelo,
cielo y plantas vecinas, algo que el modelo apenas vio.

La app responde con un marco guía que no es decorativo: `cropPhotoToOverlay` recorta la foto a la
región que el marco encierra antes de que el modelo la vea. Con las dimensiones reales medidas en
el dispositivo:

| | |
|---|---|
| Foto capturada | 3072 × 4096 px |
| Visor de cámara | 432 × 782,4 dp |
| Marco guía | 260 × 320 dp, centrado, con 15 % de margen |
| Región conservada | 1565 × 1927 px en (754, 1085) |
| Fracción del encuadre | 50,9 % de ancho × 47,0 % de alto = **24,0 % del área** |

Es decir: de los 12,58 MP que captura el sensor, el modelo recibe 3,02 MP. Tres cuartas partes de
la foto se descartan por diseño, y esa es la razón de que el banner insista en «acérquese a 20–30 cm
y llene el marco»: lo que quede fuera del marco no se analiza, y lo que quede dentro sin ser hoja sí.

::: warning Un fondo uniforme se diagnostica con confianza
Una comprobación de control lo deja claro. Tomando cinco imágenes del corpus, desenfocándolas hasta
eliminar toda estructura de hoja y pasando **solo ese fondo, sin ninguna hoja**, el modelo responde
`fall_armyworm` con confianza entre 0,806 y 0,873 en los cinco casos, y el detector OOD solo rechaza
uno.

No es un fallo de la cuantización ni del escalado: es el mismo atajo de procedencia que
[Procedencia y fuga](/es/provenance/) mide sobre el corpus, donde un anillo del 10 % del borde
—sin hoja ni lesión— clasifica el 78,3 % del test. Un encuadre que llene el marco de vegetación de
fondo cae justo en ese modo de fallo, y ni la confianza ni el detector OOD avisan.
:::

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
