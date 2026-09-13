# App React Native (Expo)

Esta guía describe **qué tiene que hacer la app** para que un modelo exportado por el pipeline principal produzca en el teléfono exactamente las mismas predicciones que produce en el servidor. Está escrita para soportar **los dos formatos a la vez** —TFLite y ONNX— con **uno solo activo** según configuración.

El principio que ordena todo lo demás: el archivo `.tflite` o `.onnx` no es un clasificador completo. Es solo la parte central. Antes hay un preprocesado y después una interpretación de la salida, y ambos se entrenaron con una convención concreta. Si la app se desvía de esa convención, el modelo no falla ni avisa: simplemente empieza a acertar menos, y el error se ve como "el modelo es malo" en vez de como "el preprocesado no coincide".

## Qué produce el pipeline

`make export-main` deja los artefactos bajo `outputs/main/<modelo>/<run_id>/export/`:

| Archivo | Qué es |
|---|---|
| `model.onnx` | ONNX FP32, archivo único |
| `model.tflite` | TFLite FP32 |
| `model_int8.onnx` | ONNX cuantizado a int8 |
| `model_int8.tflite` | TFLite cuantizado a int8 |
| `export_summary.json` | Resultado de la conversión y la paridad numérica. Cada entrada de `formats[]` incluye `sha256` del archivo exportado — usarlo para verificar que una copia (por ejemplo, la que se empaqueta en la app) no se corrompió ni quedó desactualizada |
| `eval_<formato>.json` | Métricas del artefacto sobre el split de test completo |
| `labels.json` | Orden de clases del modelo: `{"schema_version": int, "model": str, "image_size": [h, w], "labels": list[str]}`, donde `labels[i]` es el nombre de clase del índice de salida `i`. Se escribe una sola vez por run, no por formato ni por variante de cuantización — el orden de clases no cambia entre ellos |

`make compute-ood-stats` (aparte de `export-main`, sobre el mismo checkpoint) deja además `ood_stats.json` en el mismo directorio — centroides, covarianza y umbral para el detector OOD por distancia de Mahalanobis. Ver [Detección OOD](../deep-learning/ood-detection.md) para el método y [App React Native](#el-contrato-del-modelo) más abajo para cómo lo consume la app. `sync_mobile_model.py` copia los tres archivos (`.tflite`, `labels.json`, `ood_stats.json`) juntos y avisa si falta alguno.

Tamaños de los artefactos reales del pipeline (9 clases, 224×224):

| Modelo | ONNX FP32 | ONNX int8 | TFLite FP32 | TFLite int8 |
|---|---|---|---|---|
| `shufflenet_v2_x1_0` | 5.12 MB | 1.48 MB | 4.93 MB | 1.42 MB |
| `efficientnet_lite0` | 13.05 MB | 3.43 MB | 12.92 MB | 3.56 MB |
| `efficientnet_b0` | 15.90 MB | 4.39 MB | 15.48 MB | 4.44 MB |

::: tip Un solo archivo
El export de ONNX consolida los pesos dentro del `.onnx`. El exportador de PyTorch por defecto los saca a un `model.onnx.data` aparte, y un `.onnx` sin su `.data` al lado carga pero falla al inferir — un modo de fallo especialmente molesto de diagnosticar dentro de un bundle móvil. El pipeline los une para que solo haya que empaquetar un archivo.
:::

## El contrato del modelo

Esto es lo que no se puede cambiar sin romper la equivalencia con el entrenamiento.

**Entrada**

- Tensor `float32` de forma `[1, 3, 224, 224]` — layout **NCHW** (canales antes que alto/ancho).
- Canales en orden **RGB**, no BGR.
- Rango `[0, 1]` (dividir por 255) y después normalización por canal:
  - `mean = [0.485, 0.456, 0.406]`
  - `std  = [0.229, 0.224, 0.225]`
  - es decir `valor = (pixel/255 - mean[c]) / std[c]`.
- El nombre del tensor de entrada en ONNX es `input`; en TFLite se accede por índice.
- **Esto aplica igual a la variante `int8`.** La cuantización de este pipeline es solo de pesos (dynamic per-channel PTQ), no de activaciones: los tensores de entrada/salida del grafo siguen siendo `float32` en ambas variantes, nunca `uint8`/`int8`. Si al cargar `model_int8.tflite` el runtime reporta un tensor de entrada no-float32, es un bug de exportación, no el comportamiento esperado.

**Salida**

- **Dos tensores.** El primero, `float32` de forma `[1, 9]`, son **logits** (no probabilidades — la app debe aplicar softmax si quiere mostrar confianza). El segundo, `float32` de forma `[1, feature_dim]` (1280 en `efficientnet_lite0`), es el vector de features pooled de la penúltima capa (antes del head de clasificación) — lo consume el detector de imágenes fuera de dominio por distancia de Mahalanobis, ver [Detección OOD](../deep-learning/ood-detection.md). El checkpoint se envuelve en `FeatureExposedModel` (`src/models/feature_exposed.py`) antes de exportar precisamente para exponer esta segunda salida sin tocar el primer output.
- Ambas salidas son `float32` en la variante `int8` por el mismo motivo que la entrada (cuantización solo de pesos).
- El nombre del tensor de salida de logits en ONNX es `output`.
- El índice de cada clase (en la salida de logits) es su posición en `config/dataset.yaml`, en este orden exacto:

```
0  common_rust
1  fall_armyworm
2  gray_leaf_spot
3  healthy
4  lethal_necrosis
5  nitrogen_deficiency
6  northern_corn_leaf_blight
7  phosphorus_deficiency
8  potassium_deficiency
```

::: warning No hardcodear el orden de clases
Ese orden es el de `class_to_idx`, que queda guardado en el `summary.json` del run. Si el YAML cambia y la app tiene la lista quemada, las predicciones se renombran solas y en silencio: el modelo dirá "healthy" donde quería decir "lethal_necrosis". El pipeline ya resuelve esto: `export_model()` escribe automáticamente `export/labels.json` junto a cada artefacto exportado, con las etiquetas en orden de índice. La app debe leer el orden de clases de ese archivo en tiempo de ejecución, nunca hardcodearlo ni re-derivarlo por su cuenta.
:::

## Los pasos del pipeline en la app

### 1. Obtener la imagen

Cámara o galería. Nada especial todavía, salvo conservar el archivo original: no conviene dejar que el componente de UI haga un resize "de cortesía" antes de que empiece el preprocesado, porque ese resize no será el que espera el modelo.

### 2. Corregir la orientación EXIF

Las fotos de teléfono suelen venir con la rotación en metadatos EXIF en vez de aplicada a los píxeles. El pipeline de entrenamiento corrige esto siempre (`ImageOps.exif_transpose`), así que la app también debe hacerlo. Saltárselo significa alimentar el modelo con hojas giradas 90°, y aunque el entrenamiento incluye rotaciones, no cubre ese caso de forma sistemática.

### 3. Redimensionar a 224×224 **deformando el aspecto**

Este es el paso que más se hace mal. El pipeline usa un `Resize((224, 224))` directo, que **estira la imagen** hasta esa forma sin preservar la proporción y sin recortar. Es una decisión deliberada del proyecto.

Casi todas las librerías de imagen en móvil hacen lo contrario por defecto: mantienen el aspecto y recortan al centro, o rellenan con barras. Cualquiera de las dos cosas produce una imagen que el modelo nunca vio durante el entrenamiento.

- ✅ Estirar a 224×224 exactos.
- ❌ Center-crop tras escalar el lado corto.
- ❌ Letterbox / padding.

### 4. Convertir a RGB y a `[0, 1]`

Extraer los píxeles como RGB (descartar alfa) y dividir por 255.

### 5. Normalizar por canal

Aplicar `mean` y `std` de ImageNet indicados arriba. Es una resta y una división por canal, nada más, pero tiene que ir **después** de la división por 255, no antes.

### 6. Ordenar el tensor como NCHW

Los buffers de imagen en móvil salen casi siempre como `[alto, ancho, canales]` (NHWC). El modelo espera `[1, 3, 224, 224]`. Hay que transponer: primero todos los valores del canal R, luego los del G, luego los del B.

```ts
// pixels: Uint8Array en orden RGBA, longitud 224*224*4
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SIDE = 224;
const input = new Float32Array(3 * SIDE * SIDE);

for (let i = 0; i < SIDE * SIDE; i++) {
  for (let c = 0; c < 3; c++) {
    // canal c completo antes del siguiente => NCHW
    input[c * SIDE * SIDE + i] = (pixels[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
}
```

### 7. Inferir e interpretar

Correr el modelo, aplicar softmax sobre los 9 logits y mapear el índice ganador a su nombre de clase.

```ts
function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max)); // -max evita overflow
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum) as Float32Array;
}
```

Conviene además fijar un umbral de confianza por debajo del cual la app no afirma un diagnóstico. El modelo siempre devuelve una de las 9 clases, incluso si le muestran una mano o el cielo: sin umbral, la app diagnostica con total seguridad sobre cualquier cosa.

## Los dos runtimes, uno activo

La forma de sostener ambos formatos sin duplicar la app es aislar el runtime detrás de una interfaz y dejar que la configuración elija. Los pasos 1–6 son idénticos para TFLite y ONNX —el contrato de entrada es el mismo—, así que lo único que cambia es quién ejecuta el modelo.

```ts
// Todo lo específico de cada runtime vive detrás de esto.
export interface InferenceBackend {
  load(): Promise<void>;
  predict(input: Float32Array): Promise<Float32Array>; // devuelve 9 logits
  dispose(): void;
}
```

```ts
// config.ts — un solo lugar decide qué se embarca y qué se activa.
export const MODEL_CONFIG = {
  backend: "tflite" as "tflite" | "onnx",
  asset: "model_int8.tflite",
  inputSize: 224,
  confidenceThreshold: 0.6,
} as const;
```

La fábrica selecciona la implementación a partir de `MODEL_CONFIG.backend`, y el resto de la app solo conoce `InferenceBackend`. Con eso, cambiar de runtime es cambiar una constante y el asset, sin tocar preprocesado ni UI.

Dos advertencias prácticas sobre esta estructura:

- **El bundle crece con lo que empaquetes, no con lo que actives.** Si el `.tflite` y el `.onnx` van los dos en el bundle, el APK carga los dos aunque solo se use uno. Si el objetivo es tamaño, empaqueta solo el activo y deja el otro detrás de una variante de build.
- **Las librerías nativas también pesan.** Incluir los dos runtimes suma dos librerías nativas al binario. Mantener ambos soportados en el código no obliga a mantener ambos instalados: la dependencia del runtime inactivo puede quedar fuera de la build de release.

## Expo: hace falta development build

Los runtimes de inferencia son código nativo. Eso significa que **Expo Go no sirve**: no puede cargar módulos nativos que no vengan precompilados en él.

El flujo es el de siempre en Expo cuando entra código nativo:

1. Instalar la librería del runtime elegido.
2. Generar un [development build](https://docs.expo.dev/develop/development-builds/introduction/) con EAS Build o `expo run:android`.
3. Desarrollar contra ese build en vez de Expo Go.

El modelo se embarca como asset. Conviene registrar la extensión en `metro.config.js`, porque Metro no reconoce `.tflite` ni `.onnx` por defecto:

```js
// metro.config.js
const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push("tflite", "onnx");
module.exports = config;
```

Y en muchos runtimes el modelo debe existir como **archivo en disco**, no como módulo empaquetado: hay que resolver el asset y copiarlo al sistema de archivos la primera vez que arranca la app.

## Qué artefacto embarcar

- **`shufflenet_v2_x1_0` int8** es el candidato natural para gama media/baja: 1.37 MB en TFLite, el más pequeño de los tres por un margen amplio.
- **FP32 frente a int8**: la int8 reduce el tamaño unas 3.5×. La latencia *no* mejora necesariamente — en algunos dispositivos el modelo cuantizado corre más lento —, así que si el objetivo es velocidad hay que medirlo en el teléfono, no asumirlo.
- **Nunca embarcar una variante int8 sin evaluarla.** La cuantización cambia los números a propósito, y esa pérdida se concentra justo en las clases minoritarias, que son las que más importan aquí. Para eso existe `make eval-export-main`.

::: danger El artefacto puede calcular cosas distintas según el runtime
`make eval-export-main` construye el intérprete con sus delegados por defecto, es decir con
XNNPACK. El runtime que embarca la app **no** lo activa, y para algunas configuraciones de
cuantización los dos no calculan lo mismo. Un `FULLY_CONNECTED` dinámico con escalas de peso
**por canal** es una de ellas: el kernel integrado aplica una sola escala por tensor y devuelve
los logits multiplicados por `1/escala_c`, distinta por clase, sin error ni aviso.

Por eso `validate_tflite_parity` corre el artefacto con y sin delegado y falla el export si las
probabilidades difieren más de 0,05. El acuerdo top-1 no basta como señal: cuando esto se
detectó, las dos configuraciones coincidían en la clase predicha en el 100 % de la muestra y aun
así diferían hasta 0,4312 en probabilidad. Evidencia completa en
[Prototipo en dispositivo](/es/deployment/prototipo).
:::

::: warning CLAHE
Si el modelo se entrenó con `--clahe`, la app tendría que aplicar CLAHE con los mismos parámetros antes de normalizar, o las predicciones no coincidirán. Reimplementar CLAHE en JS es trabajo extra y una fuente de divergencia sutil. La recomendación es entrenar **sin** CLAHE el modelo destinado a móvil.
:::

## Verificar que la app coincide con el servidor

La forma barata de comprobar que el preprocesado quedó bien, antes de discutir si el modelo es bueno:

1. Elegir una imagen del split de test.
2. Correr `make eval-export-main` y buscar esa imagen en `eval_<formato>_predictions.csv`, que trae la predicción y la confianza que produce el artefacto en el servidor.
3. Pasar la misma imagen por la app.
4. Los logits deben coincidir con varios decimales.

Si la clase coincide pero la confianza no, suele ser la normalización. Si ni la clase coincide, casi siempre es una de tres: el resize preservó el aspecto, el tensor quedó en NHWC, o los canales están en BGR.

## Regenerar los artefactos

```bash
# Los 3 modelos, ambos formatos, FP32
make export-main EXPORT_FORMATS=onnx,tflite

# Variante int8 (no pisa los FP32)
make export-main EXPORT_FORMATS=onnx,tflite QUANTIZE=int8

# Medir el artefacto sobre el split de test completo
make eval-export-main EXPORT_FORMATS=onnx,tflite
make eval-export-main EXPORT_FORMATS=onnx,tflite QUANTIZE=int8

# Un solo modelo
make export-main MAIN_MODELS=shufflenet_v2_x1_0 EXPORT_FORMATS=tflite QUANTIZE=int8

# Estadisticas del detector OOD (mismo checkpoint que el export)
make compute-ood-stats MAIN_MODELS=efficientnet_lite0

# Copiar el trio (.tflite + labels.json + ood_stats.json) a la app
make sync-mobile-model RUN_DIR=outputs/main/efficientnet_lite0/<run_id> \
  DEST=../maize-doctor-app/assets/model
```

Los mismos flujos existen en Modal con el prefijo `modal-` (`modal-export-main`, `modal-eval-export-main`). Ver [GPU en Modal](/es/deployment/modal).
