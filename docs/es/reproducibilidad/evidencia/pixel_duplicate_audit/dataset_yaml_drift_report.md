# Diferencia entre dataset.yaml y el lock de seed_42

Fecha de revisión: 29 de septiembre de 2026. Esta nota se refiere al split
vigente seed_42, no a una nueva materialización.

## Identidad comprobada

| Artefacto | SHA-256 |
|---|---|
| config/dataset.yaml actual | 8e3f5a4feb6947db4d2b33f3b04c963f93096cd6a5e6aacfcb89eb3040010fbc |
| config/dataset.yaml registrado en manifest.lock.json | 8f0e560fe78267bb3f96f907d4668e29f1dd22d6a6645f474a84d1fad4db4730 |
| manifest.lock.json | 0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8 |
| master_manifest.csv | 64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163 |
| config/dataset_exclusions.csv actual y registrado | 4ca25fcb91d48ee6a99c5f9e46ed3d8ad787305e17af0d852a7eeacfbd7e6432 |

Los hashes de train.csv, val.csv y test.csv descargados de
corn-outputs:/splits/seed_42 coinciden con el lock: respectivamente
231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434,
6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5 y
08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724.

## Búsqueda de la versión anterior

Se calcularon los SHA-256 de 34 revisiones legibles de config/dataset.yaml en
todas las referencias Git locales (20 contenidos distintos). Ninguna coincide
con 8f0e560f…. Los siete ZIP locales que contienen config/dataset.yaml
(source_code.zip del HPO y source_snapshot.zip de multi-seed, incluidos sus
respaldos) contienen la versión 8e3f5a4f…. La copia del paquete de reportes HPO
también tiene ese hash. Las evidencias del HPO registran el hash actual; el lock
solo conserva el hash anterior, no los bytes del YAML anterior.

Por ello no se puede dar un diff de líneas entre ambas versiones ni atribuir
el cambio a un commit concreto. La diferencia exacta verificable es la de los
dos SHA-256 completos de la tabla. Describir una clave modificada sería una
inferencia sin respaldo.

## Comparación funcional que sí permite el lock

El lock guarda los parámetros efectivos usados al producir la partición. Su
seed es 42 y sus nueve clases, en orden, coinciden con dataset.seed y
dataset.classes del YAML actual. Registra estratificación por label y
environment, ratios 70/15/15, max_per_class nulo, group_by_source falso y
deduplicate_perceptual falso. El archivo de exclusiones conserva el mismo
SHA-256. El YAML actual apunta a clean y splits/seed_42; el lock no conserva
el valor anterior de esas rutas, por lo que no se puede comparar su texto.

El YAML actual también contiene target_size, augmentation, clahe, baseline,
lime, gradcam y shap. El lock no guarda los valores anteriores de esas secciones.
No es posible clasificar el cambio de bytes como exclusivamente documental,
de rutas, de metadatos o de preprocesamiento. La clasificación sustentada es
**drift de configuración fuente, de causa no localizada**.

## Impacto

La diferencia de configuración impide afirmar que una nueva generación con el
YAML actual reproduzca bit a bit la ejecución original. Es una limitación de
reproducibilidad del proceso de generación. No altera retroactivamente el
contenido congelado: master, train, val, test y exclusiones verifican contra el
lock, y el preflight confirmó 33 437 archivos y sus SHA-256. La composición
observada del corpus, las nueve clases y la membresía de los manifests vigentes
se mantienen verificadas; no se encontró un efecto del drift en esos artefactos
ni en las métricas ya calculadas sobre ellos.

No se puede determinar si la versión anterior modificaba rutas, opciones de
preprocesamiento o parámetros usados por otros comandos. Para nuevas corridas
debe conservarse el lock y registrar el YAML efectivo del entorno de ejecución.
No se reescribió el lock, el YAML ni el split.
