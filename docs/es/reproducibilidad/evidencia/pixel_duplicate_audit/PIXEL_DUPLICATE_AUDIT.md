# Auditoría de duplicados exactos por píxeles

## Identidad y cobertura

- Estado: complete; fecha UTC: 2026-09-29T18:20:00.910892+00:00.
- Commit: ef3ebe67c05ed1ad18fed39ad0f7055d028440a2 (dirty=True).
- Fuente canónica: Modal Volume corn-clean:/clean; verified against corn-outputs:/splits/seed_42.
- No se utilizó el ZIP local filtrado de aproximadamente 28 360 imágenes.
- Master manifest SHA-256: 64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163.
- manifest.lock.json SHA-256: 0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8.
- 33,429 muestras elegibles y 8 exclusiones verificadas.
- Preflight: 33,437/33,437 rutas y SHA correctos; errores de decodificación: 0.

## Hashes y grupos

| Población | Hash | Grupos duplicados | Cross-split | Conflictos |
|---|---|---:|---:|---:|
| Elegibles | file_sha256 | 0 | 0 | 0 |
| Elegibles | pixel_sha256 | 0 | 0 | 0 |
| Ocho exclusiones | file_sha256 | 4 | no aplica | 4 |
| Ocho exclusiones | pixel_sha256 | 4 | no aplica | 4 |

file_sha256 identifica bytes idénticos. pixel_sha256 identifica exactamente el
mismo raster RGB tras orientación EXIF y conversión a RGB; incorpora ancho,
alto y bytes en orden de filas. No aplica resize, aumentos ni perfiles ICC.
Los archivos recodificados pueden tener bytes distintos y píxeles iguales.

- Grupos con píxeles iguales y SHA de archivo distinto: 0.

## Particiones y etiquetas

- Grupos internos: 0; entre particiones: 0.
- Train/validation: 0; train/test: 0.
- Validation/test: 0; tres particiones: 0.
- Test afectado: 0/5015 (0.0000 %).
- Conflictos nuevos: 0; conflictos excluidos: 4 grupos / 8 archivos.
- Fuente de las exclusiones: maize-beans-tomatoes-africa; atribución por familia de nombre unívoca en master_manifest.csv.

## Hallazgo histórico y alcance

La observación histórica de aproximadamente 15 grupos y 0.68 % del test
con gemelos no se reproduce en esta materialización canónica: 0 grupos
elegibles y 0/5,015 imágenes de test afectadas. Puede corresponder a otra
versión del corpus, otra materialización o un procedimiento distinto.
Esta auditoría no demuestra que el hallazgo nunca existió.

Se conserva seed_42. Las métricas baseline y HPO no requieren invalidación
por duplicados exactos cross-split en esta materialización. La auditoría
no resuelve generalización entre fuentes ni similitud perceptual, recortes,
resize, rotaciones, cambios fotométricos o fotografías relacionadas.

## Configuración y artefactos

- SHA-256 de dataset.yaml en lock: 8f0e560fe78267bb3f96f907d4668e29f1dd22d6a6645f474a84d1fad4db4730.
- SHA-256 actual: 8e3f5a4feb6947db4d2b33f3b04c963f93096cd6a5e6aacfcb89eb3040010fbc.

El drift se analiza en [dataset_yaml_drift_report.md](dataset_yaml_drift_report.md).
Los hashes de los CSV
canónicos coinciden con el lock; la diferencia del YAML no cambia sus bytes.

- pixel_hash_inventory.csv: una fila por muestra elegible.
- pixel_excluded_images.csv: ocho exclusiones auditadas por separado.
- pixel_duplicate_groups.csv y pixel_cross_split_duplicates.csv: grupos elegibles.
- pixel_label_conflicts.csv: los cuatro conflictos ya excluidos.
- corpus_preflight.csv/json: correspondencia de rutas y SHA de archivo.

La copia de trabajo en /tmp/doctormaiz_pixel_audit_20260929/ contiene los
manifiestos y una descarga local temporal del corpus. La fuente canónica
sigue siendo corn-clean:/clean. La copia puede eliminarse con autorización
posterior del usuario y no debe incluirse en Git.
