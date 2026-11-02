# Respaldo de modelos DoctorMaiz

Paquete con los checkpoints `best.pth` ya entrenados de las corridas del proyecto y una copia de sus resultados. Contiene 18 modelos; `RESULTS_SUMMARY.md` resume las cifras de cada experimento.

## Contenido

`model_index.csv` lista cada `best.pth` con arquitectura, protocolo, semilla, fuente retenida, época, Macro-F1, ruta original, ruta dentro del paquete, tamaño y SHA-256. `model_comparison_results.csv` añade las métricas de prueba o LOSO, el ECE y las clases evaluadas cuando existen. Las celdas `N/A` indican que esa medición no existe para ese protocolo.

| Grupo | Contenido | Archivo de distribución |
|---|---|---|
| `baseline_lite0/` | Lite0 estándar, semillas 42/123/2026 | `DoctorMaiz_Baseline_Lite0.tar.gz` |
| `hpo_lite0/` | Lite0 HPO, trial 0 | `DoctorMaiz_HPO_Lite0.tar.gz` |
| `loso_maize_diseases/` | Lite0 sin `maize-diseases` en entrenamiento ni selección, tres semillas | `DoctorMaiz_LOSO_MaizeDiseases.tar.gz` |
| `loso_multicrop/lite0/` | Lite0 sin `multicrop-disease-maiz`, tres semillas | `DoctorMaiz_LOSO_Multicrop_Lite0.tar.gz` |
| `loso_multicrop/b0/` | B0 sin `multicrop-disease-maiz`, tres semillas | `DoctorMaiz_LOSO_Multicrop_B0.tar.gz` |
| `loso_multicrop/shufflenet/` | ShuffleNet-V2-x1.0 sin `multicrop-disease-maiz`, tres semillas | `DoctorMaiz_LOSO_Multicrop_ShuffleNet.tar.gz` |
| `historical_standard/` | B0 y ShuffleNet históricos del test estratificado original | `DoctorMaiz_Historicos_B0_ShuffleNet.tar.gz` |
| Archivos de la raíz y `metadata/` | Índices, checksums, este README, resultados y procedencia | `DoctorMaiz_Modelos_Metadata_Resultados.tar.gz` |

Cada carpeta de corrida tiene `best.pth`, `summary.json`, `config_snapshot.json` (extraído del summary) y `checkpoint_reference.json`. Cuando existen, también incluye `selection.lock.json` y las métricas finales por clase. El trial de HPO incluye `hpo_selection_lock.json` y los B0 y ShuffleNet multicrop, el lock conjunto del ensamble. La configuración de cada corrida está en su summary y su snapshot.

`metadata/evidence_references.json` enumera los hashes de los documentos fuente y `metadata/failed_runs_manifest.json` identifica los dos intentos interrumpidos, que no forman parte del índice. Ambos están versionados en `docs/es/reproducibilidad/evidencia/model_package/`.

## Cargar un modelo

1. Buscar el `model_id` en `model_index.csv` y abrir el `local_path`, relativo a esta carpeta. Se carga siempre `best.pth`.
2. Leer `summary.json` y `config_snapshot.json`: contienen la arquitectura, `class_to_idx`, el preprocesado y la huella del split.
3. Cargarlo con el código del proyecto compatible con la corrida y su cargador contractual. El paquete no incluye el dataset ni el entorno.

Para comparar modelos conviene filtrar `model_comparison_results.csv` por `experiment` y `evaluation_split`: el test estándar y un holdout LOSO miden cosas distintas.

## Verificar la integridad

```bash
sha256sum -c model_checksums.sha256
```

El comando se ejecuta desde una carpeta que contenga las siete carpetas `DoctorMaiz_*` de checkpoints junto a `DoctorMaiz_Modelos_Metadata_Resultados`. La suma de cada checkpoint también figura en `model_index.csv`.

## Distribución

`bash CREATE_TARS.sh` genera los ocho archivos `.tar.gz` de la tabla, los comprueba con `tar -tzf` y registra `TAR_CHECKSUMS.sha256`.

## Ensambles

El ensamble histórico de tres miembros (Lite0 `20260812_221429`, B0 `20260910_170120` y ShuffleNet `20260910_184521`) se reconstruye con los B0 y ShuffleNet de este paquete y el Lite0 de la descarga de artefactos de la etapa 2. El ensamble LOSO multicrop no tiene un `best.pth` propio: se forma con los tres checkpoints de la misma semilla y el promedio uniforme de sus softmax. Las predicciones completas permanecen en los artefactos originales.
