# Evidencia bruta

Artefactos sin procesar de los experimentos de procedencia y fuga. Se versionan para que
las cifras de los documentos sean auditables sin volver a ejecutar nada.

Los artefactos vivos se generan en `outputs/experiments/`, que está en `.gitignore`. Lo que
hay aquí es la copia congelada de cada ejecución que respalda un documento.

## Fase 0 — Auditoría de procedencia

Documento: [Fase 0](/es/provenance/fase-0-auditoria)

Generado con:

```bash
python scripts/experiments/provenance_audit.py
```

Ejecutado en local (Windows, RTX 5060) el 2026-09-09. No requiere GPU.

| Archivo | Contenido |
|---|---|
| `provenance_audit.json` | Verificación de fuentes, imágenes fuera de splits, duplicados exactos y barrido de casi-duplicados por umbral |
| `source_by_class.csv` | Tabla fuente × clase, 14 tokens × 9 clases |
| `source_by_split.csv` | Tabla fuente × partición, muestra que ninguna fuente queda retenida |
| `manifest_with_source.csv.gz` | Manifiesto completo: 33 437 filas con clase, entorno, fuente y partición |

## Fase 1 — Validación dejando una fuente fuera

Documento: [Fase 1](/es/provenance/fase-1-particion-honesta)

Generado con:

```bash
modal run scripts/modal/leave_one_source_out.py
modal run scripts/modal/leave_one_source_out.py --arm border_ring
modal volume get corn-outputs experiments/leave_one_source_out.json <destino>
```

Ejecutado en Modal sobre GPU A10 el 2026-09-09.

| Archivo | Contenido |
|---|---|
| `leave_one_source_out.json` | Resultado por pliegue y agrupado: métricas, predicciones por imagen y rutas |
| `leave_one_source_out.run.txt` | Traza completa de la ejecución en Modal, época a época |
| `loso_per_class.csv` | F1 por clase, comparando partición aleatoria y fuera de fuente |
| `leave_one_source_out_border_ring.json` | Igual que el anterior, con el brazo que solo ve el marco exterior |
| `leave_one_source_out_border_ring.run.txt` | Traza de esa ejecución |
| `loso_gate.csv` | Compuerta 1 con ambas condiciones bajo el mismo protocolo, semilla 0 |
| `loso_gate_3seeds.csv` | Compuerta 1 consolidada: media y desviación sobre tres semillas |
| `*_seed1.json`, `*_seed2.json` | Semillas adicionales de ambos brazos, con sus trazas `.run.txt` |

## Nota sobre `PYTHONIOENCODING`

El CLI de Modal falla en esta máquina con `'charmap' codec can't encode '✓'` si la consola
usa cp1252. Todas las invocaciones se hicieron con `PYTHONIOENCODING=utf-8`.

## Fase 2 — Intervenciones

Documento: [Fase 2](/es/provenance/fase-2-intervenciones)

Generado con:

```bash
make modal-loso DETACH=1 BALANCE=1
make modal-loso DETACH=1 BALANCE=1 ARM=border_ring
make modal-loso DETACH=1 BACKMIX=0.5
make modal-loso DETACH=1 BACKMIX=0.5 ARM=border_ring
```

Ejecutado en Modal sobre GPU A10 el 2026-09-10, cuatro corridas en paralelo.

| Archivo | Contenido |
|---|---|
| `leave_one_source_out_balanced*.json` | Balanceo de grupos, ambos brazos, con sus trazas `.run.txt` |
| `leave_one_source_out_backmix0.5*.json` | BackMix 0,5, ambos brazos, con sus trazas |
| `fase2_gate.csv` | Compuerta 2 aplicada: F1 y recuperación del marco por clase e intervención |
| `augmentation_none.json`, `augmentation_hardened.json` | Fase 2b, diseño B: un entrenamiento evaluado con imagen completa y con sólo el marco, con sus trazas |
| `augmentation_crop.json`, `augmentation_codec.json`, `augmentation_colour.json` | Fase 2c, ablación por componentes |
| `augmentation_crop_colour.json` | Combinación de las dos palancas |
| `ablacion_augmentation.csv`, `ablacion_augmentation_por_clase.csv` | Tablas resumen de la ablación |

## Nota sobre los targets del Makefile

`MODAL` y `PYTHON` usan rutas con barra invertida de Windows, así que los targets funcionan
desde PowerShell pero no desde Git Bash, donde la barra se pierde. Las corridas de este
directorio se lanzaron invocando Modal directamente:

```bash
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe -m modal run --detach     scripts/modal/leave_one_source_out.py --balance-groups --arm border_ring
```
