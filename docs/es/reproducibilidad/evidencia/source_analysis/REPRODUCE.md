# Reproducción local del análisis por fuente

Los CSV de predicciones usados están copiados en este directorio. Sus SHA-256
figuran en `source_analysis_summary.json`. Los manifests congelados deben
descargarse de `corn-outputs:/splits/seed_42` sin editarlos; el script comprueba
sus cinco hashes canónicos antes de calcular una cifra. `master_manifest.csv`
debe estar junto a `train.csv`, `val.csv`, `test.csv` y `manifest.lock.json`.

Desde la raíz del repositorio, con el entorno de Python del proyecto activo:

```bash
SPLITS_DIR=/ruta/a/splits/seed_42
EVIDENCE_DIR=docs/es/reproducibilidad/evidencia/source_analysis

python -m scripts.experiments.source_analysis \
  --splits-dir "$SPLITS_DIR" \
  --baseline-validation "$EVIDENCE_DIR/baseline_validation_predictions.csv" \
  --baseline-test "$EVIDENCE_DIR/baseline_test_predictions.csv" \
  --hpo-validation "$EVIDENCE_DIR/hpo_validation_predictions.csv" \
  --hpo-test "$EVIDENCE_DIR/hpo_test_predictions.csv" \
  --output-dir "$EVIDENCE_DIR"

python -m scripts.experiments.source_metadata \
  --inventory docs/es/reproducibilidad/evidencia/pixel_duplicate_audit/pixel_hash_inventory.csv \
  --output-dir "$EVIDENCE_DIR"

python -m scripts.experiments.plot_source_analysis --evidence-dir "$EVIDENCE_DIR"
```

La única inferencia nueva de esta fase fue la de validation del baseline,
realizada una vez con `best.pth` del run `20260921_204608`. Su recibo verifica
checkpoint, split y Macro-F1 exacto. Para repetirla se necesita una copia íntegra
verificada de `corn-clean:/clean` y el checkpoint original:

```bash
DATASET_ROOT=/ruta/a/corpus \
python -m scripts.experiments.infer_baseline_validation \
  --checkpoint /ruta/al/run/20260921_204608/best.pth \
  --splits-dir "$SPLITS_DIR" \
  --output-dir "$EVIDENCE_DIR"
```

No se necesita repetir esa inferencia para reproducir el análisis de tablas;
el CSV resultante ya está incluido. Ningún comando entrena, modifica splits ni
vuelve a inferir sobre test; solo relee sus predicciones congeladas.
