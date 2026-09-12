ifeq ($(OS),Windows_NT)
    PYTHON 	:= venv\Scripts\python
    PIP    	:= venv\Scripts\pip
    RUFF   	:= venv\Scripts\ruff
	PYRIGHT := venv\Scripts\pyright
	MODAL   := venv\Scripts\modal
else
    PYTHON 	:= venv/bin/python
    PIP    	:= venv/bin/pip
    RUFF   	:= venv/bin/ruff
	PYRIGHT := venv/bin/pyright
	MODAL   := venv/bin/modal
endif

# ==============================================================================
# Variables
# ==============================================================================

# Modelos: MODELS aplica a los baselines, MAIN_MODELS al pipeline principal.
MODELS ?= efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0
MAIN_MODELS ?= efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0

# Explicabilidad local: directorio raíz de runs. Vacío = default del script
# (outputs/baselines).
OUTPUT_DIR ?=
MAIN_OUTPUT_DIR ?= outputs/main

# Explicabilidad en Modal: el directorio se elige por nombre de pipeline
# ("baselines" | "main") y lo resuelve el contenedor sobre el Volume corn-outputs.
# Pasar rutas absolutas aquí no funciona: MSYS (Git Bash) reescribe /outputs/... a
# una ruta de Windows antes de que llegue al comando.
PIPELINE ?= baselines

# Entrenamiento
EPOCHS ?= 30
MAIN_EPOCHS ?=
SPLITS_DIR ?=
SEGMENTED ?=
CLASS_WEIGHTS ?=
CLAHE ?=
NO_CAP ?=
MAX_PER_CLASS ?=
BASELINE ?=
REGEN_SPLITS ?=
BATCH_SIZE ?=
IMAGE_SIZE ?=
LEARNING_RATE ?=
WEIGHT_DECAY ?=
NUM_WORKERS ?=
NO_PRETRAINED ?=
LIME ?=
DETACH ?=

# Optuna HPO & K-Fold Cross Validation
N_TRIALS ?= 20
TIMEOUT ?=
PRUNER ?= median
BASELINE_F1 ?= 0.9146
K_FOLDS ?= 5

# Exportacion (ONNX/TFLite). Vacia por defecto: opt-in via EXPORT_FORMATS=onnx[,tflite].
# QUANTIZE=int8 produce model_int8.<fmt> junto al FP32, sin pisarlo.
EXPORT_FORMATS ?=
QUANTIZE ?=
NO_PARITY ?=
TOLERANCE ?=
MIN_AGREEMENT_RATE ?=
PARITY_SAMPLE_SIZE ?=
NO_TORCH_BASELINE ?=
MAX_MACRO_F1_DROP ?=

# Explicabilidad e inferencia
NUM_SAMPLES ?=
SAMPLE_SIZE ?=
NSAMPLES ?=
MODEL ?= efficientnet_b0
IMAGE ?=
OUTPUT ?=
CHECKPOINT ?=
RUN ?=
TOP_K ?=
STABILITY_RUNS ?=

.DEFAULT_GOAL := help

# ==============================================================================
# Ayuda
# ==============================================================================

.PHONY: help
help:
	@echo "Local - setup y datos:"
	@echo "  install download-dataset upload-dataset splits splits-baseline summary test-loader"
	@echo "    upload-dataset requiere STAGE_DIR=<dir con ~18GB libres> [DRY_RUN=1] [KEEP_STAGE=1]"
	@echo ""
	@echo "Local - baselines (runs en outputs/baselines, var MODELS):"
	@echo "  train-baselines"
	@echo "  explain-visual-baselines explain-fidelity-baselines explain-errors-baselines"
	@echo ""
	@echo "Local - pipeline principal (runs en $(MAIN_OUTPUT_DIR), var MAIN_MODELS):"
	@echo "  train-main (alias: train)   [EXPORT_FORMATS=onnx,tflite para exportar al terminar]"
	@echo "  tune-main (Optuna HPO)      [N_TRIALS=20 MAIN_EPOCHS=30 PRUNER=median]"
	@echo "  tune-dashboard              (inicia optuna-dashboard en el puerto 8080)"
	@echo "  evaluate-ensemble (alias: ensemble)  (evalúa Soft Voting Ensemble en test.csv)"
	@echo "  export-main       (EXPORT_FORMATS=onnx,tflite [QUANTIZE=int8])"
	@echo "  eval-export-main  (mide el .onnx/.tflite sobre el split de test completo)"
	@echo "  explain-visual-main explain-fidelity-main explain-errors-main"
	@echo "  explain-compare-main explain-global-main   (SHAP: solo pipeline principal)"
	@echo ""
	@echo "Modal - infraestructura:"
	@echo "  modal-seed modal-splits modal-clean-outputs modal-pull"
	@echo "    modal-seed FORCE=1 vacía el Volume y re-descarga (para actualizar el dataset)"
	@echo ""
	@echo "Modal - dataset pre-segmentado (corn-clean-segmented):"
	@echo "  modal-segment-dataset modal-splits-segmented"
	@echo "    modal-train SEGMENTED=1 entrena sobre el dataset segmentado (requiere ambos antes)"
	@echo ""
	@echo "Modal - baselines (runs en /outputs/baselines, var MODELS):"
	@echo "  modal-train-baselines"
	@echo "  modal-explain-visual-baselines modal-explain-fidelity-baselines modal-explain-errors-baselines"
	@echo ""
	@echo "Modal - pipeline principal (runs en /outputs/main, var MAIN_MODELS):"
	@echo "  modal-train-main (alias: modal-train)"
	@echo "  modal-export-main       (EXPORT_FORMATS=onnx,tflite [QUANTIZE=int8])"
	@echo "  modal-eval-export-main  (mide el exportado sobre el split de test completo)"
	@echo "  modal-explain-visual-main modal-explain-fidelity-main modal-explain-errors-main"
	@echo "  modal-explain-compare-main modal-explain-global-main"
	@echo ""
	@echo "Procedencia y fuga (docs/es/provenance):"
	@echo "  provenance-audit  (local, sin GPU)"
	@echo "  modal-provenance-leak  modal-loso  modal-aligned-loso  modal-pull-provenance"
	@echo "    DETACH=1 deja la corrida viva en Modal aunque se cierre la terminal"
	@echo ""
	@echo "Otros: inference lint lint-fix fmt check docs-eda compile-pdf clean-outputs"
	@echo ""
	@echo "Los targets sin sufijo (explain-visual, modal-explain-fidelity, ...) son los genericos:"
	@echo "apuntan a baselines salvo que se pase OUTPUT_DIR (local) o PIPELINE=main (Modal)."

# ==============================================================================
# Local - setup y datos
# ==============================================================================

.PHONY: install download-dataset upload-dataset splits splits-baseline summary test-loader

install:
	$(PIP) install -e ".[dev,analysis,xai,cloud]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

# STAGE_DIR es obligatorio: el empaquetado necesita ~18 GB libres y el volumen del proyecto
# no siempre los tiene. DRY_RUN=1 reporta el plan de shards sin escribir ni subir.
upload-dataset:
	$(PYTHON) scripts/dataset/upload_to_hf.py --stage-dir $(STAGE_DIR) $(if $(DRY_RUN),--dry-run,) $(if $(KEEP_STAGE),--keep-stage,)

splits:
	$(PYTHON) scripts/pipeline/create_splits.py

splits-baseline:
	$(PYTHON) scripts/pipeline/create_splits.py --baseline $(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),)

summary:
	$(PYTHON) src/analysis/dataset_summary.py

test-loader:
	$(PYTHON) scripts/checks/smoke_loader.py

# ==============================================================================
# Local - entrenamiento
# ==============================================================================

.PHONY: train-baselines train train-main

# Baselines: comparación rápida de arquitecturas. Runs en outputs/baselines/<modelo>/.
train-baselines:
	$(PYTHON) scripts/pipeline/train_baselines.py --models $(MODELS) --baseline \
		$(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),) \
		$(if $(REGEN_SPLITS),--regenerate-splits,) \
		--epochs $(EPOCHS) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(IMAGE_SIZE),--image-size $(IMAGE_SIZE),) \
		$(if $(LEARNING_RATE),--learning-rate $(LEARNING_RATE),) \
		$(if $(WEIGHT_DECAY),--weight-decay $(WEIGHT_DECAY),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),) \
		$(if $(NO_PRETRAINED),--no-pretrained,) \
		$(if $(LIME),--lime,)

# Pipeline principal. Runs en outputs/main/<modelo>/.
train:
	$(PYTHON) scripts/pipeline/train.py --models $(MAIN_MODELS) \
		$(if $(MAIN_EPOCHS),--epochs $(MAIN_EPOCHS),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(LEARNING_RATE),--learning-rate $(LEARNING_RATE),) \
		$(if $(WEIGHT_DECAY),--weight-decay $(WEIGHT_DECAY),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),) \
		$(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),) \
		$(if $(CLASS_WEIGHTS),--class-weights $(CLASS_WEIGHTS),) \
		$(if $(CLAHE),--clahe,) \
		$(if $(NO_PRETRAINED),--no-pretrained,) \
		$(if $(EXPORT_FORMATS),--export $(EXPORT_FORMATS),)

train-main: train

# Optuna HPO - Optimización Bayesiana de Hiperparámetros (Criterio 1 Etapa 2)
.PHONY: tune tune-main tune-dashboard
tune:
	$(PYTHON) scripts/pipeline/tune.py --models $(MAIN_MODELS) \
		$(if $(N_TRIALS),--n-trials $(N_TRIALS),) \
		$(if $(MAIN_EPOCHS),--epochs $(MAIN_EPOCHS),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(TIMEOUT),--timeout $(TIMEOUT),) \
		$(if $(PRUNER),--pruner $(PRUNER),) \
		$(if $(BASELINE_F1),--baseline-f1 $(BASELINE_F1),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),) \
		$(if $(NO_PRETRAINED),--no-pretrained,)

tune-main: tune

tune-dashboard:
	$(PYTHON) -c "from optuna_dashboard._cli import main; main()" sqlite:///outputs/tuning/optuna_study.db --port 8080

# Ensamble por Soft Voting (Criterio 2 Etapa 2)
.PHONY: evaluate-ensemble ensemble
evaluate-ensemble:
	$(PYTHON) scripts/pipeline/evaluate_ensemble.py \
		$(if $(MODELS),--models $(MODELS),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),)

ensemble: evaluate-ensemble

# Validación Cruzada K-Fold (Criterio 3 Etapa 2)
.PHONY: cross-validate kfold kfold-main
cross-validate:
	$(PYTHON) scripts/pipeline/cross_validate.py \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(K_FOLDS),--k-folds $(K_FOLDS),) \
		$(if $(MAIN_EPOCHS),--epochs $(MAIN_EPOCHS),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

kfold-main: cross-validate
kfold: cross-validate

# Auditoría de Equidad y Sesgos (Criterio 4 Etapa 2)
.PHONY: fairness-report fairness
fairness-report:
	$(PYTHON) scripts/pipeline/evaluate_fairness.py \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		--run-gradcam --run-shortcut-test

fairness: fairness-report

# ==============================================================================
# Local - exportacion (ONNX/TFLite)
# ==============================================================================
# Convierte un checkpoint ya entrenado del pipeline principal. Espeja el flag
# --export de train.py; usa EXPORT_FORMATS=onnx[,tflite] (default: onnx).

.PHONY: export-main eval-export-main

# Exporta los modelos de MAIN_MODELS. Para uno solo: MAIN_MODELS=<nombre>.
# (No usa MODEL: esa variable tiene default propio para inference_report.)
export-main:
	$(PYTHON) scripts/pipeline/export.py \
		--models $(MAIN_MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		--output-dir $(MAIN_OUTPUT_DIR) \
		--formats $(if $(EXPORT_FORMATS),$(EXPORT_FORMATS),onnx) \
		$(if $(QUANTIZE),--quantize $(QUANTIZE),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(TOLERANCE),--tolerance $(TOLERANCE),) \
		$(if $(MIN_AGREEMENT_RATE),--min-agreement-rate $(MIN_AGREEMENT_RATE),) \
		$(if $(PARITY_SAMPLE_SIZE),--parity-sample-size $(PARITY_SAMPLE_SIZE),) \
		$(if $(NO_PARITY),--no-parity,)

# Calcula ood_stats.json (centroides + covarianza + umbral Mahalanobis) para
# MAIN_MODELS, a partir del mismo checkpoint que export-main. Requiere haber
# exportado antes con FORMATS que incluya el modelo de dos salidas (export-main
# ya envuelve el checkpoint en FeatureExposedModel).
compute-ood-stats:
	$(PYTHON) scripts/pipeline/compute_ood_stats.py \
		--models $(MAIN_MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		--output-dir $(MAIN_OUTPUT_DIR) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(PERCENTILE),--percentile $(PERCENTILE),)

# Mide el archivo exportado sobre el split de test completo (accuracy/macro-F1 reales),
# no solo la paridad numerica de una muestra que valida export-main.
eval-export-main:
	$(PYTHON) scripts/pipeline/evaluate_export.py \
		--models $(MAIN_MODELS) \
		$(if $(RUN),--run $(RUN),) \
		--output-dir $(MAIN_OUTPUT_DIR) \
		--formats $(if $(EXPORT_FORMATS),$(EXPORT_FORMATS),onnx) \
		$(if $(QUANTIZE),--quantize $(QUANTIZE),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(NO_TORCH_BASELINE),--no-torch-baseline,) \
		$(if $(MAX_MACRO_F1_DROP),--max-macro-f1-drop $(MAX_MACRO_F1_DROP),)

.PHONY: sync-mobile-model
# Copia el modelo exportado + labels.json a maize-doctor-app/assets/model, con
# verificacion de hash. RUN_DIR y DEST son obligatorios.
sync-mobile-model:
	$(PYTHON) -m scripts.pipeline.sync_mobile_model \
		--run-dir $(RUN_DIR) \
		--dest $(DEST) \
		$(if $(FORMAT),--format $(FORMAT),) \
		$(if $(QUANTIZE),--quantize $(QUANTIZE),)

# ==============================================================================
# Local - explicabilidad (post-hoc)
# ==============================================================================
# Los targets genéricos aceptan MODELS y OUTPUT_DIR; sin OUTPUT_DIR apuntan a
# outputs/baselines. Las variantes -baselines / -main fijan modelos y directorio.

.PHONY: explain-visual explain-fidelity explain-errors explain-compare explain-global \
	explain-visual-baselines explain-fidelity-baselines explain-errors-baselines \
	explain-visual-main explain-fidelity-main explain-errors-main \
	explain-compare-main explain-global-main

explain-visual:
	$(PYTHON) scripts/pipeline/explain.py visual --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(IMAGE),--image $(IMAGE),) \
		$(if $(OUTPUT),--output $(OUTPUT),) $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-fidelity:
	$(PYTHON) scripts/pipeline/explain.py fidelity --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-errors:
	$(PYTHON) scripts/pipeline/explain.py errors --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-compare:
	$(PYTHON) scripts/pipeline/explain.py compare --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-global:
	$(PYTHON) scripts/pipeline/explain.py global --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

# Runs de baselines (outputs/baselines).
explain-visual-baselines:
	$(MAKE) explain-visual MODELS="$(MODELS)"

explain-fidelity-baselines:
	$(MAKE) explain-fidelity MODELS="$(MODELS)"

explain-errors-baselines:
	$(MAKE) explain-errors MODELS="$(MODELS)"

# Runs del pipeline principal (outputs/main). SHAP (compare/global) es exclusivo de aqui.
explain-visual-main:
	$(MAKE) explain-visual MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-fidelity-main:
	$(MAKE) explain-fidelity MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-errors-main:
	$(MAKE) explain-errors MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-compare-main:
	$(MAKE) explain-compare MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-global-main:
	$(MAKE) explain-global MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

# ==============================================================================
# Local - inferencia puntual
# ==============================================================================
# Inferencia + interpretabilidad completa de una imagen.
# Uso: make inference IMAGE=foto.jpg [MODEL=<nombre> RUN=<run_id> CHECKPOINT=<ruta.pth>
#      STABILITY_RUNS=<n> TOP_K=<k>]

.PHONY: predict inference

# Predicción rápida en imagen o carpeta (soporta MODEL=ensemble, efficientnet_b0, shufflenet_v2_x1_0).
# Uso: make predict IMAGE=experiments/clahe/input [MODEL=ensemble TOP_K=4]
predict:
	$(PYTHON) scripts/pipeline/predict.py \
		$(if $(MODEL),--model $(MODEL),--model ensemble) \
		--image $(IMAGE) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(TOP_K),--top-k $(TOP_K),)

inference:
	$(PYTHON) scripts/pipeline/inference_report.py --model $(MODEL) --image $(IMAGE) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(RUN),--run $(RUN),) \
		$(if $(STABILITY_RUNS),--stability-runs $(STABILITY_RUNS),) \
		$(if $(TOP_K),--top-k $(TOP_K),)

# ==============================================================================
# Modal - infraestructura (Volumes corn-clean / corn-outputs)
# ==============================================================================

.PHONY: modal-seed modal-splits modal-clean-outputs modal-pull

# FORCE=1 vacía el Volume corn-clean antes de descargar: necesario para actualizar el dataset,
# ya que la extracción de shards no elimina archivos renombrados o borrados aguas arriba.
modal-seed:
	$(MODAL) run scripts/modal/train.py::seed_dataset $(if $(FORCE),--force,)

modal-splits:
	$(MODAL) run scripts/modal/train.py::make_splits \
		$(if $(BASELINE),--baseline,) \
		$(if $(NO_CAP),--no-cap,) \
		$(if $(MAX_PER_CLASS),--max-per-class "$(MAX_PER_CLASS)",)

# Splits sobre el dataset pre-segmentado (corn-clean-segmented -> splits/seed_42_segmented).
# Requiere haber corrido antes modal-segment-dataset.
modal-splits-segmented:
	$(MODAL) run scripts/modal/train.py::make_splits_segmented

modal-clean-outputs:
	$(MODAL) run scripts/modal/train.py::clean_outputs

modal-pull:
	$(MODAL) volume get --force corn-outputs / ./outputs-remote

.PHONY: modal-segment-dataset modal-pull-segmentation-previews

# Pre-segmentación del dataset en GPU Modal (detached con DETACH=1).
# Uso: make modal-segment-dataset [DETACH=1 PROFILE=crop_mask_letterbox MAX_IMAGES=1000 MAX_PREVIEWS=50]
modal-segment-dataset:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/segment_dataset.py \
		$(if $(PROFILE),--profile "$(PROFILE)",) \
		$(if $(MAX_IMAGES),--max-images "$(MAX_IMAGES)",) \
		$(if $(MAX_PREVIEWS),--max-previews "$(MAX_PREVIEWS)",)

modal-pull-segmentation-previews:
	$(MODAL) volume get --force corn-outputs segmentation_previews ./outputs/segmentation_previews

# ==============================================================================
# Procedencia y fuga (docs/es/provenance)
# ==============================================================================

.PHONY: provenance-audit modal-provenance-leak modal-loso modal-aligned-loso modal-pull-provenance

# Auditoria local: fuente por imagen, duplicados exactos y barrido de casi-duplicados.
# No requiere GPU ni Modal.
provenance-audit:
	$(PYTHON) scripts/experiments/provenance_audit.py

# Test de fuga de procedencia sobre la particion aleatoria (detached con DETACH=1).
# Uso: make modal-provenance-leak [DETACH=1 ARMS=original,border_ring SEEDS=0,1,2]
modal-provenance-leak:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/provenance_leak.py \
		$(if $(ARMS),--arms "$(ARMS)",) \
		$(if $(SEEDS),--seeds "$(SEEDS)",) \
		$(if $(TRAIN_CAP),--train-cap "$(TRAIN_CAP)",) \
		$(if $(RING_FRACTION),--ring-fraction "$(RING_FRACTION)",)

# Validacion dejando una fuente fuera (detached con DETACH=1).
# Uso: make modal-loso [DETACH=1 SEED=1 ARM=border_ring BALANCE=1 BACKMIX=0.5]
modal-loso:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/leave_one_source_out.py \
		$(if $(SEED),--seed "$(SEED)",) \
		$(if $(ARM),--arm "$(ARM)",) \
		$(if $(FOLDS),--folds "$(FOLDS)",) \
		$(if $(BALANCE),--balance-groups,) \
		$(if $(BACKMIX),--backmix "$(BACKMIX)",) \
		$(if $(TRAIN_CAP),--train-cap "$(TRAIN_CAP)",)

# Validacion por fuente con los componentes del pipeline principal (detached con DETACH=1).
# Uso: make modal-aligned-loso [DETACH=1 ARM=colour_strong SEED=1 TRAIN_CAP=1000]
modal-aligned-loso:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/pipeline_aligned_loso.py 		$(if $(ARM),--arm "$(ARM)",) 		$(if $(SEED),--seed "$(SEED)",) 		$(if $(FOLDS),--folds "$(FOLDS)",) 		$(if $(TRAIN_CAP),--train-cap "$(TRAIN_CAP)",)

modal-pull-provenance:
	$(MODAL) volume get --force corn-outputs experiments ./outputs/experiments

# ==============================================================================
# Modal - entrenamiento
# ==============================================================================

.PHONY: modal-train-baselines modal-train modal-train-main

# Baselines en GPU. Runs en /outputs/baselines/<modelo>/.
modal-train-baselines:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/train.py --models "$(MODELS)" --epochs "$(EPOCHS)" \
		$(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class "$(MAX_PER_CLASS)",) \
		$(if $(REGEN_SPLITS),--regenerate-splits,) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(IMAGE_SIZE),--image-size "$(IMAGE_SIZE)",) \
		$(if $(LEARNING_RATE),--learning-rate "$(LEARNING_RATE)",) \
		$(if $(WEIGHT_DECAY),--weight-decay "$(WEIGHT_DECAY)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(NO_PRETRAINED),--no-pretrained,) \
		$(if $(LIME),--lime,)

# Pipeline principal en GPU. Runs en /outputs/main/<modelo>/.
# SEGMENTED=1 entrena sobre corn-clean-segmented + splits/seed_42_segmented (requiere
# modal-segment-dataset y modal-splits-segmented corridos antes).
modal-train:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/train.py::train_main --models "$(MAIN_MODELS)" \
		$(if $(MAIN_EPOCHS),--epochs "$(MAIN_EPOCHS)",) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(LEARNING_RATE),--learning-rate "$(LEARNING_RATE)",) \
		$(if $(CLASS_WEIGHTS),--class-weights "$(CLASS_WEIGHTS)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(CLAHE),--clahe,) \
		$(if $(NO_PRETRAINED),--no-pretrained,) \
		$(if $(SEGMENTED),--segmented,) \
		$(if $(SPLITS_DIR),--splits-dir "$(SPLITS_DIR)",)

modal-train-main: modal-train

.PHONY: modal-tune modal-tune-dashboard
modal-tune:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/train.py::tune_main \
		$(if $(MODEL),--models "$(MODEL)",--models "$(MODELS)") \
		$(if $(N_TRIALS),--n-trials "$(N_TRIALS)",) \
		$(if $(EPOCHS),--epochs "$(EPOCHS)",) \
		$(if $(TIMEOUT),--timeout "$(TIMEOUT)",) \
		$(if $(PRUNER),--pruner "$(PRUNER)",) \
		$(if $(BASELINE_F1),--baseline-macro-f1 "$(BASELINE_F1)",) \
		$(if $(SPLITS_DIR),--splits-dir "$(SPLITS_DIR)",)

modal-tune-dashboard:
	$(MODAL) serve scripts/modal/train.py

# Ensamble, Validación Cruzada y Fairness en GPU Modal (Etapa 2)
.PHONY: modal-ensemble modal-evaluate-ensemble modal-cross-validate modal-kfold modal-fairness modal-fairness-report

modal-ensemble:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/train.py::evaluate_ensemble_modal \
		$(if $(MODELS),--models "$(MODELS)",--models "$(MAIN_MODELS)") \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(SPLITS_DIR),--splits-dir "$(SPLITS_DIR)",) \
		$(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

modal-evaluate-ensemble: modal-ensemble

modal-cross-validate:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/train.py::cross_validate_modal \
		$(if $(MODEL),--model "$(MODEL)",) \
		$(if $(K_FOLDS),--k-folds "$(K_FOLDS)",) \
		$(if $(MAIN_EPOCHS),--epochs "$(MAIN_EPOCHS)",$(if $(EPOCHS),--epochs "$(EPOCHS)",)) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(LEARNING_RATE),--learning-rate "$(LEARNING_RATE)",) \
		$(if $(WEIGHT_DECAY),--weight-decay "$(WEIGHT_DECAY)",) \
		$(if $(CLASS_WEIGHTS),--class-weights "$(CLASS_WEIGHTS)",) \
		$(if $(BEST_PARAMS),--best-params "$(BEST_PARAMS)",) \
		$(if $(SPLITS_DIR),--splits-dir "$(SPLITS_DIR)",) \
		$(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

modal-kfold: modal-cross-validate

modal-fairness:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/train.py::fairness_report_modal \
		$(if $(MODEL),--model "$(MODEL)",) \
		$(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(SPLITS_DIR),--splits-dir "$(SPLITS_DIR)",) \
		$(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

modal-fairness-report: modal-fairness

# ==============================================================================
# Modal - exportacion (ONNX/TFLite)
# ==============================================================================

.PHONY: modal-export-main modal-eval-export-main modal-compute-ood-stats

# SEGMENTED=1 en los tres targets de esta seccion: usa DATASET_ROOT=/data_segmented para
# que la muestra de paridad / eval / features de OOD lean las mismas imagenes que vio el
# checkpoint (necesario si el run se entreno con SEGMENTED=1 via modal-train).
modal-export-main:
	$(MODAL) run scripts/modal/export.py::export_main --models "$(MAIN_MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		--formats "$(if $(EXPORT_FORMATS),$(EXPORT_FORMATS),onnx)" \
		$(if $(QUANTIZE),--quantize "$(QUANTIZE)",) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(TOLERANCE),--tolerance $(TOLERANCE),) \
		$(if $(MIN_AGREEMENT_RATE),--min-agreement-rate $(MIN_AGREEMENT_RATE),) \
		$(if $(PARITY_SAMPLE_SIZE),--parity-sample-size $(PARITY_SAMPLE_SIZE),) \
		$(if $(NO_PARITY),--no-parity,) \
		$(if $(SEGMENTED),--segmented,)

modal-eval-export-main:
	$(MODAL) run scripts/modal/export.py::evaluate_export_main --models "$(MAIN_MODELS)" \
		$(if $(RUN),--run $(RUN),) \
		--formats "$(if $(EXPORT_FORMATS),$(EXPORT_FORMATS),onnx)" \
		$(if $(QUANTIZE),--quantize "$(QUANTIZE)",) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(NO_TORCH_BASELINE),--no-torch-baseline,) \
		$(if $(MAX_MACRO_F1_DROP),--max-macro-f1-drop $(MAX_MACRO_F1_DROP),) \
		$(if $(SEGMENTED),--segmented,)

modal-compute-ood-stats:
	$(MODAL) run scripts/modal/export.py::compute_ood_stats --models "$(MAIN_MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(PERCENTILE),--percentile $(PERCENTILE),) \
		$(if $(SEGMENTED),--segmented,)

# ==============================================================================
# Modal - explicabilidad (post-hoc)
# ==============================================================================
# Mismos genéricos que en local, pero el directorio de runs se elige con
# PIPELINE=baselines|main. Las variantes -baselines / -main fijan modelos y pipeline.

.PHONY: modal-explain-visual modal-explain-fidelity modal-explain-errors \
	modal-explain-compare modal-explain-global \
	modal-explain-visual-baselines modal-explain-fidelity-baselines modal-explain-errors-baselines \
	modal-explain-visual-main modal-explain-fidelity-main modal-explain-errors-main \
	modal-explain-compare-main modal-explain-global-main

modal-explain-visual:
	$(MODAL) run scripts/modal/explain.py::explain_visual --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

modal-explain-fidelity:
	$(MODAL) run scripts/modal/explain.py::explain_fidelity --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-explain-errors:
	$(MODAL) run scripts/modal/explain.py::explain_errors --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-explain-compare:
	$(MODAL) run scripts/modal/explain.py::explain_compare --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),) $(if $(SEGMENTED),--segmented,)

modal-explain-global:
	$(MODAL) run scripts/modal/explain.py::explain_global --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),) $(if $(SEGMENTED),--segmented,)

modal-explain-visual-baselines:
	$(MAKE) modal-explain-visual MODELS="$(MODELS)" PIPELINE=baselines

modal-explain-fidelity-baselines:
	$(MAKE) modal-explain-fidelity MODELS="$(MODELS)" PIPELINE=baselines

modal-explain-errors-baselines:
	$(MAKE) modal-explain-errors MODELS="$(MODELS)" PIPELINE=baselines

modal-explain-visual-main:
	$(MAKE) modal-explain-visual MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-fidelity-main:
	$(MAKE) modal-explain-fidelity MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-errors-main:
	$(MAKE) modal-explain-errors MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-compare-main:
	$(MAKE) modal-explain-compare MODELS="$(MAIN_MODELS)" PIPELINE=main SEGMENTED="$(SEGMENTED)"

modal-explain-global-main:
	$(MAKE) modal-explain-global MODELS="$(MAIN_MODELS)" PIPELINE=main SEGMENTED="$(SEGMENTED)"

# ==============================================================================
# Calidad, documentación y limpieza
# ==============================================================================

.PHONY: lint lint-fix fmt check docs-eda compile-pdf clean-outputs

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/

# Requiere shell POSIX (Powershell/Git Bash/WSL en Windows) y que existan outputs/eda/eda_*.png
docs-eda:
	cp outputs/eda/eda_*.png public/eda/

.PHONY: docs docs-dev
docs-dev:
	npm run docs:dev

docs: docs-dev

compile-pdf:
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex

clean-outputs:
	rm -rf outputs/
