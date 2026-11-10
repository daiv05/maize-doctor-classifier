ifeq ($(OS),Windows_NT)
    PYTHON := venv\Scripts\python
    PIP    := venv\Scripts\pip
    RUFF   := venv\Scripts\ruff
    PYRIGHT := venv\Scripts\pyright
    MODAL   := venv\Scripts\modal
else
    PYTHON := venv/bin/python
    PIP    := venv/bin/pip
    RUFF   := venv/bin/ruff
    PYRIGHT := venv/bin/pyright
    MODAL   := venv/bin/modal
endif

MODELS ?= efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0
MAIN_MODELS ?= efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0

OUTPUT_DIR ?=
MAIN_OUTPUT_DIR ?= outputs/main

PIPELINE ?= baselines

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

N_TRIALS ?= 20
TIMEOUT ?=
PRUNER ?= median
BASELINE_F1 ?=
K_FOLDS ?= 5

EXPORT_FORMATS ?=
QUANTIZE ?=
NO_PARITY ?=
TOLERANCE ?=
MIN_AGREEMENT_RATE ?=
PARITY_SAMPLE_SIZE ?=
NO_TORCH_BASELINE ?=
MAX_MACRO_F1_DROP ?=

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

.PHONY: help
help:
	@echo "Local - setup y datos:"
	@echo "  install download-dataset upload-dataset splits splits-baseline summary test-loader test"
	@echo "    install incluye los extras dev, analysis, xai, cloud y tuning"
	@echo "    upload-dataset requiere STAGE_DIR=<dir con ~18GB libres> [DRY_RUN=1] [KEEP_STAGE=1]"
	@echo ""
	@echo "Local - baselines (runs en outputs/baselines, var MODELS):"
	@echo "  train-baselines"
	@echo "  explain-visual-baselines explain-fidelity-baselines explain-errors-baselines"
	@echo ""
	@echo "Local - pipeline principal (runs en $(MAIN_OUTPUT_DIR), var MAIN_MODELS):"
	@echo "  train-main (alias: train)   [EXPORT_FORMATS=onnx,tflite para exportar al terminar]"
	@echo "  tune-main (alias: tune)     Optuna HPO [N_TRIALS=20 MAIN_EPOCHS=30 PRUNER=median BASELINE_F1=<valor>]"
	@echo "    sin BASELINE_F1 se usa el baseline de validacion del modelo oficial (tuning_study.BASELINE_F1)"
	@echo "  hpo-main                    protocolo HPO formal Lite0 seed 42"
	@echo "  tune-dashboard              optuna-dashboard en el puerto 8080 (requiere el extra tuning)"
	@echo "  evaluate-ensemble (alias: ensemble)  Soft Voting Ensemble sobre test.csv"
	@echo "  cross-validate (alias: kfold, kfold-main)  [K_FOLDS=5]"
	@echo "  fairness-report (alias: fairness)"
	@echo "  predict           (IMAGE=<archivo|carpeta> [MODEL=ensemble TOP_K=4])"
	@echo "  inference         (IMAGE=<archivo> [MODEL RUN CHECKPOINT STABILITY_RUNS TOP_K])"
	@echo "  export-main       (EXPORT_FORMATS=onnx,tflite [QUANTIZE=int8])"
	@echo "  compute-ood-stats (ood_stats.json del modelo exportado [PERCENTILE NUM_WORKERS])"
	@echo "  eval-export-main  (mide el .onnx/.tflite sobre el split de test completo [NUM_WORKERS])"
	@echo "  sync-mobile-model (RUN_DIR=<run> DEST=<assets/model de la app> [FORMAT QUANTIZE])"
	@echo "  explain-visual-main explain-fidelity-main explain-errors-main"
	@echo "  explain-compare-main explain-global-main   (SHAP: solo pipeline principal)"
	@echo ""
	@echo "Modal - infraestructura:"
	@echo "  modal-seed modal-splits modal-clean-outputs modal-pull"
	@echo "    modal-seed FORCE=1 vacia el Volume y re-descarga (para actualizar el dataset)"
	@echo ""
	@echo "Modal - dataset pre-segmentado (corn-clean-segmented):"
	@echo "  modal-segment-dataset modal-splits-segmented modal-pull-segmentation-previews"
	@echo "    modal-train SEGMENTED=1 entrena sobre el dataset segmentado (requiere ambos antes)"
	@echo ""
	@echo "Modal - baselines (runs en /outputs/baselines, var MODELS):"
	@echo "  modal-train-baselines"
	@echo "  modal-explain-visual-baselines modal-explain-fidelity-baselines modal-explain-errors-baselines"
	@echo ""
	@echo "Modal - pipeline principal (runs en /outputs/main, var MAIN_MODELS):"
	@echo "  modal-train-main (alias: modal-train)"
	@echo "  modal-tune modal-tune-dashboard"
	@echo "  modal-hpo-preflight modal-hpo-smoke modal-hpo   (protocolo HPO formal)"
	@echo "  modal-ensemble (alias: modal-evaluate-ensemble) modal-cross-validate (alias: modal-kfold)"
	@echo "  modal-fairness (alias: modal-fairness-report)"
	@echo "  modal-export-main       (EXPORT_FORMATS=onnx,tflite [QUANTIZE=int8])"
	@echo "  modal-compute-ood-stats (ood_stats.json en Modal [NUM_WORKERS])"
	@echo "  modal-eval-export-main  (mide el exportado sobre el split de test completo)"
	@echo "  modal-explain-visual-main modal-explain-fidelity-main modal-explain-errors-main"
	@echo "  modal-explain-compare-main modal-explain-global-main"
	@echo ""
	@echo "Procedencia y fuga (docs/es/provenance/evidencia):"
	@echo "  provenance-audit  (local, sin GPU)"
	@echo "  modal-provenance-leak  modal-loso  modal-aligned-loso  modal-pull-provenance"
	@echo "    DETACH=1 deja la corrida viva en Modal aunque se cierre la terminal"
	@echo ""
	@echo "Calidad, documentacion y limpieza:"
	@echo "  lint lint-fix fmt check docs-eda docs (alias: docs-dev) clean-outputs"
	@echo "  compile-tesina          (3 pasadas de pdflatex en reports/tesina)"
	@echo "  compile-pdf-first-phase (documentacion de la primera fase)"
	@echo ""
	@echo "Los targets sin sufijo (explain-visual, modal-explain-fidelity, ...) son los genericos:"
	@echo "apuntan a baselines salvo que se pase OUTPUT_DIR (local) o PIPELINE=main (Modal)."

.PHONY: install download-dataset upload-dataset splits splits-baseline summary test-loader

install:
	$(PIP) install -e ".[dev,analysis,xai,cloud,tuning]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

upload-dataset:
	$(PYTHON) scripts/dataset/upload_to_hf.py --stage-dir $(STAGE_DIR) $(if $(DRY_RUN),--dry-run,) $(if $(KEEP_STAGE),--keep-stage,)

splits:
	$(PYTHON) scripts/pipeline/create_splits.py $(if $(GROUP_BY_SOURCE),--group-by-source,)

splits-baseline:
	$(PYTHON) scripts/pipeline/create_splits.py --baseline $(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),) $(if $(GROUP_BY_SOURCE),--group-by-source,)

summary:
	$(PYTHON) src/analysis/dataset_summary.py

test-loader:
	$(PYTHON) scripts/checks/smoke_loader.py

.PHONY: train-baselines train train-main

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

.PHONY: hpo-main modal-hpo-preflight modal-hpo-smoke modal-hpo
hpo-main:
	$(PYTHON) scripts/pipeline/tune.py --formal-hpo $(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),)

modal-hpo-preflight:
	$(MODAL) run scripts/modal/train.py::tune_main --formal-hpo --preflight-only

modal-hpo-smoke:
	$(MODAL) run scripts/modal/train.py::tune_main --formal-hpo --smoke

modal-hpo:
	$(MODAL) run --detach scripts/modal/train.py::tune_main --formal-hpo --continue-study

tune-dashboard:
	$(PYTHON) -c "from optuna_dashboard._cli import main; main()" sqlite:///outputs/tuning/optuna_study.db --port 8080

.PHONY: evaluate-ensemble ensemble
evaluate-ensemble:
	$(PYTHON) scripts/pipeline/evaluate_ensemble.py \
		$(if $(MODELS),--models $(MODELS),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),)

ensemble: evaluate-ensemble

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

.PHONY: export-main eval-export-main

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

.PHONY: compute-ood-stats
compute-ood-stats:
	$(PYTHON) scripts/pipeline/compute_ood_stats.py \
		--models $(MAIN_MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		--output-dir $(MAIN_OUTPUT_DIR) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(PERCENTILE),--percentile $(PERCENTILE),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),)

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
		$(if $(MAX_MACRO_F1_DROP),--max-macro-f1-drop $(MAX_MACRO_F1_DROP),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),)

.PHONY: sync-mobile-model
sync-mobile-model:
	$(PYTHON) -m scripts.pipeline.sync_mobile_model \
		--run-dir $(RUN_DIR) \
		--dest $(DEST) \
		$(if $(FORMAT),--format $(FORMAT),) \
		$(if $(QUANTIZE),--quantize $(QUANTIZE),)

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

explain-visual-baselines:
	$(MAKE) explain-visual MODELS="$(MODELS)"

explain-fidelity-baselines:
	$(MAKE) explain-fidelity MODELS="$(MODELS)"

explain-errors-baselines:
	$(MAKE) explain-errors MODELS="$(MODELS)"

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

.PHONY: predict inference

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

.PHONY: modal-seed modal-splits modal-splits-segmented modal-clean-outputs modal-pull

modal-seed:
	$(MODAL) run scripts/modal/train.py::seed_dataset $(if $(FORCE),--force,)

modal-splits:
	$(MODAL) run scripts/modal/train.py::make_splits \
		$(if $(BASELINE),--baseline,) \
		$(if $(NO_CAP),--no-cap,) \
		$(if $(MAX_PER_CLASS),--max-per-class "$(MAX_PER_CLASS)",) \
		$(if $(ALLOW_INCOMPLETE),--allow-incomplete,) \
		$(if $(GROUP_BY_SOURCE),--group-by-source,)

modal-splits-segmented:
	$(MODAL) run scripts/modal/train.py::make_splits_segmented

modal-clean-outputs:
	$(MODAL) run scripts/modal/train.py::clean_outputs

modal-pull:
	$(MODAL) volume get --force corn-outputs / ./outputs-remote

.PHONY: modal-segment-dataset modal-pull-segmentation-previews

modal-segment-dataset:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/segment_dataset.py \
		$(if $(PROFILE),--profile "$(PROFILE)",) \
		$(if $(MAX_IMAGES),--max-images "$(MAX_IMAGES)",) \
		$(if $(MAX_PREVIEWS),--max-previews "$(MAX_PREVIEWS)",)

modal-pull-segmentation-previews:
	$(MODAL) volume get --force corn-outputs segmentation_previews ./outputs/segmentation_previews

.PHONY: provenance-audit modal-provenance-leak modal-loso modal-aligned-loso modal-pull-provenance

provenance-audit:
	$(PYTHON) scripts/experiments/provenance_audit.py

modal-provenance-leak:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/provenance_leak.py \
		$(if $(ARMS),--arms "$(ARMS)",) \
		$(if $(SEEDS),--seeds "$(SEEDS)",) \
		$(if $(TRAIN_CAP),--train-cap "$(TRAIN_CAP)",) \
		$(if $(RING_FRACTION),--ring-fraction "$(RING_FRACTION)",)

modal-loso:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/leave_one_source_out.py \
		$(if $(SEED),--seed "$(SEED)",) \
		$(if $(ARM),--arm "$(ARM)",) \
		$(if $(FOLDS),--folds "$(FOLDS)",) \
		$(if $(BALANCE),--balance-groups,) \
		$(if $(BACKMIX),--backmix "$(BACKMIX)",) \
		$(if $(TRAIN_CAP),--train-cap "$(TRAIN_CAP)",)

modal-aligned-loso:
	$(MODAL) run $(if $(DETACH),--detach,) scripts/modal/pipeline_aligned_loso.py \
		$(if $(ARM),--arm "$(ARM)",) \
		$(if $(SEED),--seed "$(SEED)",) \
		$(if $(FOLDS),--folds "$(FOLDS)",) \
		$(if $(TRAIN_CAP),--train-cap "$(TRAIN_CAP)",)

modal-pull-provenance:
	$(MODAL) volume get --force corn-outputs experiments ./outputs/experiments

.PHONY: modal-train-baselines modal-train modal-train-main

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

.PHONY: modal-export-main modal-eval-export-main modal-compute-ood-stats

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
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),) \
		$(if $(SEGMENTED),--segmented,)

modal-compute-ood-stats:
	$(MODAL) run scripts/modal/export.py::compute_ood_stats --models "$(MAIN_MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(SPLITS_DIR),--splits-dir $(SPLITS_DIR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(PERCENTILE),--percentile $(PERCENTILE),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),) \
		$(if $(SEGMENTED),--segmented,)

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

.PHONY: test lint lint-fix fmt check docs-eda compile-pdf-first-phase compile-tesina clean-outputs

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/

docs-eda:
	cp outputs/eda/eda_*.png public/eda/

.PHONY: docs docs-dev
docs-dev:
	npm run docs:dev

docs: docs-dev

compile-pdf-first-phase:
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex

compile-tesina:
	cd reports/tesina && pdflatex -interaction=nonstopmode tesina.tex
	cd reports/tesina && pdflatex -interaction=nonstopmode tesina.tex
	cd reports/tesina && pdflatex -interaction=nonstopmode tesina.tex

clean-outputs:
	rm -rf outputs/
