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
MAIN_MODELS ?= shufflenet_v2_x1_0

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
	@echo "  install download-dataset splits splits-baseline summary test-loader"
	@echo ""
	@echo "Local - baselines (runs en outputs/baselines, var MODELS):"
	@echo "  train-baselines"
	@echo "  explain-visual-baselines explain-fidelity-baselines explain-errors-baselines"
	@echo ""
	@echo "Local - pipeline principal (runs en $(MAIN_OUTPUT_DIR), var MAIN_MODELS):"
	@echo "  train-main (alias: train)"
	@echo "  explain-visual-main explain-fidelity-main explain-errors-main"
	@echo "  explain-compare-main explain-global-main   (SHAP: solo pipeline principal)"
	@echo ""
	@echo "Modal - infraestructura:"
	@echo "  modal-seed modal-splits modal-clean-outputs modal-pull"
	@echo ""
	@echo "Modal - baselines (runs en /outputs/baselines, var MODELS):"
	@echo "  modal-train-baselines"
	@echo "  modal-explain-visual-baselines modal-explain-fidelity-baselines modal-explain-errors-baselines"
	@echo ""
	@echo "Modal - pipeline principal (runs en /outputs/main, var MAIN_MODELS):"
	@echo "  modal-train-main (alias: modal-train)"
	@echo "  modal-explain-visual-main modal-explain-fidelity-main modal-explain-errors-main"
	@echo "  modal-explain-compare-main modal-explain-global-main"
	@echo ""
	@echo "Otros: inference lint lint-fix fmt check docs-eda compile-pdf clean-outputs"
	@echo ""
	@echo "Los targets sin sufijo (explain-visual, modal-explain-fidelity, ...) son los genericos:"
	@echo "apuntan a baselines salvo que se pase OUTPUT_DIR (local) o PIPELINE=main (Modal)."

# ==============================================================================
# Local - setup y datos
# ==============================================================================

.PHONY: install download-dataset splits splits-baseline summary test-loader

install:
	$(PIP) install -e ".[dev,analysis,xai,cloud]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

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
		$(if $(CLASS_WEIGHTS),--class-weights $(CLASS_WEIGHTS),) \
		$(if $(CLAHE),--clahe,) \
		$(if $(NO_PRETRAINED),--no-pretrained,)

train-main: train

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

.PHONY: inference

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

modal-seed:
	$(MODAL) run scripts/modal/train.py::seed_dataset

modal-splits:
	$(MODAL) run scripts/modal/train.py::make_splits \
		$(if $(BASELINE),--baseline,) \
		$(if $(NO_CAP),--no-cap,) \
		$(if $(MAX_PER_CLASS),--max-per-class "$(MAX_PER_CLASS)",)

modal-clean-outputs:
	$(MODAL) run scripts/modal/train.py::clean_outputs

modal-pull:
	$(MODAL) volume get --force corn-outputs / ./outputs-remote

# ==============================================================================
# Modal - entrenamiento
# ==============================================================================

.PHONY: modal-train-baselines modal-train modal-train-main

# Baselines en GPU. Runs en /outputs/baselines/<modelo>/.
modal-train-baselines:
	$(MODAL) run scripts/modal/train.py --models "$(MODELS)" --epochs "$(EPOCHS)" \
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
modal-train:
	$(MODAL) run scripts/modal/train.py::train_main --models "$(MAIN_MODELS)" \
		$(if $(MAIN_EPOCHS),--epochs "$(MAIN_EPOCHS)",) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(LEARNING_RATE),--learning-rate "$(LEARNING_RATE)",) \
		$(if $(CLASS_WEIGHTS),--class-weights "$(CLASS_WEIGHTS)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(CLAHE),--clahe,) \
		$(if $(NO_PRETRAINED),--no-pretrained,)

modal-train-main: modal-train

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
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),)

modal-explain-global:
	$(MODAL) run scripts/modal/explain.py::explain_global --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),)

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
	$(MAKE) modal-explain-compare MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-global-main:
	$(MAKE) modal-explain-global MODELS="$(MAIN_MODELS)" PIPELINE=main

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

compile-pdf:
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex

clean-outputs:
	rm -rf outputs/
