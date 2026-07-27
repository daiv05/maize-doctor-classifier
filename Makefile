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

MODELS ?= efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0
MAIN_MODELS ?= shufflenet_v2_x1_0
EPOCHS ?= 30
MAIN_EPOCHS ?=
SPLITS_DIR ?=
CLASS_WEIGHTS ?=
CLAHE ?=
NO_CAP ?=
MAX_PER_CLASS ?=
REGEN_SPLITS ?=
BATCH_SIZE ?=
IMAGE_SIZE ?=
LEARNING_RATE ?=
WEIGHT_DECAY ?=
NUM_WORKERS ?=
NO_PRETRAINED ?=
LIME ?=
NUM_SAMPLES ?=
MODEL ?= efficientnet_b0
IMAGE ?=
CHECKPOINT ?=
RUN ?=
TOP_K ?=
STABILITY_RUNS ?=

.PHONY: compile-pdf install download-dataset splits splits-baseline train train-baselines inference explain-lime explain-report explain-errors test-loader summary docs-eda lint lint-fix fmt check clean-outputs modal-seed modal-train modal-train-baselines modal-clean-outputs modal-explain-lime modal-explain-report modal-explain-errors modal-pull

install:
	$(PIP) install -e ".[dev,analysis,xai,cloud]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

splits:
	$(PYTHON) scripts/pipeline/create_splits.py

splits-baseline:
	$(PYTHON) scripts/pipeline/create_splits.py --baseline $(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),)

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

clean-outputs:
	rm -rf outputs/

modal-seed:
	$(MODAL) run scripts/modal/train.py::seed_dataset

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

modal-train:
	$(MODAL) run scripts/modal/train.py::train_main --models "$(MAIN_MODELS)" \
		$(if $(MAIN_EPOCHS),--epochs "$(MAIN_EPOCHS)",) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(LEARNING_RATE),--learning-rate "$(LEARNING_RATE)",) \
		$(if $(CLASS_WEIGHTS),--class-weights "$(CLASS_WEIGHTS)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(CLAHE),--clahe,) \
		$(if $(NO_PRETRAINED),--no-pretrained,)

modal-clean-outputs:
	$(MODAL) run scripts/modal/train.py::clean_outputs

modal-explain-lime:
	$(MODAL) run scripts/modal/explain.py::explain_lime --models "$(MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

modal-explain-report:
	$(MODAL) run scripts/modal/explain.py::explain_report --models "$(MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-explain-errors:
	$(MODAL) run scripts/modal/explain.py::explain_errors --models "$(MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-pull:
	$(MODAL) volume get --force corn-outputs / ./outputs-remote

# Inferencia + interpretabilidad completa de una imagen puntual.
# Uso: make inference IMAGE=foto.jpg [MODEL=<nombre> RUN=<run_id> CHECKPOINT=<ruta.pth>
#      STABILITY_RUNS=<n> TOP_K=<k>]
inference:
	$(PYTHON) scripts/pipeline/inference_report.py --model $(MODEL) --image $(IMAGE) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(RUN),--run $(RUN),) \
		$(if $(STABILITY_RUNS),--stability-runs $(STABILITY_RUNS),) \
		$(if $(TOP_K),--top-k $(TOP_K),)

explain-lime:
	$(PYTHON) scripts/pipeline/explain_lime.py --models $(MODELS) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

explain-report:
	$(PYTHON) scripts/pipeline/explain_report.py --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

explain-errors:
	$(PYTHON) scripts/pipeline/explain_report.py --models $(MODELS) --errors-only \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

test-loader:
	$(PYTHON) scripts/checks/smoke_loader.py

summary:
	$(PYTHON) src/analysis/dataset_summary.py

# Requiere shell POSIX (Powershell/Git Bash/WSL en Windows) y que existan outputs/eda/eda_*.png
docs-eda:
	cp outputs/eda/eda_*.png public/eda/

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/

compile-pdf:
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
