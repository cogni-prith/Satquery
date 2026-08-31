# Every recipe is a single line. No backslash continuations.
#
# This project runs in the `tf-torch` miniconda environment, which already carries a
# CUDA build of PyTorch. Override with `make PY=/path/to/python <target>` if needed.
CONDA_ENV ?= tf-torch
CONDA_ROOT ?= $(HOME)/miniconda3
PY  ?= $(CONDA_ROOT)/envs/$(CONDA_ENV)/bin/python
RUN := PYTHONPATH=src $(PY)

.DEFAULT_GOAL := help
.PHONY: help env install install-gpu lint format test registry smoke fetch-model infer train-lora train-change eval clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## Show which interpreter and GPU the targets will use
	@$(RUN) -c "import sys,torch;print('python  ',sys.version.split()[0],sys.executable);print('torch   ',torch.__version__);print('cuda    ',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

install: ## Install the CPU dependencies into the conda env. Deliberately does not touch torch.
	$(PY) -m pip install "rasterio>=1.3" "affine>=2.4" "omegaconf>=2.3" "pydantic>=2.7" "pyyaml>=6" "scipy>=1.11" "pillow>=10" "tqdm>=4.66" "pytest>=8" "pytest-cov" "ruff>=0.6"

install-gpu: ## Add the training-only packages. torch is already present, so it is not reinstalled.
	$(PY) -m pip install --no-deps "peft>=0.12" "accelerate>=0.33" "bitsandbytes>=0.43"

lint: ## Check formatting and lint rules
	$(RUN) -m ruff format --check . && $(RUN) -m ruff check .

format: ## Apply formatting and autofixable lint rules
	$(RUN) -m ruff format . && $(RUN) -m ruff check --fix .

test: ## Run the CPU test suite
	$(RUN) -m pytest

registry: ## Dump every registered tool spec as JSON (this is what the backend reads)
	@$(RUN) -m satquery.models.registry --json

smoke: ## End-to-end ingest and preprocessing check on synthetic rasters, no weights needed
	$(RUN) scripts/smoke_test.py

fetch-model: ## Download the EarthDial-4B checkpoint and repair its missing remote code
	$(RUN) scripts/fetch_model.py --config configs/model/earthdial_4b_rgb.yaml

infer: ## Run one real inference through the VLM tools. Needs the checkpoint and a GPU.
	$(RUN) scripts/infer_demo.py --config configs/model/earthdial_4b_rgb.yaml

train-lora: ## LoRA fine-tune the VLM backbone on the data mix
	$(RUN) scripts/train_lora.py --config configs/train/lora_stage1.yaml

train-change: ## Train the discriminative CDVQA change head
	$(RUN) scripts/train_change_head.py --config configs/train/change_head.yaml

eval: ## Run the full evaluation suite and write the scores table
	$(RUN) scripts/run_eval.py --config configs/eval/full_suite.yaml

clean: ## Remove caches and build products. Never touches data/ or artifacts/.
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info && find . -name __pycache__ -not -path './.venv/*' -type d -exec rm -rf {} + 2>/dev/null || true
