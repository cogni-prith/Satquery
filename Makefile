.PHONY: install lint test registry smoke train-clip train-seg train-change eval

install:
	uv sync --extra gpu || pip install -e ".[gpu]"

lint:
	ruff check src tests scripts && ruff format --check src tests scripts

test:
	pytest tests -q

registry:
	@PYTHONPATH=src python -m satquery.models.registry --json

smoke:
	PYTHONPATH=src python scripts/smoke_test.py

train-clip:
	PYTHONPATH=src python scripts/train_clip.py --config configs/train/contrastive.yaml

train-seg:
	PYTHONPATH=src python scripts/train_seg.py --config configs/train/segmentation.yaml

train-change:
	PYTHONPATH=src python scripts/train_change.py --config configs/train/change.yaml

eval:
	PYTHONPATH=src python scripts/run_eval.py --config configs/eval/full_suite.yaml
