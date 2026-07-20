# Metalcore AI — convenience targets for the CPU authoring/verification path.
# GPU stages (2/3/4 training & inference) run on Kaggle via the notebooks.

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev smoke dsp compile lint check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install Stage 1 + assembly (CPU) dependencies
	$(PY) -m pip install -r requirements-dataset.txt -r requirements-assembly.txt

install-dev:  ## Install the package (editable) + ruff for linting
	$(PY) -m pip install -e . ruff

compile:  ## Byte-compile all packages (fast syntax check, no GPU imports)
	$(PY) -m compileall -q metalcore dataset_tools music_training \
		lyric_training rvc_training inference scripts

smoke:  ## Stage 1 end-to-end smoke test (synthesises audio, no GPU/data)
	bash scripts/smoke_test.sh

dsp:  ## Self-contained DSP smoke test for Stages 4 & 5
	PYTHONPATH=. $(PY) scripts/smoke_test_dsp.py

lint:  ## Lint with ruff (install with `make install-dev`)
	ruff check metalcore dataset_tools music_training lyric_training \
		rvc_training inference scripts

check: compile smoke dsp  ## Run the full CPU verification suite (what CI runs)

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info .pytest_cache
