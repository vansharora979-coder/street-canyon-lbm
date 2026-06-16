# Makefile for the 2D LBM street-canyon ventilation study.
# All targets use a local virtual environment in .venv/.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest

.DEFAULT_GOAL := help
.PHONY: help setup test lint run figures reproduce clean freeze

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(PY): ## Create the virtual environment.
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(PY) ## Create venv and install pinned dependencies + package (editable).
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Environment ready. Activate with: source $(VENV)/bin/activate"

test: ## Run the test suite.
	$(PYTEST)

lint: ## Lint with ruff if available (non-fatal if not installed).
	@$(PY) -m ruff check src scripts tests 2>/dev/null || \
		echo "ruff not installed (optional): pip install ruff"

run: ## Run the Phase 1 Poiseuille validation case.
	$(PY) scripts/run_case.py --poiseuille

figures: ## Regenerate all currently-available figures.
	$(PY) scripts/make_figures.py

reproduce: setup test figures ## Full reproduction: env -> tests -> figures.
	@echo "Reproduction complete. See figures/ and results/."

freeze: ## Snapshot the exact installed versions into requirements.txt.
	$(PIP) freeze --exclude-editable > requirements.txt
	@echo "requirements.txt updated."

clean: ## Remove caches and generated artefacts (keeps the venv).
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__
	rm -f results/*.json results/*.meta.json
	rm -f figures/*.png figures/*.svg
