PY ?= python3
VENV ?= .venv

.PHONY: install test lint demo clean

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -e .

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m compileall -q src tests

demo:
	$(PY) -m asr.pipeline --config config.yaml

clean:
	rm -rf .pytest_cache src/*.egg-info build dist checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
