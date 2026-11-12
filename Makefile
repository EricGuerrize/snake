.PHONY: install install-dev lint test train gif report dashboard clean ci

PYTHON ?= python3
EPISODES ?= 800
SEED ?= 42

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,dashboard]"

lint:
	$(PYTHON) -m ruff check snake_rl tests
	$(PYTHON) -m ruff format --check snake_rl tests

format:
	$(PYTHON) -m ruff format snake_rl tests
	$(PYTHON) -m ruff check --fix snake_rl tests

test:
	$(PYTHON) -m pytest -q

train:
	$(PYTHON) -m snake_rl.cli train --episodes $(EPISODES) --seed $(SEED) \
		--weights models/snake_dqn.npz --curve docs/learning_curve.png

gif:
	$(PYTHON) -m snake_rl.cli gif --weights models/snake_dqn.npz --out docs/snake_play.gif

report:
	$(PYTHON) -m snake_rl.cli report

dashboard:
	$(PYTHON) -m snake_rl.cli dashboard

ci: lint test

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
