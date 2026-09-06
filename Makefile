.PHONY: help install lint test eval eval-semantic eval-live eval-baseline \
	locomo locomo-live check-bedrock serve trace-demo demo up down logs docker-demo clean

# Prefer uv when it is installed; fall back to the stdlib venv otherwise.
UV := $(shell command -v uv 2>/dev/null)
VENV := .venv
PY := $(VENV)/bin/python
CASES := eval/cases/core.jsonl

help:  ## Show available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the env and install the package with dev extras
ifdef UV
	uv venv $(VENV)
	uv pip install --python $(PY) -e ".[dev,embed,otel]"
else
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev,embed,otel]"
endif

lint:  ## Ruff + mypy --strict
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts
	$(VENV)/bin/mypy src

test:  ## Offline test suite (no network, no credentials)
	$(VENV)/bin/pytest

eval:  ## Benchmark all three arms, deterministic config (this is the CI gate)
	ENGRAM_LOG_LEVEL=WARNING $(PY) -m engram.eval.runner $(CASES)

eval-semantic:  ## Benchmark with real sentence embeddings (needs the embed extra)
	ENGRAM_LOG_LEVEL=WARNING $(PY) -m engram.eval.runner $(CASES) --embedder fastembed

eval-live:  ## Benchmark a balanced sample against real Bedrock (costs a little)
	ENGRAM_LOG_LEVEL=ERROR $(PY) -m engram.eval.runner $(CASES) \
		--provider bedrock --embedder bedrock --limit 2

eval-baseline:  ## Compare against mem0 on identical models (needs `make up` + baseline extra)
	ENGRAM_LOG_LEVEL=ERROR $(PY) -m engram.eval.runner $(CASES) \
		--provider bedrock --embedder bedrock --limit 2 --arms manager mem0

locomo:  ## Fetch LoCoMo, then run one conversation offline (a floor, not a score)
	$(PY) scripts/fetch_locomo.py
	ENGRAM_LOG_LEVEL=ERROR $(PY) -m engram.eval.locomo --conversations 1 --questions 30

locomo-live:  ## Run LoCoMo against real Bedrock with an LLM judge
	ENGRAM_LOG_LEVEL=ERROR $(PY) -m engram.eval.locomo --conversations 1 --questions 30 \
		--provider bedrock --embedder bedrock

check-bedrock:  ## Preflight the live path and compare Nova models
	$(PY) scripts/check_bedrock.py

serve:  ## Run the HTTP API on :8000 (docs at /docs)
	$(VENV)/bin/uvicorn engram.api.main:app --reload --port 8000

trace-demo:  ## Print OpenTelemetry spans for a couple of turns
	ENGRAM_OTEL_EXPORTER=console ENGRAM_LOG_LEVEL=ERROR $(PY) scripts/demo.py

demo:  ## Cross-session recall demo, in-process
	ENGRAM_LOG_LEVEL=WARNING $(PY) scripts/demo.py

up:  ## Start Postgres + pgvector
	docker compose up -d postgres

down:  ## Stop and remove Postgres, including its volume
	docker compose down -v

logs:  ## Tail Postgres logs
	docker compose logs -f postgres

docker-demo: up  ## Run the demo in Docker against Postgres
	docker compose --profile demo run --rm demo

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
