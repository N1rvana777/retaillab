.PHONY: install dev test lint type-check format clean up down train

# ──────────────── Development ────────────────

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# ──────────────── Quality ────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=quant_engine --cov=data_pipeline --cov=api_gateway

lint:
	ruff check .

format:
	ruff format .

type-check:
	mypy quant_engine/ data_pipeline/ api_gateway/

# ──────────────── Docker ────────────────

up:
	docker compose -f infrastructure/docker-compose.yml up -d

down:
	docker compose -f infrastructure/docker-compose.yml down

train:
	docker compose -f infrastructure/docker-compose.yml --profile training up trainer

# ──────────────── Cleanup ────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
