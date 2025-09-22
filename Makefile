.PHONY: help install dev-install clean lint test quickstart ingest features analysis figures docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install package in production mode"
	@echo "  dev-install   Install package in development mode with dev dependencies"
	@echo "  clean         Clean build artifacts and cache"
	@echo "  lint          Run code linting (black, isort, flake8)"
	@echo "  test          Run unit tests"
	@echo "  quickstart    Run full pipeline: ingest -> features -> analysis"
	@echo "  ingest        Run data ingestion pipeline"
	@echo "  features      Run feature engineering pipeline"
	@echo "  analysis      Run analysis pipeline"
	@echo "  figures       Generate figures and visualizations"
	@echo "  docs          Build documentation"

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

# Development
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

test:
	pytest tests/ -v

# Data pipeline
quickstart: ingest features analysis figures

ingest:
	python -m src.pipelines.run_ingest

features:
	python -m src.pipelines.run_features

analysis:
	python -m src.pipelines.run_analysis

figures:
	python -m scripts.make_figures.sh

# Documentation
docs:
	cd docs && make html

# Environment setup
setup-env:
	cp .env.example .env
	@echo "Please edit .env with your API keys"

# Data management
clean-data:
	rm -rf data/raw/*
	rm -rf data/interim/*
	rm -rf data/processed/*

# Jupyter
notebook:
	jupyter notebook notebooks/

lab:
	jupyter lab notebooks/
