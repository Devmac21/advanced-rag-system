# Makefile for Advanced RAG System

.PHONY: help install install-dev clean test lint format run-example serve docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make clean         - Clean generated files"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make run-example   - Run quickstart example"
	@echo "  make serve         - Start web UI"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start with Docker Compose"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

install-dev:
	pip install -r requirements-dev.txt
	python -m spacy download en_core_web_sm
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

test:
	pytest tests/ -v --cov=src/rag_system --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

run-example:
	python examples/quickstart.py

serve:
	streamlit run web/app.py

docker-build:
	docker build -t advanced-rag .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
