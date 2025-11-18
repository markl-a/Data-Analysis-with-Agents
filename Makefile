.PHONY: help build up down restart logs test clean install lint format docker-build docker-push

# Default target
help:
	@echo "Data Analysis with Chatbots - Makefile Commands"
	@echo "================================================"
	@echo "Docker Commands:"
	@echo "  make build         - Build Docker images"
	@echo "  make up            - Start all services"
	@echo "  make down          - Stop all services"
	@echo "  make restart       - Restart all services"
	@echo "  make logs          - View logs"
	@echo "  make shell         - Open shell in app container"
	@echo ""
	@echo "Development Commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo ""
	@echo "CI/CD Commands:"
	@echo "  make ci            - Run full CI pipeline locally"
	@echo "  make coverage      - Generate coverage report"

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec app /bin/bash

# Development commands
install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -v --cov=src/data_analysis_chatbots --cov-report=term-missing

coverage:
	pytest --cov=src/data_analysis_chatbots --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports
	bandit -r src/ -ll

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/

# CI pipeline
ci: format lint test
	@echo "CI pipeline completed successfully!"

# Docker registry
docker-build:
	docker build -t data-analysis-chatbots:latest .

docker-push:
	docker tag data-analysis-chatbots:latest yourusername/data-analysis-chatbots:latest
	docker push yourusername/data-analysis-chatbots:latest

# Run Streamlit locally
run:
	streamlit run app.py

# Quick start
quickstart: install
	@echo "Setup complete! Run 'make run' to start the app."
