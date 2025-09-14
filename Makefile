# AI Test Automation Platform Makefile

.PHONY: help install dev-install test lint format type-check clean docker-build docker-run docker-compose-up docker-compose-down init-examples deploy-staging deploy-prod

# Default target
help: ## Show this help message
	@echo "AI Test Automation Platform - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install black flake8 mypy pytest pytest-cov pytest-asyncio

# Development
test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black
	black src/ tests/

type-check: ## Run type checking
	mypy src/ --ignore-missing-imports

check-all: lint type-check test ## Run all checks

# Database and data
init-db: ## Initialize database and vector store
	python main.py init
	python scripts/init_vector_store.py

# Docker
docker-build: ## Build Docker image
	docker build -t ai-test-platform:latest .

docker-build-dev: ## Build development Docker image
	docker build --target development -t ai-test-platform:dev .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env ai-test-platform:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

# Kubernetes
k8s-apply: ## Apply Kubernetes manifests
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/ingress.yaml

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f k8s/ingress.yaml
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/namespace.yaml

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods -n ai-test-automation
	kubectl get services -n ai-test-automation
	kubectl get ingress -n ai-test-automation

# Application
start: ## Start the application
	python main.py dashboard

start-dev: ## Start in development mode
	uvicorn src.dashboard.dashboard_api:app --reload --host 0.0.0.0 --port 8000

run-examples: ## Run example workflows
	python scripts/run_examples.py

run-ingest: ## Run data ingestion
	python main.py ingest

# Monitoring
monitor-start: ## Start monitoring stack
	docker-compose -f docker-compose.yml up -d prometheus grafana

monitor-stop: ## Stop monitoring stack
	docker-compose -f docker-compose.yml stop prometheus grafana

# Cleanup
clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

clean-data: ## Clean up data directories
	rm -rf data/
	rm -rf logs/
	rm -rf screenshots/
	rm -rf test_results/

# Security
security-check: ## Run security checks
	bandit -r src/
	safety check

# Documentation
docs: ## Generate documentation
	@echo "Documentation is available in README.md and IMPLEMENTATION_SUMMARY.md"

# CI/CD
ci-test: ## Run CI tests (used in GitHub Actions)
	pip install -r requirements.txt
	pip install black flake8 mypy pytest pytest-cov
	black --check src/
	flake8 src/
	mypy src/ --ignore-missing-imports
	pytest tests/ --cov=src --cov-report=xml

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	kubectl apply -f k8s/staging/

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	kubectl apply -f k8s/production/

# Environment setup
setup-env: ## Set up environment file
	cp env.example .env
	@echo "Please edit .env file with your configuration"

# Backup and restore
backup: ## Backup data
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ logs/

restore: ## Restore data from backup
	@echo "Usage: make restore BACKUP_FILE=backup-YYYYMMDD-HHMMSS.tar.gz"
	tar -xzf $(BACKUP_FILE)

# Performance testing
perf-test: ## Run performance tests
	pytest tests/performance/ -v

# Load testing
load-test: ## Run load tests
	@echo "Load testing with locust..."
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Health check
health: ## Check application health
	curl -f http://localhost:8000/health || exit 1

# Logs
logs: ## View application logs
	tail -f logs/app.log

logs-docker: ## View Docker logs
	docker-compose logs -f ai-test-platform
