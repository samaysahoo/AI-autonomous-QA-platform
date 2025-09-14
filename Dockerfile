# Multi-stage Docker build for AI Test Automation Platform

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/chroma_db data/faiss_index logs screenshots test_results

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "main.py", "dashboard"]

# Stage 3: Production image
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy source code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p data/chroma_db data/faiss_index logs screenshots test_results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "main.py", "dashboard"]

# Stage 4: Test image
FROM development as test

# Install additional test dependencies
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-mock \
    pytest-benchmark

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html"]
