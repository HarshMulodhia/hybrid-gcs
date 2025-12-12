FROM python:3.10-slim

LABEL maintainer="Hybrid-GCS Team"
LABEL version="2.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/results/models \
    && mkdir -p /app/results/logs \
    && mkdir -p /app/deployment

# Expose ports
EXPOSE 6006  # TensorBoard
EXPOSE 8888  # Jupyter
EXPOSE 5000  # API

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from hybrid_gcs import GCSDecomposer; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "hybrid_gcs.cli.train_cli", "--help"]
