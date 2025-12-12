# Installation Guide

Complete installation instructions for Hybrid-GCS on all platforms.

## Prerequisites

- Python 3.8+
- pip or conda
- Virtual environment (recommended)
- Git (for development)

## Quick Install

### Local Installation

```bash
# 1. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install package
pip install hybrid-gcs

# 3. Verify installation
python -c "from hybrid_gcs import GCSDecomposer; print('✅ Installation successful!')"
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/hybrid-gcs/hybrid-gcs.git
cd hybrid-gcs

# Create environment
python3.10 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,docs,jupyter]"

# Run tests
pytest tests/
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Verify
docker-compose exec hybrid-gcs python -c "from hybrid_gcs import GCSDecomposer; print('✅ OK')"
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Follow Quick Install steps above
```

### macOS

```bash
# Install with Homebrew
brew install python@3.10

# Create environment
python3.10 -m venv venv
source venv/bin/activate

# Install
pip install hybrid-gcs
```

### Windows

```bash
# Using PowerShell as Administrator
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install
pip install hybrid-gcs
```

## Optional Dependencies

```bash
# For visualization
pip install hybrid-gcs[visualization]

# For Jupyter notebooks
pip install hybrid-gcs[jupyter]

# For documentation
pip install hybrid-gcs[docs]

# For physics simulation
pip install hybrid-gcs[simulation]

# For all extras
pip install hybrid-gcs[dev,docs,jupyter,visualization,simulation]
```

## Verification

After installation, verify everything works:

```bash
# Test import
python -c "from hybrid_gcs import *; print('✅ All imports successful')"

# Run basic test
pytest tests/unit/test_gcs.py -v

# Start Jupyter
jupyter notebook notebooks/01_getting_started.ipynb
```

## Troubleshooting

### Import Error

```python
# If you get "ModuleNotFoundError: No module named 'hybrid_gcs'"
# Make sure you're in the virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Then reinstall
pip install -e .
```

### CUDA/GPU Issues

```bash
# For GPU support with PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set device to CPU if GPU unavailable
export CUDA_VISIBLE_DEVICES=""
```

### Docker Issues

```bash
# Clean and rebuild
docker-compose down --volumes
docker-compose up --build
```

## Next Steps

- [Getting Started Guide](getting_started.md)
- [Quick Tutorial](examples/basic_usage.md)
- [Full API Reference](api_reference.md)

---

**Still having issues?** Check [Troubleshooting](troubleshooting.md) or open an [issue on GitHub](https://github.com/hybrid-gcs/hybrid-gcs/issues).
