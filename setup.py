"""
Package setup script
File: setup.py

Standard setuptools configuration for Hybrid-GCS.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="hybrid-gcs",
    version="2.0.0",
    author="Hybrid-GCS Team",
    author_email="team@hybrid-gcs.dev",
    description="Hybrid GCS-RL for Robot Learning and Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hybrid-gcs/hybrid-gcs",
    project_urls={
        "Documentation": "https://hybrid-gcs.readthedocs.io",
        "Source Code": "https://github.com/hybrid-gcs/hybrid-gcs",
        "Issue Tracker": "https://github.com/hybrid-gcs/hybrid-gcs/issues",
    },
    
    packages=find_packages(),
    
    python_requires=">=3.8",
    
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "gym>=0.19.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "pytest-xdist>=2.3.0",
            "black>=21.6b0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "visualization": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "jupyterlab>=3.1.0",
        ],
        "simulation": [
            "pybullet>=3.1.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "hybrid-gcs=hybrid_gcs.cli:main",
            "hybrid-train=scripts.train:main",
            "hybrid-eval=scripts.evaluate:main",
            "hybrid-benchmark=scripts.benchmark:main",
            "hybrid-deploy=scripts.deploy:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    keywords=[
        "reinforcement learning",
        "robot learning",
        "control",
        "gcs",
        "trajectory optimization",
        "hybrid control",
    ],
    
    zip_safe=False,
    
    include_package_data=True,
)
