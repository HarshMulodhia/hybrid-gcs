"""
Test Configuration and Fixtures
File: tests/conftest.py

Pytest configuration with shared fixtures.
"""

import pytest
import numpy as np
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def sample_state():
    """Sample state for testing."""
    return np.array([0.5, 0.5, 0.5, 0, 0, 0])


@pytest.fixture
def sample_goal():
    """Sample goal state."""
    return np.array([1.5, 1.5, 1.5, 0, 0, 0])


@pytest.fixture
def sample_obstacles():
    """Sample obstacles for testing."""
    return [
        (np.array([1.0, 1.0, 1.0]), 0.2),
        (np.array([0.5, 1.5, 0.5]), 0.15),
    ]


@pytest.fixture
def sample_trajectory():
    """Generate sample trajectory."""
    t = np.linspace(0, 1, 50)
    trajectory = np.array([[0.5 + t_i, 0.5 + t_i, 0.5 + t_i] for t_i in t])
    return trajectory


@pytest.fixture
def config_space_dim():
    """Configuration space dimension."""
    return 6


@pytest.fixture
def action_dim():
    """Action dimension."""
    return 6


@pytest.fixture
def state_dim():
    """State dimension."""
    return 20


@pytest.fixture
def tmp_test_dir(tmp_path):
    """Create temporary test directory."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir
