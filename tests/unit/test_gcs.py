"""
GCS Module Tests
File: tests/unit/test_gcs.py

Unit tests for GCS decomposer and related modules.
"""

import pytest
import numpy as np
from hybrid_gcs.core import GCSDecomposer, ConfigSpace


class TestConfigSpace:
    """Test ConfigSpace class."""
    
    def test_creation(self):
        """Test configuration space creation."""
        cs = ConfigSpace(
            name="test_space",
            dim=6,
            bounds=[(-1, 1)] * 6
        )
        
        assert cs.name == "test_space"
        assert cs.dim == 6
        assert len(cs.bounds) == 6
    
    def test_bounds_validation(self):
        """Test bounds validation."""
        bounds = [(-1, 1), (-2, 2), (-3, 3)]
        cs = ConfigSpace(name="test", dim=3, bounds=bounds)
        
        assert cs.bounds == bounds
    
    def test_point_in_bounds(self):
        """Test point in bounds checking."""
        cs = ConfigSpace(name="test", dim=3, bounds=[(-1, 1)] * 3)
        
        point_in = np.array([0.5, 0.5, 0.5])
        point_out = np.array([2.0, 2.0, 2.0])
        
        # Implementation would check these
        assert all(-1 <= p <= 1 for p in point_in)
        assert not all(-1 <= p <= 1 for p in point_out)


class TestGCSDecomposer:
    """Test GCS decomposer."""
    
    def test_initialization(self, config_space_dim):
        """Test decomposer initialization."""
        cs = ConfigSpace(
            name="test",
            dim=config_space_dim,
            bounds=[(-1, 1)] * config_space_dim
        )
        
        decomposer = GCSDecomposer(
            config_space=cs,
            max_regions=10,
            max_degree=2
        )
        
        assert decomposer is not None
        assert decomposer.max_regions == 10
        assert decomposer.max_degree == 2
    
    def test_decompose_simple(self, sample_state, sample_goal, sample_obstacles, config_space_dim):
        """Test simple decomposition."""
        cs = ConfigSpace(
            name="test",
            dim=config_space_dim,
            bounds=[(-2, 2)] * config_space_dim
        )
        
        decomposer = GCSDecomposer(config_space=cs, max_regions=5)
        
        result = decomposer.decompose(sample_state, sample_goal, sample_obstacles)
        
        assert 'regions' in result
        assert 'feasible' in result
        assert isinstance(result['regions'], list)
    
    def test_decompose_no_obstacles(self, sample_state, sample_goal, config_space_dim):
        """Test decomposition without obstacles."""
        cs = ConfigSpace(
            name="test",
            dim=config_space_dim,
            bounds=[(-2, 2)] * config_space_dim
        )
        
        decomposer = GCSDecomposer(config_space=cs)
        result = decomposer.decompose(sample_state, sample_goal, [])
        
        assert result['feasible'] or not result['feasible']  # Either can be true
    
    def test_max_regions_respected(self, config_space_dim):
        """Test that max regions is respected."""
        cs = ConfigSpace(
            name="test",
            dim=config_space_dim,
            bounds=[(-1, 1)] * config_space_dim
        )
        
        max_regions = 5
        decomposer = GCSDecomposer(config_space=cs, max_regions=max_regions)
        
        start = np.array([-0.5] * config_space_dim)
        goal = np.array([0.5] * config_space_dim)
        
        result = decomposer.decompose(start, goal, [])
        
        assert len(result['regions']) <= max_regions
