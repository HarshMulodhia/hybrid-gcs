"""
Utils Tests
File: tests/unit/test_utils.py

Unit tests for utility modules.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from hybrid_gcs.utils import (
    ConfigManager,
    DataBuffer,
    normalize_data,
)


class TestConfigManager:
    """Test configuration manager."""
    
    def test_yaml_loading(self, tmp_test_dir):
        """Test YAML configuration loading."""
        config_file = tmp_test_dir / "test_config.yaml"
        
        config_content = """
learning_rate: 0.001
batch_size: 32
epochs: 100
optimizer:
  type: adam
  beta1: 0.9
  beta2: 0.999
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = ConfigManager(str(config_file))
        
        assert config.get('learning_rate') == 0.001
        assert config.get('batch_size') == 32
        assert config.get('optimizer.type') == 'adam'
    
    def test_dot_notation_access(self, tmp_test_dir):
        """Test dot notation access."""
        config_file = tmp_test_dir / "test.yaml"
        
        config_content = """
nested:
  level1:
    level2: value
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        config = ConfigManager(str(config_file))
        
        assert config.get('nested.level1.level2') == 'value'
    
    def test_default_values(self, tmp_test_dir):
        """Test default value handling."""
        config_file = tmp_test_dir / "test.yaml"
        
        with open(config_file, 'w') as f:
            f.write("key: value\n")
        
        config = ConfigManager(str(config_file))
        
        assert config.get('nonexistent', 'default') == 'default'


class TestDataBuffer:
    """Test data buffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = DataBuffer(max_size=100, data_dim=6)
        
        assert buffer.max_size == 100
        assert buffer.data_dim == 6
        assert len(buffer) == 0
    
    def test_add_data(self):
        """Test adding data to buffer."""
        buffer = DataBuffer(max_size=10, data_dim=3)
        
        data = np.array([1.0, 2.0, 3.0])
        buffer.add(data)
        
        assert len(buffer) == 1
    
    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        buffer = DataBuffer(max_size=5, data_dim=2)
        
        # Add 7 items to buffer with max size 5
        for i in range(7):
            buffer.add(np.array([float(i), float(i)]))
        
        # Buffer should only contain last 5 items
        assert len(buffer) <= 5
    
    def test_get_batch(self):
        """Test getting batch from buffer."""
        buffer = DataBuffer(max_size=20, data_dim=3)
        
        for i in range(10):
            buffer.add(np.ones(3) * i)
        
        batch = buffer.get_batch(5)
        
        assert batch.shape[0] == 5
        assert batch.shape[1] == 3


class TestNormalization:
    """Test data normalization."""
    
    def test_normalize_data(self):
        """Test data normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        normalized = normalize_data(data)
        
        # Check shape preservation
        assert normalized.shape == data.shape
        
        # Check normalization (should be in roughly [-1, 1] range)
        assert np.max(np.abs(normalized)) <= 1.0
    
    def test_normalize_preserves_shape(self):
        """Test normalization preserves shape."""
        shapes = [(10, 5), (100, 20), (50,)]
        
        for shape in shapes:
            data = np.random.randn(*shape)
            normalized = normalize_data(data)
            
            assert normalized.shape == shape
    
    def test_normalize_zero_variance(self):
        """Test normalization with zero variance."""
        data = np.ones((5, 3))
        
        normalized = normalize_data(data)
        
        # Should handle gracefully
        assert normalized is not None
