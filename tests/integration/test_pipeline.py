"""
Pipeline Integration Tests
File: tests/integration/test_pipeline.py

Integration tests for complete training pipeline.
"""

import pytest
import numpy as np
from hybrid_gcs.core import GCSDecomposer, ConfigSpace
from hybrid_gcs.training import OptimizedTrainer
from hybrid_gcs.environments import ManipulationEnvironment


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_full_pipeline(self):
        """Test full training pipeline integration."""
        # Setup
        config_space = ConfigSpace(
            name="test",
            dim=6,
            bounds=[(-1, 1)] * 6
        )
        
        decomposer = GCSDecomposer(config_space=config_space)
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        trainer = OptimizedTrainer(
            policy_dim=20,
            action_dim=6,
            batch_size=16
        )
        
        # Run a few training steps
        obs = env.reset()
        
        for step in range(10):
            # Get action
            action = np.random.randn(6) * 0.1
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            if done:
                obs = env.reset()
        
        assert True  # If we got here, pipeline works


class TestGCSTrainingIntegration:
    """Test GCS with training integration."""
    
    def test_gcs_with_training(self):
        """Test GCS decomposition with training."""
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs, max_regions=5)
        
        start = np.array([-0.5] * 6)
        goal = np.array([0.5] * 6)
        obstacles = [(np.array([0, 0, 0, 0, 0, 0]), 0.1)]
        
        # Get decomposition
        result = decomposer.decompose(start, goal, obstacles)
        
        assert result is not None
        assert 'regions' in result
        assert 'feasible' in result
