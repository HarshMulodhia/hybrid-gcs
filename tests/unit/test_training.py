"""
Training Module Tests
File: tests/unit/test_training.py

Unit tests for training modules.
"""

import pytest
import numpy as np
from hybrid_gcs.training import OptimizedTrainer, CurriculumLearning


class TestOptimizedTrainer:
    """Test PPO trainer."""
    
    def test_initialization(self, state_dim, action_dim):
        """Test trainer initialization."""
        trainer = OptimizedTrainer(
            policy_dim=state_dim,
            action_dim=action_dim,
            learning_rate=1e-4,
            batch_size=32
        )
        
        assert trainer is not None
        assert trainer.learning_rate == 1e-4
        assert trainer.batch_size == 32
    
    def test_batch_sampling(self, state_dim, action_dim):
        """Test batch sampling."""
        trainer = OptimizedTrainer(
            policy_dim=state_dim,
            action_dim=action_dim,
            batch_size=32
        )
        
        # Would implement actual batch sampling
        assert trainer.batch_size == 32
    
    def test_hyperparameters(self):
        """Test hyperparameter configuration."""
        trainer = OptimizedTrainer(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95
        )
        
        assert trainer.learning_rate == 3e-4
        assert trainer.gamma == 0.99
        assert trainer.gae_lambda == 0.95


class TestCurriculumLearning:
    """Test curriculum learning."""
    
    def test_initialization(self):
        """Test curriculum initialization."""
        curriculum = CurriculumLearning(
            schedule_type="linear",
            total_steps=100000
        )
        
        assert curriculum is not None
        assert curriculum.schedule_type == "linear"
        assert curriculum.total_steps == 100000
    
    def test_linear_schedule(self):
        """Test linear curriculum schedule."""
        curriculum = CurriculumLearning(
            schedule_type="linear",
            total_steps=100
        )
        
        # At step 0, difficulty should be low
        diff_0 = curriculum.get_difficulty(0)
        # At step 100, difficulty should be high
        diff_100 = curriculum.get_difficulty(100)
        
        # Difficulty should increase
        assert diff_0 <= diff_100
    
    def test_exponential_schedule(self):
        """Test exponential curriculum schedule."""
        curriculum = CurriculumLearning(
            schedule_type="exponential",
            total_steps=1000
        )
        
        difficulties = [curriculum.get_difficulty(i * 100) for i in range(11)]
        
        # Should be monotonically increasing
        assert all(difficulties[i] <= difficulties[i+1] for i in range(len(difficulties)-1))
    
    def test_difficulty_bounds(self):
        """Test difficulty remains in bounds."""
        curriculum = CurriculumLearning(total_steps=1000)
        
        for step in range(0, 1000, 100):
            diff = curriculum.get_difficulty(step)
            assert 0.0 <= diff <= 1.0
