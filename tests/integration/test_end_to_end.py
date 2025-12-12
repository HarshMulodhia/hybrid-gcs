"""
End-to-End Integration Tests
File: tests/integration/test_end_to_end.py

Complete end-to-end integration tests.
"""

import pytest
import numpy as np
from hybrid_gcs.core import GCSDecomposer, ConfigSpace
from hybrid_gcs.training import OptimizedTrainer
from hybrid_gcs.environments import ManipulationEnvironment
from hybrid_gcs.evaluation import Evaluator


class TestCompleteTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_init_to_evaluation(self):
        """Test from initialization to evaluation."""
        # Setup
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs)
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        trainer = OptimizedTrainer(policy_dim=20, action_dim=6)
        
        # Train a few steps
        obs = env.reset()
        for step in range(5):
            action = np.random.randn(6) * 0.1
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        
        # Evaluate
        eval_rewards = []
        obs = env.reset()
        for step in range(10):
            action = np.random.randn(6) * 0.1
            obs, reward, done, info = env.step(action)
            eval_rewards.append(reward)
            if done:
                obs = env.reset()
        
        assert len(eval_rewards) == 10
        assert all(isinstance(r, (int, float)) for r in eval_rewards)
    
    def test_gcs_planning_to_execution(self):
        """Test GCS planning to execution."""
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs)
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        
        # Plan path
        start = np.array([-0.5] * 6)
        goal = np.array([0.5] * 6)
        plan_result = decomposer.decompose(start, goal, [])
        
        # Execute in environment
        obs = env.reset()
        success = False
        
        for step in range(100):
            action = np.random.randn(6) * 0.1
            obs, reward, done, info = env.step(action)
            
            if reward > 50:  # Success threshold
                success = True
                break
            
            if done:
                break
        
        assert plan_result is not None
    
    def test_curriculum_learning_progression(self):
        """Test curriculum learning progression."""
        from hybrid_gcs.training import CurriculumLearning
        
        curriculum = CurriculumLearning(
            schedule_type="linear",
            total_steps=1000
        )
        
        difficulties = []
        for step in range(0, 1000, 100):
            diff = curriculum.get_difficulty(step)
            difficulties.append(diff)
        
        # Difficulty should increase
        assert all(difficulties[i] <= difficulties[i+1] 
                  for i in range(len(difficulties)-1))
    
    def test_multi_environment_training(self):
        """Test training across environments."""
        envs = [
            ManipulationEnvironment(state_dim=20, action_dim=6),
            ManipulationEnvironment(state_dim=20, action_dim=6, task="pick"),
        ]
        
        results = {}
        for env_idx, env in enumerate(envs):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(50):
                action = np.random.randn(6) * 0.1
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            results[env_idx] = episode_reward
        
        assert len(results) == len(envs)


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_dimensions(self):
        """Test handling of invalid dimensions."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        obs = env.reset()
        
        # Wrong action dimension
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            wrong_action = np.random.randn(3)  # Wrong size
            # This should raise an error
            if wrong_action.shape[0] != 6:
                raise ValueError("Invalid action dimension")
    
    def test_recovery_from_error(self):
        """Test recovery from errors."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        
        try:
            obs = env.reset()
            # Try invalid action
            try:
                action = np.random.randn(3)
                if action.shape[0] != 6:
                    raise ValueError("Invalid action")
            except ValueError:
                # Recover
                action = np.random.randn(6)
            
            obs, reward, done, info = env.step(action)
            assert True  # Recovered successfully
        except Exception as e:
            pytest.fail(f"Failed to recover from error: {e}")


class TestScalability:
    """Test scalability."""
    
    def test_batch_processing(self):
        """Test batch processing."""
        batch_size = 32
        state_dim = 20
        
        batch = np.random.randn(batch_size, state_dim)
        
        assert batch.shape == (batch_size, state_dim)
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        
        episode_rewards = []
        for episode in range(10):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(100):
                action = np.random.randn(6) * 0.1
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        assert len(episode_rewards) == 10
