"""
Curriculum Learning for Progressive Training
File: hybrid_gcs/training/curriculum.py

Implements progressive curriculum learning strategies for efficient RL training.
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CurriculumSchedule(Enum):
    """Types of curriculum schedules."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    SIGMOID = "sigmoid"


@dataclass
class TaskDifficulty:
    """Represents a task at a specific difficulty level."""
    
    level: int
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    config: Dict[str, Any]
    min_success_rate: float = 0.8
    episodes_per_task: int = 1000


class CurriculumLearning:
    """
    Progressive curriculum learning manager.
    
    Gradually increases task difficulty as the agent improves.
    """
    
    def __init__(
        self,
        initial_difficulty: float = 0.0,
        final_difficulty: float = 1.0,
        schedule: CurriculumSchedule = CurriculumSchedule.LINEAR,
        num_tasks: int = 5,
        milestone_episodes: Optional[List[int]] = None,
    ):
        """
        Initialize curriculum learning.
        
        Args:
            initial_difficulty: Starting difficulty (0-1)
            final_difficulty: Target difficulty (0-1)
            schedule: Difficulty schedule type
            num_tasks: Number of difficulty levels
            milestone_episodes: Episodes at which to transition
        """
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.schedule = schedule
        self.num_tasks = num_tasks
        
        self.current_task = 0
        self.current_difficulty = initial_difficulty
        self.episode_count = 0
        self.task_success_count = 0
        self.task_episodes = 0
        
        # Create milestones
        if milestone_episodes is None:
            self.milestone_episodes = [
                int((i + 1) * 1_000_000 / num_tasks)
                for i in range(num_tasks)
            ]
        else:
            self.milestone_episodes = milestone_episodes
        
        logger.info(
            f"Initialized CurriculumLearning: "
            f"difficulty {initial_difficulty}->{final_difficulty}, "
            f"schedule={schedule}, tasks={num_tasks}"
        )
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def get_task_config(self, difficulty: float) -> Dict[str, Any]:
        """
        Generate task configuration for given difficulty.
        
        Args:
            difficulty: Difficulty level (0-1)
            
        Returns:
            Task configuration dictionary
        """
        config = {
            'difficulty': difficulty,
            'obstacle_density': difficulty * 0.8,
            'collision_penalty': -100 * difficulty,
            'time_limit': int(500 * (1 + difficulty)),
            'task_type': self._get_task_type(difficulty),
        }
        return config
    
    def update_progress(self, success: bool) -> bool:
        """
        Update curriculum progress.
        
        Args:
            success: Whether task was successful
            
        Returns:
            True if curriculum advanced
        """
        self.episode_count += 1
        self.task_episodes += 1
        
        if success:
            self.task_success_count += 1
        
        # Check if should advance curriculum
        if self._should_advance():
            return self.advance()
        
        # Update difficulty based on schedule
        self._update_difficulty()
        
        return False
    
    def advance(self) -> bool:
        """
        Advance to next curriculum task.
        
        Returns:
            True if advanced, False if at final task
        """
        if self.current_task >= self.num_tasks - 1:
            logger.info("Curriculum complete - at final task")
            return False
        
        self.current_task += 1
        self.task_success_count = 0
        self.task_episodes = 0
        
        self.current_difficulty = self._get_difficulty_for_task(self.current_task)
        
        logger.info(
            f"Advanced curriculum to task {self.current_task}, "
            f"difficulty={self.current_difficulty:.3f}"
        )
        
        return True
    
    def _should_advance(self) -> bool:
        """Check if curriculum should advance."""
        if self.current_task >= self.num_tasks - 1:
            return False
        
        # Advance if success rate is high
        if self.task_episodes >= 100:
            success_rate = self.task_success_count / self.task_episodes
            if success_rate >= 0.8:
                return True
        
        # Advance based on episode milestones
        if self.episode_count in self.milestone_episodes:
            return True
        
        return False
    
    def _update_difficulty(self) -> None:
        """Update difficulty based on schedule."""
        progress = min(self.episode_count / self.milestone_episodes[-1], 1.0)
        
        if self.schedule == CurriculumSchedule.LINEAR:
            self.current_difficulty = (
                self.initial_difficulty +
                (self.final_difficulty - self.initial_difficulty) * progress
            )
        elif self.schedule == CurriculumSchedule.EXPONENTIAL:
            self.current_difficulty = (
                self.initial_difficulty +
                (self.final_difficulty - self.initial_difficulty) * (progress ** 2)
            )
        elif self.schedule == CurriculumSchedule.SIGMOID:
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            self.current_difficulty = (
                self.initial_difficulty +
                (self.final_difficulty - self.initial_difficulty) * sigmoid
            )
    
    def _get_difficulty_for_task(self, task: int) -> float:
        """Get difficulty for task index."""
        progress = task / max(self.num_tasks - 1, 1)
        
        if self.schedule == CurriculumSchedule.LINEAR:
            return self.initial_difficulty + (
                self.final_difficulty - self.initial_difficulty
            ) * progress
        elif self.schedule == CurriculumSchedule.EXPONENTIAL:
            return self.initial_difficulty + (
                self.final_difficulty - self.initial_difficulty
            ) * (progress ** 2)
        
        return self.initial_difficulty + (
            self.final_difficulty - self.initial_difficulty
        ) * progress
    
    def _get_task_type(self, difficulty: float) -> str:
        """Get task type for difficulty level."""
        if difficulty < 0.25:
            return "simple_reach"
        elif difficulty < 0.5:
            return "reach_obstacle_free"
        elif difficulty < 0.75:
            return "reach_with_obstacles"
        else:
            return "complex_manipulation"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            'current_task': self.current_task,
            'current_difficulty': self.current_difficulty,
            'total_episodes': self.episode_count,
            'task_episodes': self.task_episodes,
            'task_success_count': self.task_success_count,
            'success_rate': (
                self.task_success_count / max(self.task_episodes, 1)
            ),
        }
    
    def __repr__(self) -> str:
        return (
            f"CurriculumLearning(task={self.current_task}/{self.num_tasks}, "
            f"difficulty={self.current_difficulty:.3f})"
        )
