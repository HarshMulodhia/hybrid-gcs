"""
Training Module Initialization
File: hybrid_gcs/training/__init__.py
"""

from hybrid_gcs.training.optimized_trainer import OptimizedTrainer, TrainingConfig
from hybrid_gcs.training.curriculum import CurriculumLearning, CurriculumSchedule
from hybrid_gcs.training.callbacks import (
    Callback, 
    CheckpointCallback, 
    EarlyStoppingCallback,
    LoggingCallback,
)
from hybrid_gcs.training.reward_shaping import (
    RewardShaper,
    DistanceReward,
    SmoothPathReward,
    EfficiencyReward,
)

__all__ = [
    "OptimizedTrainer",
    "TrainingConfig",
    "CurriculumLearning",
    "CurriculumSchedule",
    "Callback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "RewardShaper",
    "DistanceReward",
    "SmoothPathReward",
    "EfficiencyReward",
]