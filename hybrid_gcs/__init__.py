"""
Hybrid-GCS: Complete Package Initialization
File: hybrid_gcs/__init__.py

Main entry point for the Hybrid-GCS package.
Combines GCS decomposition with reinforcement learning for advanced motion planning.
"""

__version__ = "2.0.0"
__author__ = "Hybrid-GCS Team"
__description__ = "Hybrid Grasp Planning System: GCS + Reinforcement Learning"
__license__ = "Apache 2.0"

# Core imports
from hybrid_gcs.core import (
    GCSDecomposer,
    TrajectoryOptimizer,
    PolicyNetwork,
    ActorNetwork,
    CriticNetwork,
    ConfigSpace,
    CollisionChecker,
)

# Training imports
from hybrid_gcs.training import (
    OptimizedTrainer,
    TrainingConfig,
    CurriculumLearning,
    CurriculumSchedule,
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    RewardShaper,
)

# Evaluation imports
from hybrid_gcs.evaluation import (
    Evaluator,
    EvaluationConfig,
    MetricsComputer,
    PerformanceMetrics,
)

# Visualization imports
from hybrid_gcs.visualization import (
    Dashboard,
    DashboardConfig,
    PlotGenerator,
    ReportGenerator,
    TrajectoryVisualizer,
)

# Utilities
from hybrid_gcs.utils import (
    setup_logging,
    load_config,
    save_config,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__description__",
    
    # Core
    "GCSDecomposer",
    "TrajectoryOptimizer",
    "PolicyNetwork",
    "ActorNetwork",
    "CriticNetwork",
    "ConfigSpace",
    "CollisionChecker",
    
    # Training
    "OptimizedTrainer",
    "TrainingConfig",
    "CurriculumLearning",
    "CurriculumSchedule",
    "Callback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "RewardShaper",
    
    # Evaluation
    "Evaluator",
    "EvaluationConfig",
    "MetricsComputer",
    "PerformanceMetrics",
    
    # Visualization
    "Dashboard",
    "DashboardConfig",
    "PlotGenerator",
    "ReportGenerator",
    "TrajectoryVisualizer",
    
    # Utilities
    "setup_logging",
    "load_config",
    "save_config",
]

# Lazy imports for submodules
def __getattr__(name):
    """Lazy load submodules."""
    if name == "environments":
        from hybrid_gcs import environments
        return environments
    elif name == "integration":
        from hybrid_gcs import integration
        return integration
    elif name == "cli":
        from hybrid_gcs import cli
        return cli
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
