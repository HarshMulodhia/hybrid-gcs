"""
Evaluation Module for Hybrid-GCS

Provides evaluation, metrics computation, method comparison, and statistical analysis.

Submodules:
- evaluator: Main evaluation pipeline
- metrics: Performance metrics computation
- comparator: Method comparison framework
- analysis: Statistical analysis tools

Author: Hybrid-GCS Team
Version: 1.0.0
"""

from hybrid_gcs.evaluation.evaluator import (
    Evaluator,
    EvaluationConfig,
)
from hybrid_gcs.evaluation.metrics import (
    MetricsComputer,
    PerformanceMetrics,
    TrajectoryMetrics,
)
from hybrid_gcs.evaluation.comparator import (
    MethodComparator,
    ComparisonResult,
)
from hybrid_gcs.evaluation.analysis import (
    StatisticalAnalysis,
    StatisticalResult,
    TrendAnalysis,
)

__version__ = "1.0.0"
__author__ = "Hybrid-GCS Team"

__all__ = [
    # Evaluator
    "Evaluator",
    "EvaluationConfig",
    
    # Metrics
    "MetricsComputer",
    "PerformanceMetrics",
    "TrajectoryMetrics",
    
    # Comparator
    "MethodComparator",
    "ComparisonResult",
    
    # Analysis
    "StatisticalAnalysis",
    "StatisticalResult",
    "TrendAnalysis",
]
