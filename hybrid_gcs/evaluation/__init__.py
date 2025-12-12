"""
Evaluation Module Initialization
File: hybrid_gcs/evaluation/__init__.py
"""

from hybrid_gcs.evaluation.evaluator import Evaluator, EvaluationConfig
from hybrid_gcs.evaluation.metrics import MetricsComputer, PerformanceMetrics
from hybrid_gcs.evaluation.comparator import MethodComparator, ComparisonResult
from hybrid_gcs.evaluation.analysis import StatisticalAnalysis

__all__ = [
    "Evaluator",
    "EvaluationConfig",
    "MetricsComputer",
    "PerformanceMetrics",
    "MethodComparator",
    "ComparisonResult",
    "StatisticalAnalysis",
]