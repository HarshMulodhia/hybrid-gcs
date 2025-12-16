"""
Method Comparison Framework
File: hybrid_gcs/evaluation/comparator.py

Compares different planning/learning methods for benchmarking.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results of method comparison."""
    
    method_a: str
    method_b: str
    metric: str
    
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    p_value: float
    significant: bool
    winner: str
    improvement: float  # Percentage improvement


class MethodComparator:
    """
    Compares performance of different methods.
    
    Performs statistical tests and generates comparison reports.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize comparator.
        
        Args:
            confidence_level: Confidence level for statistical tests (0-1)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results: List[ComparisonResult] = []
        
        logger.info(f"Initialized MethodComparator (confidence={confidence_level:.0%})")
    
    def compare_methods(
        self,
        method_a_results: Dict[str, List[float]],
        method_b_results: Dict[str, List[float]],
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
    ) -> Dict[str, ComparisonResult]:
        """
        Compare two methods across metrics.
        
        Args:
            method_a_results: Dict of {metric: [values]}
            method_b_results: Dict of {metric: [values]}
            method_a_name: Name of first method
            method_b_name: Name of second method
            
        Returns:
            Dict of {metric: ComparisonResult}
        """
        comparison_results = {}
        
        for metric in method_a_results.keys():
            if metric not in method_b_results:
                logger.warning(f"Metric {metric} not in method B")
                continue
            
            values_a = np.array(method_a_results[metric])
            values_b = np.array(method_b_results[metric])
            
            result = self._compare_metric(
                values_a, values_b,
                metric,
                method_a_name, method_b_name
            )
            
            comparison_results[metric] = result
            self.results.append(result)
        
        logger.info(f"Compared {len(comparison_results)} metrics")
        return comparison_results
    
    def _compare_metric(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
        metric: str,
        name_a: str,
        name_b: str,
    ) -> ComparisonResult:
        """
        Compare single metric between methods.
        
        Args:
            values_a: Values from method A
            values_b: Values from method B
            metric: Metric name
            name_a: Name of method A
            name_b: Name of method B
            
        Returns:
            ComparisonResult
        """
        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        std_a = float(np.std(values_a))
        std_b = float(np.std(values_b))
        
        # T-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        significant = p_value < self.alpha
        
        # Determine winner
        if mean_a > mean_b:
            winner = name_a
            improvement = ((mean_a - mean_b) / (mean_b + 1e-6)) * 100
        else:
            winner = name_b
            improvement = ((mean_b - mean_a) / (mean_a + 1e-6)) * 100
        
        result = ComparisonResult(
            method_a=name_a,
            method_b=name_b,
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            p_value=p_value,
            significant=significant,
            winner=winner,
            improvement=improvement,
        )
        
        logger.debug(
            f"Metric {metric}: {name_a}={mean_a:.3f}±{std_a:.3f} vs "
            f"{name_b}={mean_b:.3f}±{std_b:.3f} (p={p_value:.4f})"
        )
        
        return result
    
    def multi_method_comparison(
        self,
        methods_results: Dict[str, Dict[str, List[float]]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple methods across metrics.
        
        Args:
            methods_results: Dict of {method_name: {metric: [values]}}
            metrics: Optional list of metrics to compare
            
        Returns:
            Dict of {metric: comparison_stats}
        """
        results = {}
        
        method_names = list(methods_results.keys())
        if metrics is None:
            metrics = list(methods_results[method_names[0]].keys())
        
        for metric in metrics:
            # Collect all values for this metric
            all_values = {}
            for method_name in method_names:
                if metric in methods_results[method_name]:
                    all_values[method_name] = np.array(
                        methods_results[method_name][metric]
                    )
            
            if len(all_values) < 2:
                continue
            
            # ANOVA test
            values_list = list(all_values.values())
            f_stat, p_value = stats.f_oneway(*values_list)
            
            # Compute stats for each method
            stats_dict = {}
            for method_name, values in all_values.items():
                stats_dict[method_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
            
            results[metric] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'methods': stats_dict,
                'best_method': max(
                    stats_dict.keys(),
                    key=lambda m: stats_dict[m]['mean']
                ),
            }
        
        logger.info(f"Multi-method comparison: {len(method_names)} methods, {len(results)} metrics")
        return results
    
    def effect_size(self, values_a: np.ndarray, values_b: np.ndarray) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            values_a: Values from method A
            values_b: Values from method B
            
        Returns:
            Cohen's d value
        """
        mean_diff = np.mean(values_a) - np.mean(values_b)
        pooled_std = np.sqrt(
            (np.std(values_a) ** 2 + np.std(values_b) ** 2) / 2
        )
        
        return float(mean_diff / (pooled_std + 1e-6))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        if not self.results:
            return {}
        
        significant_count = sum(1 for r in self.results if r.significant)
        
        return {
            'total_comparisons': len(self.results),
            'significant_differences': significant_count,
            'significance_rate': significant_count / len(self.results),
            'average_improvement': np.mean([r.improvement for r in self.results]),
        }

@dataclass
class ComparisonResult:
    """
    Results from comparing two methods.
    
    Attributes:
        method_a_name: Name of first method
        method_b_name: Name of second method
        method_a_mean: Mean score of method A
        method_b_mean: Mean score of method B
        method_a_std: Std dev of method A
        method_b_std: Std dev of method B
        improvement: Percent improvement from A to B
        statistical_significant: Whether difference is significant
        p_value: P-value from statistical test
        effect_size: Effect size (Cohen's d or similar)
        notes: Additional notes
    """
    method_a_name: str
    method_b_name: str
    method_a_mean: float
    method_b_mean: float
    method_a_std: float
    method_b_std: float
    improvement: float
    statistical_significant: bool
    p_value: float
    effect_size: float
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "method_a_name": self.method_a_name,
            "method_b_name": self.method_b_name,
            "method_a_mean": self.method_a_mean,
            "method_b_mean": self.method_b_mean,
            "method_a_std": self.method_a_std,
            "method_b_std": self.method_b_std,
            "improvement_percent": self.improvement,
            "statistical_significant": self.statistical_significant,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "notes": self.notes,
            **self.metadata,
        }
    
    def __str__(self) -> str:
        """String representation."""
        sig_str = "✓ Significant" if self.statistical_significant else "✗ Not significant"
        return (
            f"Comparison Result: {self.method_a_name} vs {self.method_b_name}\n"
            f"  {self.method_a_name}: {self.method_a_mean:.4f} ± {self.method_a_std:.4f}\n"
            f"  {self.method_b_name}: {self.method_b_mean:.4f} ± {self.method_b_std:.4f}\n"
            f"  Improvement: {self.improvement:.2f}%\n"
            f"  Effect Size: {self.effect_size:.4f}\n"
            f"  P-value: {self.p_value:.4f} ({sig_str})\n"
            f"  Notes: {self.notes}"
        )
    
    def winner(self) -> str:
        """Get the winning method name."""
        if self.method_b_mean > self.method_a_mean:
            return self.method_b_name
        elif self.method_a_mean > self.method_b_mean:
            return self.method_a_name
        else:
            return "Tie"


class ResultsValidator:
    """Validate and check comparison results."""
    
    @staticmethod
    def validate_results(
        method_a_data: np.ndarray,
        method_b_data: np.ndarray,
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Validate results data.
        
        Args:
            method_a_data: Results from method A
            method_b_data: Results from method B
            min_samples: Minimum samples required
            
        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Check sample sizes
        if len(method_a_data) < min_samples:
            report["errors"].append(
                f"Method A: insufficient samples ({len(method_a_data)} < {min_samples})"
            )
            report["valid"] = False
        
        if len(method_b_data) < min_samples:
            report["errors"].append(
                f"Method B: insufficient samples ({len(method_b_data)} < {min_samples})"
            )
            report["valid"] = False
        
        # Check for NaN values
        a_nans = np.isnan(method_a_data).sum()
        b_nans = np.isnan(method_b_data).sum()
        
        if a_nans > 0:
            report["warnings"].append(f"Method A: {a_nans} NaN values")
        if b_nans > 0:
            report["warnings"].append(f"Method B: {b_nans} NaN values")
        
        # Check for infinite values
        a_infs = np.isinf(method_a_data).sum()
        b_infs = np.isinf(method_b_data).sum()
        
        if a_infs > 0:
            report["errors"].append(f"Method A: {a_infs} infinite values")
            report["valid"] = False
        if b_infs > 0:
            report["errors"].append(f"Method B: {b_infs} infinite values")
            report["valid"] = False
        
        return report
    
    @staticmethod
    def check_practical_significance(
        method_a_mean: float,
        method_b_mean: float,
        effect_size: float,
        min_effect_size: float = 0.2,
    ) -> str:
        """
        Check if improvement is practically significant.
        
        Args:
            method_a_mean: Mean of method A
            method_b_mean: Mean of method B
            effect_size: Effect size (Cohen's d)
            min_effect_size: Minimum effect size threshold
            
        Returns:
            Significance assessment
        """
        if abs(effect_size) < min_effect_size:
            return "negligible"
        elif abs(effect_size) < 0.5:
            return "small"
        elif abs(effect_size) < 0.8:
            return "medium"
        else:
            return "large"