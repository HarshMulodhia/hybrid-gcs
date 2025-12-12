"""
Statistical Analysis Module
File: hybrid_gcs/evaluation/analysis.py

Performs statistical analysis on evaluation results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
import json

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Statistical analysis result."""
    
    metric: str
    mean: float
    median: float
    std: float
    variance: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float


class StatisticalAnalyzer:
    """
    Performs statistical analysis on evaluation metrics.
    
    Provides descriptive statistics, hypothesis testing, and trend analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analyzer.
        
        Args:
            confidence_level: Confidence level for tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.analyses: Dict[str, AnalysisResult] = {}
        
        logger.info(f"Initialized StatisticalAnalyzer (confidence={confidence_level:.0%})")
    
    def analyze(self, data: Dict[str, List[float]]) -> Dict[str, AnalysisResult]:
        """
        Perform statistical analysis on data.
        
        Args:
            data: Dict of {metric: [values]}
            
        Returns:
            Dict of {metric: AnalysisResult}
        """
        results = {}
        
        for metric, values in data.items():
            values_array = np.array(values)
            
            result = AnalysisResult(
                metric=metric,
                mean=float(np.mean(values_array)),
                median=float(np.median(values_array)),
                std=float(np.std(values_array)),
                variance=float(np.var(values_array)),
                min_val=float(np.min(values_array)),
                max_val=float(np.max(values_array)),
                q25=float(np.percentile(values_array, 25)),
                q75=float(np.percentile(values_array, 75)),
                skewness=float(stats.skew(values_array)),
                kurtosis=float(stats.kurtosis(values_array)),
            )
            
            results[metric] = result
            self.analyses[metric] = result
            
            logger.debug(f"Analyzed {metric}: mean={result.mean:.3f}, std={result.std:.3f}")
        
        return results
    
    def confidence_interval(
        self,
        values: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval.
        
        Args:
            values: Data values
            
        Returns:
            (lower_bound, upper_bound)
        """
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = sem * stats.t.ppf((1 + self.confidence_level) / 2, len(values) - 1)
        
        return float(mean - ci), float(mean + ci)
    
    def normality_test(self, values: np.ndarray) -> Tuple[float, float, bool]:
        """
        Test normality using Shapiro-Wilk test.
        
        Args:
            values: Data values
            
        Returns:
            (statistic, p_value, is_normal)
        """
        stat, p_value = stats.shapiro(values)
        is_normal = p_value > self.alpha
        
        logger.debug(f"Normality test: p={p_value:.4f}, normal={is_normal}")
        return float(stat), float(p_value), is_normal
    
    def outlier_detection(
        self,
        values: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers in data.
        
        Args:
            values: Data values
            method: "iqr" or "zscore"
            threshold: Threshold for outlier detection
            
        Returns:
            (outliers_mask, outlier_values)
        """
        if method == "iqr":
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (values < lower) | (values > upper)
            
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(values))
            mask = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Found {np.sum(mask)} outliers using {method} method")
        return mask, values[mask]
    
    def trend_analysis(
        self,
        values: List[float],
        window_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze trend in values.
        
        Args:
            values: Time-series values
            window_size: Moving average window size
            
        Returns:
            Trend analysis dict
        """
        values_array = np.array(values)
        
        # Moving average
        ma = np.convolve(
            values_array,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        
        # Trend direction (linear regression)
        x = np.arange(len(values_array))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
        
        # Trend classification
        if abs(slope) < std_err:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'trend': trend,
            'moving_average': ma.tolist(),
        }
    
    def correlation_analysis(self, data: Dict[str, List[float]]) -> np.ndarray:
        """
        Compute correlation matrix between metrics.
        
        Args:
            data: Dict of {metric: [values]}
            
        Returns:
            Correlation matrix
        """
        metrics = list(data.keys())
        values_array = np.array([data[m] for m in metrics])
        
        corr_matrix = np.corrcoef(values_array)
        
        logger.debug(f"Computed {len(metrics)}x{len(metrics)} correlation matrix")
        return corr_matrix
    
    def distribution_test(
        self,
        values: np.ndarray,
        expected_dist: str = "normal",
    ) -> Dict[str, Any]:
        """
        Test data distribution.
        
        Args:
            values: Data values
            expected_dist: Expected distribution type
            
        Returns:
            Test results dict
        """
        if expected_dist == "normal":
            stat, p_value = stats.shapiro(values)
            test_name = "Shapiro-Wilk"
            
        elif expected_dist == "uniform":
            stat, p_value = stats.kstest(values, 'uniform')
            test_name = "Kolmogorov-Smirnov"
            
        else:
            raise ValueError(f"Unknown distribution: {expected_dist}")
        
        is_distribution = p_value > self.alpha
        
        return {
            'test': test_name,
            'statistic': float(stat),
            'p_value': float(p_value),
            'matches_distribution': is_distribution,
        }
    
    def variance_analysis(
        self,
        groups: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Perform ANOVA test on groups.
        
        Args:
            groups: Dict of {group_name: [values]}
            
        Returns:
            ANOVA results
        """
        values_list = [np.array(v) for v in groups.values()]
        f_stat, p_value = stats.f_oneway(*values_list)
        
        # Effect size (eta squared)
        grand_mean = np.mean(np.concatenate(values_list))
        ss_between = sum(
            len(v) * (np.mean(v) - grand_mean) ** 2
            for v in values_list
        )
        ss_total = sum(
            np.sum((v - grand_mean) ** 2)
            for v in values_list
        )
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'eta_squared': float(eta_squared),
            'significant': p_value < self.alpha,
        }
    
    def export_analysis(self, filepath: str) -> None:
        """
        Export analysis results to JSON.
        
        Args:
            filepath: Output file path
        """
        export_data = {}
        
        for metric, result in self.analyses.items():
            export_data[metric] = {
                'mean': result.mean,
                'median': result.median,
                'std': result.std,
                'variance': result.variance,
                'min': result.min_val,
                'max': result.max_val,
                'q25': result.q25,
                'q75': result.q75,
                'skewness': result.skewness,
                'kurtosis': result.kurtosis,
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported analysis to {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        if not self.analyses:
            return {}
        
        return {
            'metrics_analyzed': len(self.analyses),
            'metrics': {
                metric: {
                    'mean': result.mean,
                    'std': result.std,
                    'range': (result.min_val, result.max_val),
                }
                for metric, result in self.analyses.items()
            }
        }
