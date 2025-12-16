"""
Statistical Analysis Module for Hybrid-GCS Evaluation

Provides comprehensive statistical tools for analyzing RL training results,
including descriptive statistics, hypothesis testing, and trend analysis.

Author: Hybrid-GCS Team
Date: 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy import stats
from scipy.stats import t, f_oneway, mannwhitneyu, wilcoxon
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = field(default=(0.0, 0.0))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of results."""
        sig_str = "✓ Significant" if self.significant else "✗ Not significant"
        return (
            f"{self.test_name}:\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f} ({sig_str})\n"
            f"  Effect Size: {self.effect_size:.4f}\n"
            f"  CI: {self.confidence_interval}"
        )


@dataclass
class TrendAnalysis:
    """Container for trend analysis results."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    trend_direction: str  # "increasing", "decreasing", or "stable"
    confidence_interval: Tuple[float, float]
    
    def __str__(self) -> str:
        """String representation of trend."""
        return (
            f"Trend Analysis:\n"
            f"  Direction: {self.trend_direction}\n"
            f"  Slope: {self.slope:.6f}\n"
            f"  R²: {self.r_squared:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  95% CI: {self.confidence_interval}"
        )


class StatisticalAnalysis:
    """
    Comprehensive statistical analysis toolkit for RL evaluation.
    
    Features:
    - Descriptive statistics (mean, std, CI)
    - Parametric tests (t-test, ANOVA)
    - Non-parametric tests (Mann-Whitney U, Wilcoxon)
    - Effect size calculations (Cohen's d, Cohen's w)
    - Trend analysis and linear regression
    - Distribution analysis
    - Outlier detection
    
    Example:
        >>> analysis = StatisticalAnalysis()
        >>> data = np.random.randn(100)
        >>> stats_result = analysis.describe(data)
        >>> print(stats_result)
    """
    
    def __init__(self, alpha: float = 0.05, confidence: float = 0.95):
        """
        Initialize statistical analysis engine.
        
        Args:
            alpha: Significance level for hypothesis tests
            confidence: Confidence level for confidence intervals (0.0-1.0)
        """
        self.alpha = alpha
        self.confidence = confidence
        self.results_cache: Dict[str, Any] = {}
        logger.info(f"StatisticalAnalysis initialized (α={alpha}, CI={confidence:.1%})")
    
    # ==================== DESCRIPTIVE STATISTICS ====================
    
    def describe(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive descriptive statistics.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with statistics
        """
        data = np.asarray(data, dtype=float)
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) == 0:
            logger.warning("No valid data for description")
            return {}
        
        ci_low, ci_high = self.confidence_interval(data_clean)
        
        stats_dict = {
            "count": len(data_clean),
            "mean": float(np.mean(data_clean)),
            "std": float(np.std(data_clean, ddof=1)) if len(data_clean) > 1 else 0.0,
            "min": float(np.min(data_clean)),
            "q25": float(np.percentile(data_clean, 25)),
            "median": float(np.median(data_clean)),
            "q75": float(np.percentile(data_clean, 75)),
            "max": float(np.max(data_clean)),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "sem": float(stats.sem(data_clean)),  # Standard error of mean
            "skewness": float(stats.skew(data_clean)),
            "kurtosis": float(stats.kurtosis(data_clean)),
        }
        
        return stats_dict
    
    def confidence_interval(
        self,
        data: np.ndarray,
        confidence: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for data mean.
        
        Args:
            data: Input data array
            confidence: Confidence level (default: self.confidence)
            
        Returns:
            (lower, upper) bounds of confidence interval
        """
        if confidence is None:
            confidence = self.confidence
            
        data = np.asarray(data, dtype=float)
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 2:
            return (data_clean[0], data_clean[0]) if len(data_clean) == 1 else (0.0, 0.0)
        
        n = len(data_clean)
        mean = np.mean(data_clean)
        std = np.std(data_clean, ddof=1)
        
        # t-distribution based CI
        t_critical = t.ppf((1 + confidence) / 2, n - 1)
        margin = t_critical * (std / np.sqrt(n))
        
        return (mean - margin, mean + margin)
    
    # ==================== HYPOTHESIS TESTING ====================
    
    def ttest_independent(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        equal_var: bool = True
    ) -> StatisticalResult:
        """
        Independent samples t-test.
        
        Tests if two independent samples have equal means.
        
        Args:
            data1: First sample
            data2: Second sample
            equal_var: Assume equal variances
            
        Returns:
            StatisticalResult object
        """
        data1 = np.asarray(data1, dtype=float).flatten()
        data2 = np.asarray(data2, dtype=float).flatten()
        
        # Remove NaN values
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Cohen's d effect size
        cohen_d = self._cohens_d(data1, data2)
        
        # Confidence interval for difference of means
        mean_diff = np.mean(data1) - np.mean(data2)
        ci_low, ci_high = self._ci_mean_difference(data1, data2)
        
        significant = p_value < self.alpha
        
        return StatisticalResult(
            test_name="Independent t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            effect_size=cohen_d,
            confidence_interval=(ci_low, ci_high),
            metadata={
                "mean_diff": mean_diff,
                "n1": len(data1),
                "n2": len(data2),
                "equal_var": equal_var,
            }
        )
    
    def ttest_paired(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalResult:
        """
        Paired samples t-test.
        
        Tests if paired samples have equal means.
        
        Args:
            data1: First sample
            data2: Second sample (same length as data1)
            
        Returns:
            StatisticalResult object
        """
        data1 = np.asarray(data1, dtype=float).flatten()
        data2 = np.asarray(data2, dtype=float).flatten()
        
        # Remove pairs with NaN
        mask = ~(np.isnan(data1) | np.isnan(data2))
        data1 = data1[mask]
        data2 = data2[mask]
        
        if len(data1) < 2:
            logger.warning("Insufficient data for paired t-test")
            return StatisticalResult(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False
            )
        
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        # Cohen's d for paired data
        cohen_d = self._cohens_d_paired(data1, data2)
        
        mean_diff = np.mean(data1 - data2)
        ci_low, ci_high = self.confidence_interval(data1 - data2)
        
        significant = p_value < self.alpha
        
        return StatisticalResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            effect_size=cohen_d,
            confidence_interval=(ci_low, ci_high),
            metadata={
                "mean_diff": mean_diff,
                "n": len(data1),
            }
        )
    
    def anova(self, *groups: np.ndarray) -> StatisticalResult:
        """
        One-way ANOVA (Analysis of Variance).
        
        Tests if multiple groups have equal means.
        
        Args:
            *groups: Variable number of sample arrays
            
        Returns:
            StatisticalResult object
        """
        # Clean data
        clean_groups = []
        for group in groups:
            group = np.asarray(group, dtype=float).flatten()
            group = group[~np.isnan(group)]
            if len(group) > 0:
                clean_groups.append(group)
        
        if len(clean_groups) < 2:
            logger.warning("ANOVA requires at least 2 groups")
            return StatisticalResult(
                test_name="One-way ANOVA",
                statistic=0.0,
                p_value=1.0,
                significant=False
            )
        
        f_stat, p_value = f_oneway(*clean_groups)
        
        # Effect size: eta-squared
        eta_squared = self._eta_squared(*clean_groups)
        
        significant = p_value < self.alpha
        
        return StatisticalResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            significant=significant,
            effect_size=eta_squared,
            metadata={
                "k": len(clean_groups),  # number of groups
                "sample_sizes": [len(g) for g in clean_groups],
            }
        )
    
    def mannwhitneyu_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> StatisticalResult:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            data1: First sample
            data2: Second sample
            
        Returns:
            StatisticalResult object
        """
        data1 = np.asarray(data1, dtype=float).flatten()
        data2 = np.asarray(data2, dtype=float).flatten()
        
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        if len(data1) == 0 or len(data2) == 0:
            logger.warning("Insufficient data for Mann-Whitney U test")
            return StatisticalResult(
                test_name="Mann-Whitney U test",
                statistic=0.0,
                p_value=1.0,
                significant=False
            )
        
        u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Effect size: rank-biserial correlation
        r_rb = self._rank_biserial(data1, data2)
        
        significant = p_value < self.alpha
        
        return StatisticalResult(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=p_value,
            significant=significant,
            effect_size=r_rb,
            metadata={
                "n1": len(data1),
                "n2": len(data2),
            }
        )
    
    def wilcoxon_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalResult:
        """
        Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            data1: First sample
            data2: Second sample (paired with data1)
            
        Returns:
            StatisticalResult object
        """
        data1 = np.asarray(data1, dtype=float).flatten()
        data2 = np.asarray(data2, dtype=float).flatten()
        
        # Remove pairs with NaN
        mask = ~(np.isnan(data1) | np.isnan(data2))
        data1 = data1[mask]
        data2 = data2[mask]
        
        if len(data1) < 2:
            logger.warning("Insufficient data for Wilcoxon test")
            return StatisticalResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                significant=False
            )
        
        w_stat, p_value = wilcoxon(data1, data2)
        
        # Effect size: rank-biserial
        r_rb = self._rank_biserial_paired(data1, data2)
        
        significant = p_value < self.alpha
        
        return StatisticalResult(
            test_name="Wilcoxon signed-rank test",
            statistic=w_stat,
            p_value=p_value,
            significant=significant,
            effect_size=r_rb,
            metadata={
                "n": len(data1),
            }
        )
    
    # ==================== TREND ANALYSIS ====================
    
    def linear_regression(self, x: np.ndarray, y: np.ndarray) -> TrendAnalysis:
        """
        Perform linear regression analysis.
        
        Args:
            x: Independent variable (e.g., time/episodes)
            y: Dependent variable (e.g., reward)
            
        Returns:
            TrendAnalysis object
        """
        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        
        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 3:
            logger.warning("Insufficient data for linear regression")
            return TrendAnalysis(
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                p_value=1.0,
                std_err=0.0,
                trend_direction="insufficient_data",
                confidence_interval=(0.0, 0.0),
            )
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Trend direction
        if abs(slope) < 1e-6:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # 95% CI for slope
        n = len(x)
        t_crit = t.ppf(0.975, n - 2)
        margin = t_crit * std_err
        ci_lower = slope - margin
        ci_upper = slope + margin
        
        return TrendAnalysis(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            std_err=std_err,
            trend_direction=trend_direction,
            confidence_interval=(ci_lower, ci_upper),
        )
    
    def trend_over_time(self, data: np.ndarray) -> TrendAnalysis:
        """
        Analyze trend in data over sequential time points.
        
        Args:
            data: Time series data
            
        Returns:
            TrendAnalysis object
        """
        data = np.asarray(data, dtype=float).flatten()
        x = np.arange(len(data))
        return self.linear_regression(x, data)
    
    # ==================== OUTLIER DETECTION ====================
    
    def detect_outliers_iqr(
        self,
        data: np.ndarray,
        k: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            data: Input data
            k: IQR multiplier (default: 1.5)
            
        Returns:
            Dictionary with outlier info
        """
        data = np.asarray(data, dtype=float).flatten()
        data_clean = data[~np.isnan(data)]
        
        q1 = np.percentile(data_clean, 25)
        q3 = np.percentile(data_clean, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        outliers = (data_clean < lower_bound) | (data_clean > upper_bound)
        outlier_indices = np.where(outliers)[0]
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_count": np.sum(outliers),
            "outlier_percentage": 100 * np.sum(outliers) / len(data_clean),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_values": data_clean[outliers].tolist(),
        }
    
    def detect_outliers_zscore(
        self,
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Input data
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Dictionary with outlier info
        """
        data = np.asarray(data, dtype=float).flatten()
        data_clean = data[~np.isnan(data)]
        
        z_scores = np.abs(stats.zscore(data_clean))
        outliers = z_scores > threshold
        outlier_indices = np.where(outliers)[0]
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_count": np.sum(outliers),
            "outlier_percentage": 100 * np.sum(outliers) / len(data_clean),
            "threshold": threshold,
            "z_scores": z_scores.tolist(),
            "outlier_values": data_clean[outliers].tolist(),
        }
    
    # ==================== NORMALITY TESTING ====================
    
    def shapiro_wilk_test(self, data: np.ndarray) -> StatisticalResult:
        """
        Shapiro-Wilk test for normality.
        
        Args:
            data: Input data
            
        Returns:
            StatisticalResult object
        """
        data = np.asarray(data, dtype=float).flatten()
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            logger.warning("Shapiro-Wilk requires at least 3 samples")
            return StatisticalResult(
                test_name="Shapiro-Wilk test",
                statistic=0.0,
                p_value=1.0,
                significant=False
            )
        
        w_stat, p_value = stats.shapiro(data_clean)
        significant = p_value < self.alpha  # Reject normality
        
        return StatisticalResult(
            test_name="Shapiro-Wilk test",
            statistic=w_stat,
            p_value=p_value,
            significant=significant,  # True = non-normal
            metadata={
                "n": len(data_clean),
                "interpretation": "Data is NON-normal" if significant else "Data is normal",
            }
        )
    
    def anderson_darling_test(self, data: np.ndarray) -> StatisticalResult:
        """
        Anderson-Darling test for normality.
        
        Args:
            data: Input data
            
        Returns:
            StatisticalResult object
        """
        data = np.asarray(data, dtype=float).flatten()
        data_clean = data[~np.isnan(data)]
        
        result = stats.anderson(data_clean, dist='norm')
        
        # Interpret at 5% significance level (index 2)
        critical_value = result.critical_values[2]
        p_value = result.significance_level[2] / 100.0
        significant = result.statistic > critical_value
        
        return StatisticalResult(
            test_name="Anderson-Darling test",
            statistic=result.statistic,
            p_value=p_value,
            significant=significant,
            metadata={
                "critical_value": critical_value,
                "interpretation": "Data is NON-normal" if significant else "Data is normal",
            }
        )
    
    # ==================== EFFECT SIZE CALCULATIONS ====================
    
    def _cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for independent samples."""
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        if n1 + n2 <= 2:
            return 0.0
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        return float(cohens_d)
    
    def _cohens_d_paired(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d for paired samples."""
        diff = data1 - data2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        if std_diff == 0:
            return 0.0
        
        cohens_d = mean_diff / std_diff
        return float(cohens_d)
    
    def _eta_squared(self, *groups: np.ndarray) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        eta_sq = ss_between / ss_total
        return float(eta_sq)
    
    def _rank_biserial(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate rank-biserial correlation (effect size for Mann-Whitney U)."""
        n1, n2 = len(data1), len(data2)
        combined = np.concatenate([data1, data2])
        ranks = stats.rankdata(combined)
        
        r1 = np.sum(ranks[:n1])
        u_stat, _ = mannwhitneyu(data1, data2, alternative='two-sided')
        
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        return float(r_rb)
    
    def _rank_biserial_paired(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate effect size for Wilcoxon test."""
        diff = data1 - data2
        n = len(diff[diff != 0])
        
        if n == 0:
            return 0.0
        
        w_stat, _ = wilcoxon(data1, data2)
        r = 1 - (2 * w_stat) / (n * (n + 1))
        return float(r)
    
    def _ci_mean_difference(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate CI for difference of means (independent samples)."""
        n1, n2 = len(data1), len(data2)
        mean_diff = np.mean(data1) - np.mean(data2)
        
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        
        se_diff = np.sqrt(var1 / n1 + var2 / n2)
        
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        t_crit = t.ppf(0.975, df)
        
        margin = t_crit * se_diff
        return (mean_diff - margin, mean_diff + margin)
    
    # ==================== UTILITY METHODS ====================
    
    def compare_methods(
        self,
        method1_data: np.ndarray,
        method2_data: np.ndarray,
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        parametric: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between two methods.
        
        Args:
            method1_data: Results from method 1
            method2_data: Results from method 2
            method1_name: Name of method 1
            method2_name: Name of method 2
            parametric: Use parametric tests if True
            
        Returns:
            Dictionary with comparison results
        """
        # Descriptive statistics
        desc1 = self.describe(method1_data)
        desc2 = self.describe(method2_data)
        
        # Normality tests
        normality1 = self.shapiro_wilk_test(method1_data)
        normality2 = self.shapiro_wilk_test(method2_data)
        
        # Choose appropriate test
        if parametric and not normality1.significant and not normality2.significant:
            comparison = self.ttest_independent(method1_data, method2_data)
        else:
            comparison = self.mannwhitneyu_test(method1_data, method2_data)
        
        return {
            "method1_name": method1_name,
            "method2_name": method2_name,
            "method1_stats": desc1,
            "method2_stats": desc2,
            "normality_test1": {
                "test": normality1.test_name,
                "p_value": normality1.p_value,
                "is_normal": not normality1.significant,
            },
            "normality_test2": {
                "test": normality2.test_name,
                "p_value": normality2.p_value,
                "is_normal": not normality2.significant,
            },
            "comparison": {
                "test": comparison.test_name,
                "statistic": comparison.statistic,
                "p_value": comparison.p_value,
                "significant": comparison.significant,
                "effect_size": comparison.effect_size,
                "ci": comparison.confidence_interval,
            },
            "interpretation": (
                f"{method1_name} is significantly better than {method2_name}"
                if desc1.get("mean", 0) > desc2.get("mean", 0) and comparison.significant
                else f"{method2_name} is significantly better than {method1_name}"
                if comparison.significant
                else "No significant difference"
            ),
        }
    
    def summary_report(self, data: np.ndarray, name: str = "Data") -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            data: Input data
            name: Name of the dataset
            
        Returns:
            Formatted report string
        """
        desc = self.describe(data)
        normality = self.shapiro_wilk_test(data)
        trend = self.trend_over_time(data)
        outliers = self.detect_outliers_iqr(data)
        
        report = f"""
╔════════════════════════════════════════════════╗
║  Statistical Summary Report: {name:30s}║
╚════════════════════════════════════════════════╝

📊 DESCRIPTIVE STATISTICS
───────────────────────────────────────────────
  Mean:        {desc.get('mean', 0):.4f}
  Std Dev:     {desc.get('std', 0):.4f}
  Median:      {desc.get('median', 0):.4f}
  Range:       [{desc.get('min', 0):.4f}, {desc.get('max', 0):.4f}]
  IQR:         [{desc.get('q25', 0):.4f}, {desc.get('q75', 0):.4f}]
  Skewness:    {desc.get('skewness', 0):.4f}
  Kurtosis:    {desc.get('kurtosis', 0):.4f}

🎯 CONFIDENCE INTERVAL (95%)
───────────────────────────────────────────────
  [{desc.get('ci_lower', 0):.4f}, {desc.get('ci_upper', 0):.4f}]

📈 TREND ANALYSIS
───────────────────────────────────────────────
  Direction:   {trend.trend_direction}
  Slope:       {trend.slope:.6f}
  R²:          {trend.r_squared:.4f}
  p-value:     {trend.p_value:.4f}

✔️  NORMALITY TEST (Shapiro-Wilk)
───────────────────────────────────────────────
  Test Stat:   {normality.statistic:.4f}
  p-value:     {normality.p_value:.4f}
  Result:      {'Non-normal' if normality.significant else 'Normal'}

⚠️  OUTLIERS (IQR method)
───────────────────────────────────────────────
  Count:       {outliers['outlier_count']} ({outliers['outlier_percentage']:.1f}%)
  Bounds:      [{outliers['lower_bound']:.4f}, {outliers['upper_bound']:.4f}]

═════════════════════════════════════════════════
"""
        return report


# ==================== MODULE EXPORTS ====================

__all__ = [
    "StatisticalAnalysis",
    "StatisticalResult",
    "TrendAnalysis",
]
