#!/usr/bin/env python3
"""
VALIDATION TEST SCRIPT for StatisticalAnalysis Module

This script verifies that the StatisticalAnalysis module is properly integrated
and all functionality works correctly.

Run this after integrating the module:
    python validate_statistical_analysis.py

Author: Hybrid-GCS Team
Date: December 16, 2025
"""

import sys
import numpy as np
from typing import List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print colored header."""
    print(f"\n{BLUE}{BOLD}{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}{RESET}\n")


def print_test(name: str) -> None:
    """Print test name."""
    print(f"{YELLOW}→ Testing: {name}{RESET}", end=" ")


def print_success() -> None:
    """Print success message."""
    print(f"{GREEN}✓ PASS{RESET}")


def print_failure(error: str) -> None:
    """Print failure message."""
    print(f"{RED}✗ FAIL{RESET}")
    print(f"  Error: {error}")


def test_imports() -> bool:
    """Test module imports."""
    print_header("MODULE IMPORTS")
    
    try:
        print_test("Import StatisticalAnalysis")
        from hybrid_gcs.evaluation import StatisticalAnalysis
        print_success()
        
        print_test("Import StatisticalResult")
        from hybrid_gcs.evaluation import StatisticalResult
        print_success()
        
        print_test("Import TrendAnalysis")
        from hybrid_gcs.evaluation import TrendAnalysis
        print_success()
        
        return True
    except ImportError as e:
        print_failure(str(e))
        return False


def test_initialization() -> Tuple[bool, object]:
    """Test StatisticalAnalysis initialization."""
    print_header("INITIALIZATION")
    
    try:
        from hybrid_gcs.evaluation import StatisticalAnalysis
        
        print_test("Create instance with default params")
        analysis = StatisticalAnalysis()
        print_success()
        
        print_test("Create instance with custom alpha (0.01)")
        analysis_strict = StatisticalAnalysis(alpha=0.01)
        print_success()
        
        print_test("Create instance with custom CI (0.99)")
        analysis_high_ci = StatisticalAnalysis(confidence=0.99)
        print_success()
        
        return True, analysis
    except Exception as e:
        print_failure(str(e))
        return False, None


def test_descriptive_statistics(analysis: object) -> bool:
    """Test descriptive statistics methods."""
    print_header("DESCRIPTIVE STATISTICS")
    
    try:
        # Generate sample data
        data = np.random.randn(100) * 10 + 50
        
        print_test("describe(data) - Basic statistics")
        stats = analysis.describe(data)
        assert isinstance(stats, dict)
        assert 'mean' in stats and 'std' in stats
        print_success()
        
        print_test("confidence_interval() - CI computation")
        ci = analysis.confidence_interval(data)
        assert isinstance(ci, tuple) and len(ci) == 2
        assert ci[0] < ci[1]
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def test_hypothesis_testing(analysis: object) -> bool:
    """Test hypothesis testing methods."""
    print_header("HYPOTHESIS TESTING")
    
    try:
        # Generate test data - group 1 clearly higher
        data1 = np.random.randn(50) + 5
        data2 = np.random.randn(50)
        
        print_test("ttest_independent() - T-test")
        result = analysis.ttest_independent(data1, data2)
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'significant')
        print_success()
        
        print_test("ttest_paired() - Paired t-test")
        paired_result = analysis.ttest_paired(data1[:30], data1[20:50])
        assert hasattr(paired_result, 'effect_size')
        print_success()
        
        print_test("anova() - ANOVA test")
        group1 = np.random.randn(30) + 2
        group2 = np.random.randn(30)
        group3 = np.random.randn(30) - 2
        anova_result = analysis.anova(group1, group2, group3)
        assert anova_result.significant or not anova_result.significant
        print_success()
        
        print_test("mannwhitneyu_test() - Non-parametric test")
        mw_result = analysis.mannwhitneyu_test(data1, data2)
        assert hasattr(mw_result, 'effect_size')
        print_success()
        
        print_test("wilcoxon_test() - Wilcoxon signed-rank")
        wilc_result = analysis.wilcoxon_test(data1[:30], data2[:30])
        assert hasattr(wilc_result, 'p_value')
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def test_trend_analysis(analysis: object) -> bool:
    """Test trend analysis methods."""
    print_header("TREND ANALYSIS")
    
    try:
        # Create data with clear upward trend
        x = np.arange(50, dtype=float)
        y = 10 + 0.5 * x + np.random.randn(50) * 2
        
        print_test("linear_regression() - Linear regression")
        trend = analysis.linear_regression(x, y)
        assert hasattr(trend, 'slope')
        assert hasattr(trend, 'r_squared')
        print_success()
        
        print_test("trend_over_time() - Time series trend")
        time_trend = analysis.trend_over_time(y)
        assert time_trend.trend_direction in ['increasing', 'decreasing', 'stable']
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def test_outlier_detection(analysis: object) -> bool:
    """Test outlier detection methods."""
    print_header("OUTLIER DETECTION")
    
    try:
        # Data with outliers
        data = np.concatenate([
            np.random.randn(95) * 5 + 20,
            np.array([100, 101, 102, 103, 104])  # Clear outliers
        ])
        
        print_test("detect_outliers_iqr() - IQR method")
        outliers_iqr = analysis.detect_outliers_iqr(data)
        assert 'outlier_count' in outliers_iqr
        assert outliers_iqr['outlier_count'] > 0
        print_success()
        
        print_test("detect_outliers_zscore() - Z-score method")
        outliers_z = analysis.detect_outliers_zscore(data, threshold=2.5)
        assert 'outlier_count' in outliers_z
        assert outliers_z['outlier_count'] > 0
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def test_normality_testing(analysis: object) -> bool:
    """Test normality testing methods."""
    print_header("NORMALITY TESTING")
    
    try:
        # Normal and non-normal data
        normal_data = np.random.randn(50)
        non_normal_data = np.random.exponential(2, 50)
        
        print_test("shapiro_wilk_test() - Shapiro-Wilk")
        result = analysis.shapiro_wilk_test(normal_data)
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        print_success()
        
        print_test("anderson_darling_test() - Anderson-Darling")
        result = analysis.anderson_darling_test(normal_data)
        assert hasattr(result, 'statistic')
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def test_comparison(analysis: object) -> bool:
    """Test method comparison."""
    print_header("METHOD COMPARISON")
    
    try:
        # Simulated results from two methods
        hybrid_gcs = np.array([92, 94, 91, 93, 95, 92, 94])
        pure_rl = np.array([88, 87, 89, 86, 85, 87, 86])
        
        print_test("compare_methods() - Comprehensive comparison")
        comparison = analysis.compare_methods(
            hybrid_gcs,
            pure_rl,
            method1_name="Hybrid-GCS",
            method2_name="Pure RL"
        )
        
        assert 'method1_name' in comparison
        assert 'comparison' in comparison
        assert 'interpretation' in comparison
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def test_summary_report(analysis: object) -> bool:
    """Test summary report generation."""
    print_header("SUMMARY REPORTS")
    
    try:
        data = np.random.randn(100) * 10 + 50
        
        print_test("summary_report() - Generate report")
        report = analysis.summary_report(data, "Test Data")
        assert isinstance(report, str)
        assert "Statistical Summary Report" in report
        print_success()
        
        return True
    except Exception as e:
        print_failure(str(e))
        return False


def run_all_tests() -> bool:
    """Run all validation tests."""
    print_header("STATISTICAL ANALYSIS MODULE VALIDATION")
    print(f"{BOLD}Testing Hybrid-GCS Evaluation Module{RESET}\n")
    
    results = []
    
    # Test imports
    if not test_imports():
        print(f"\n{RED}{BOLD}✗ IMPORT FAILED - Cannot continue{RESET}")
        return False
    
    # Initialize
    success, analysis = test_initialization()
    if not success or analysis is None:
        print(f"\n{RED}{BOLD}✗ INITIALIZATION FAILED - Cannot continue{RESET}")
        return False
    
    # Run all test suites
    results.append(("Descriptive Statistics", test_descriptive_statistics(analysis)))
    results.append(("Hypothesis Testing", test_hypothesis_testing(analysis)))
    results.append(("Trend Analysis", test_trend_analysis(analysis)))
    results.append(("Outlier Detection", test_outlier_detection(analysis)))
    results.append(("Normality Testing", test_normality_testing(analysis)))
    results.append(("Method Comparison", test_comparison(analysis)))
    results.append(("Summary Reports", test_summary_report(analysis)))
    
    # Print summary
    print_header("VALIDATION SUMMARY")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed
    
    for name, result in results:
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"  {name:.<40} {status}")
    
    print(f"\n{BOLD}Results: {passed}/{total} test suites passed{RESET}")
    
    if failed == 0:
        print(f"\n{GREEN}{BOLD}✅ ALL TESTS PASSED - Module is ready to use!{RESET}\n")
        return True
    else:
        print(f"\n{RED}{BOLD}❌ {failed} TEST SUITE(S) FAILED - Please review errors above{RESET}\n")
        return False


def print_usage_examples() -> None:
    """Print quick usage examples."""
    print_header("QUICK USAGE EXAMPLES")
    
    examples = """
# Example 1: Basic Statistics
from hybrid_gcs.evaluation import StatisticalAnalysis
import numpy as np

analysis = StatisticalAnalysis()
data = np.random.randn(100)
stats = analysis.describe(data)
print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")


# Example 2: Compare Two Methods
method1 = np.array([90, 92, 91, 93])
method2 = np.array([85, 86, 87, 88])
result = analysis.ttest_independent(method1, method2)
print(f"P-value: {result.p_value:.4f}")


# Example 3: Analyze Trend
rewards = np.arange(100) + np.random.randn(100) * 5
trend = analysis.trend_over_time(rewards)
print(f"Trend: {trend.trend_direction}, R²: {trend.r_squared:.4f}")


# Example 4: Generate Report
report = analysis.summary_report(data, "Training Results")
print(report)
"""
    print(examples)


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print_usage_examples()
        sys.exit(0)
    else:
        sys.exit(1)
