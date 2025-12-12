"""
Benchmarking Script
File: scripts/benchmark.py

Comprehensive benchmarking for Hybrid-GCS.
"""

import logging
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    )


def benchmark_gcs_decomposition() -> Dict[str, float]:
    """Benchmark GCS decomposition."""
    logger.info("Benchmarking GCS decomposition...")
    
    from hybrid_gcs.core import GCSDecomposer, ConfigSpace
    import numpy as np
    
    cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
    decomposer = GCSDecomposer(config_space=cs, max_regions=10)
    
    times = []
    for _ in range(10):
        start = np.array([-0.5] * 6)
        goal = np.array([0.5] * 6)
        obstacles = [(np.array([0, 0, 0, 0, 0, 0]), 0.1)]
        
        start_time = time.time()
        result = decomposer.decompose(start, goal, obstacles)
        times.append(time.time() - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
    }


def benchmark_environment_step() -> Dict[str, float]:
    """Benchmark environment step."""
    logger.info("Benchmarking environment step...")
    
    from hybrid_gcs.environments import ManipulationEnvironment
    
    env = ManipulationEnvironment(state_dim=20, action_dim=6)
    obs = env.reset()
    
    times = []
    for _ in range(1000):
        action = np.random.randn(6) * 0.1
        start_time = time.time()
        obs, reward, done, info = env.step(action)
        times.append(time.time() - start_time)
        
        if done:
            obs = env.reset()
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'frequency_hz': 1.0 / np.mean(times),
    }


def benchmark_training_step() -> Dict[str, float]:
    """Benchmark training step."""
    logger.info("Benchmarking training step...")
    
    from hybrid_gcs.training import OptimizedTrainer
    
    trainer = OptimizedTrainer(
        policy_dim=20,
        action_dim=6,
        batch_size=32
    )
    
    times = []
    for _ in range(100):
        # Simulate training step
        start_time = time.time()
        # Mock training operation
        time.sleep(0.001)  # Placeholder
        times.append(time.time() - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
    }


def run_benchmarks() -> Dict[str, Dict[str, float]]:
    """Run all benchmarks."""
    logger.info("Starting comprehensive benchmarking...")
    
    results = {
        'gcs_decomposition': benchmark_gcs_decomposition(),
        'environment_step': benchmark_environment_step(),
        'training_step': benchmark_training_step(),
    }
    
    return results


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    for component, metrics in results.items():
        print(f"\n{component.upper()}:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            if 'frequency' in metric_name or 'hz' in metric_name:
                print(f"  {metric_name:20s}: {value:10.2f} Hz")
            elif 'time' in metric_name:
                print(f"  {metric_name:20s}: {value*1000:10.4f} ms")
            else:
                print(f"  {metric_name:20s}: {value:10.6f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Hybrid-GCS")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    results = run_benchmarks()
    print_results(results)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
