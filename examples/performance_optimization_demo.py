"""
Performance optimization demonstration for regional monetary policy analysis.

This script demonstrates the performance improvements achieved through:
1. Profiling and bottleneck identification
2. Parallel processing for regional parameter estimation
3. Memory management for large datasets
4. Intelligent caching of API responses and computation results
5. System monitoring and alerting
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from regional_monetary_policy.performance import (
    PerformanceProfiler, ComputationOptimizer, MemoryManager,
    IntelligentCacheManager, SystemMonitor, LogAlertHandler
)
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.econometric.models import EstimationConfig


def generate_synthetic_dataset(n_regions: int = 10, n_periods: int = 120) -> RegionalDataset:
    """Generate synthetic regional dataset for benchmarking."""
    
    logger.info(f"Generating synthetic dataset: {n_regions} regions, {n_periods} periods")
    
    regions = [f"region_{i:02d}" for i in range(n_regions)]
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="M")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate correlated regional data
    # Create correlation structure
    correlation_matrix = np.eye(n_regions)
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                distance = abs(i - j)
                correlation_matrix[i, j] = np.exp(-distance / 3.0) * 0.3
    
    # Generate output gaps with spatial correlation
    innovations = np.random.multivariate_normal(
        np.zeros(n_regions), 
        correlation_matrix * 0.01,
        size=n_periods
    )
    
    output_gaps = pd.DataFrame(
        innovations,
        index=dates,
        columns=regions
    )
    
    # Generate inflation rates with persistence
    inflation_innovations = np.random.multivariate_normal(
        np.zeros(n_regions),
        correlation_matrix * 0.005,
        size=n_periods
    )
    
    inflation_rates = pd.DataFrame(
        index=dates,
        columns=regions
    )
    
    # Add persistence to inflation
    for i, region in enumerate(regions):
        inflation_series = np.zeros(n_periods)
        inflation_series[0] = 0.02 + inflation_innovations[0, i]
        
        for t in range(1, n_periods):
            inflation_series[t] = (0.7 * inflation_series[t-1] + 
                                 0.3 * 0.02 + 
                                 inflation_innovations[t, i])
        
        inflation_rates[region] = inflation_series
    
    # Generate interest rates (Taylor rule)
    aggregate_inflation = inflation_rates.mean(axis=1)
    aggregate_output = output_gaps.mean(axis=1)
    
    interest_rates = (0.02 + 
                     1.5 * (aggregate_inflation - 0.02) + 
                     0.5 * aggregate_output + 
                     np.random.normal(0, 0.002, n_periods))
    
    interest_rates = pd.Series(interest_rates, index=dates)
    
    return RegionalDataset(
        output_gaps=output_gaps,
        inflation_rates=inflation_rates,
        interest_rates=interest_rates,
        real_time_estimates={},
        metadata={
            "source": "synthetic",
            "n_regions": n_regions,
            "n_periods": n_periods,
            "generated_at": pd.Timestamp.now()
        }
    )


def benchmark_standard_estimation(dataset: RegionalDataset) -> dict:
    """Benchmark standard estimation without optimization."""
    
    logger.info("Running standard estimation benchmark")
    
    regions = list(dataset.output_gaps.columns)
    spatial_handler = SpatialModelHandler(regions)
    config = EstimationConfig()
    
    # Create estimator without optimization
    estimator = ParameterEstimator(
        spatial_handler, 
        config, 
        enable_performance_optimization=False
    )
    
    start_time = time.time()
    
    try:
        # Run Stage 1 and 2 only for benchmarking
        spatial_results = estimator.estimate_stage_one(dataset)
        regional_params = estimator.estimate_stage_two(dataset, spatial_results.weight_matrix)
        
        end_time = time.time()
        
        return {
            "success": True,
            "execution_time": end_time - start_time,
            "n_regions": len(regions),
            "memory_usage": "N/A",
            "method": "standard"
        }
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"Standard estimation failed: {e}")
        
        return {
            "success": False,
            "execution_time": end_time - start_time,
            "error": str(e),
            "method": "standard"
        }


def benchmark_optimized_estimation(dataset: RegionalDataset) -> dict:
    """Benchmark optimized estimation with performance features."""
    
    logger.info("Running optimized estimation benchmark")
    
    regions = list(dataset.output_gaps.columns)
    spatial_handler = SpatialModelHandler(regions)
    config = EstimationConfig()
    
    # Create estimator with optimization
    estimator = ParameterEstimizer(
        spatial_handler, 
        config, 
        enable_performance_optimization=True
    )
    
    # Setup system monitoring
    estimator.system_monitor.add_alert_handler(LogAlertHandler())
    
    start_time = time.time()
    memory_start = estimator.memory_manager.get_memory_stats()
    
    try:
        # Run optimized estimation
        spatial_results = estimator.estimate_stage_one(dataset)
        
        # Use parallel estimation for Stage 2
        regional_params = estimator.estimate_stage_two_parallel(dataset, spatial_results.weight_matrix)
        
        end_time = time.time()
        memory_end = estimator.memory_manager.get_memory_stats()
        
        # Get performance report
        performance_report = estimator.get_performance_report()
        
        # Cleanup
        estimator.cleanup_performance_resources()
        
        return {
            "success": True,
            "execution_time": end_time - start_time,
            "n_regions": len(regions),
            "memory_usage": memory_end.process_memory - memory_start.process_memory,
            "performance_report": performance_report,
            "method": "optimized"
        }
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"Optimized estimation failed: {e}")
        
        # Cleanup even on failure
        estimator.cleanup_performance_resources()
        
        return {
            "success": False,
            "execution_time": end_time - start_time,
            "error": str(e),
            "method": "optimized"
        }


def demonstrate_caching_benefits():
    """Demonstrate caching performance benefits."""
    
    logger.info("Demonstrating caching benefits")
    
    cache_manager = IntelligentCacheManager(
        cache_dir="data/cache/demo",
        max_cache_size_gb=1.0
    )
    
    # Simulate expensive computation
    def expensive_computation(n: int) -> np.ndarray:
        """Simulate expensive matrix computation."""
        time.sleep(0.1)  # Simulate computation time
        return np.random.rand(n, n)
    
    # Test without caching
    start_time = time.time()
    result1 = expensive_computation(100)
    no_cache_time = time.time() - start_time
    
    # Test with caching (first call)
    params = {"n": 100, "seed": 42}
    
    start_time = time.time()
    cached_result = cache_manager.get_cached_computation("expensive_comp", params)
    if cached_result is None:
        result2 = expensive_computation(100)
        cache_manager.cache_computation_result("expensive_comp", params, result2)
    else:
        result2 = cached_result
    first_cache_time = time.time() - start_time
    
    # Test with caching (second call - should be fast)
    start_time = time.time()
    cached_result = cache_manager.get_cached_computation("expensive_comp", params)
    second_cache_time = time.time() - start_time
    
    logger.info(f"No cache: {no_cache_time:.3f}s")
    logger.info(f"First cache call: {first_cache_time:.3f}s")
    logger.info(f"Second cache call: {second_cache_time:.3f}s")
    logger.info(f"Cache speedup: {no_cache_time / second_cache_time:.1f}x")
    
    # Get cache statistics
    cache_stats = cache_manager.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")
    
    return {
        "no_cache_time": no_cache_time,
        "first_cache_time": first_cache_time,
        "second_cache_time": second_cache_time,
        "speedup": no_cache_time / second_cache_time if second_cache_time > 0 else 0,
        "cache_stats": cache_stats
    }


def demonstrate_memory_optimization():
    """Demonstrate memory optimization features."""
    
    logger.info("Demonstrating memory optimization")
    
    memory_manager = MemoryManager(memory_limit_gb=4.0)
    
    # Create large DataFrame with suboptimal types
    n_rows = 100000
    df = pd.DataFrame({
        'id': np.arange(n_rows, dtype='int64'),
        'value': np.random.rand(n_rows).astype('float64'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'flag': np.random.choice([True, False], n_rows)
    })
    
    original_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
    logger.info(f"Original DataFrame memory: {original_memory:.1f} MB")
    
    # Optimize DataFrame
    with memory_manager.memory_context("dataframe_optimization"):
        optimized_df = memory_manager.optimize_dataframe(df)
    
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024**2)  # MB
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    
    logger.info(f"Optimized DataFrame memory: {optimized_memory:.1f} MB")
    logger.info(f"Memory reduction: {memory_reduction:.1f}%")
    
    # Demonstrate chunked processing
    large_array = np.random.rand(10000, 500)
    
    def sum_operation(chunk):
        return np.sum(chunk, axis=1)
    
    with memory_manager.memory_context("chunked_processing"):
        chunked_result = memory_manager.chunk_array_processing(
            large_array, 
            sum_operation,
            chunk_size=2000
        )
    
    # Compare with direct processing
    direct_result = np.sum(large_array, axis=1)
    
    # Verify results are equivalent
    assert np.allclose(chunked_result, direct_result)
    logger.info("Chunked processing verification: PASSED")
    
    return {
        "original_memory_mb": original_memory,
        "optimized_memory_mb": optimized_memory,
        "memory_reduction_percent": memory_reduction,
        "chunked_processing_verified": True
    }


def run_scalability_benchmark():
    """Run scalability benchmark across different problem sizes."""
    
    logger.info("Running scalability benchmark")
    
    region_counts = [5, 10, 15, 20]
    results = []
    
    for n_regions in region_counts:
        logger.info(f"Benchmarking {n_regions} regions")
        
        # Generate dataset
        dataset = generate_synthetic_dataset(n_regions=n_regions, n_periods=60)
        
        # Benchmark standard method
        standard_result = benchmark_standard_estimation(dataset)
        
        # Benchmark optimized method
        optimized_result = benchmark_optimized_estimation(dataset)
        
        results.append({
            "n_regions": n_regions,
            "standard_time": standard_result.get("execution_time", 0),
            "optimized_time": optimized_result.get("execution_time", 0),
            "standard_success": standard_result.get("success", False),
            "optimized_success": optimized_result.get("success", False),
            "speedup": (standard_result.get("execution_time", 0) / 
                       optimized_result.get("execution_time", 1) 
                       if optimized_result.get("execution_time", 0) > 0 else 0)
        })
        
        logger.info(f"Results for {n_regions} regions:")
        logger.info(f"  Standard: {standard_result.get('execution_time', 0):.2f}s")
        logger.info(f"  Optimized: {optimized_result.get('execution_time', 0):.2f}s")
        logger.info(f"  Speedup: {results[-1]['speedup']:.2f}x")
    
    return results


def create_performance_plots(scalability_results: list, output_dir: str = "output"):
    """Create performance visualization plots."""
    
    logger.info("Creating performance plots")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract data for plotting
    n_regions = [r["n_regions"] for r in scalability_results]
    standard_times = [r["standard_time"] for r in scalability_results]
    optimized_times = [r["optimized_time"] for r in scalability_results]
    speedups = [r["speedup"] for r in scalability_results]
    
    # Create execution time comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_regions, standard_times, 'o-', label='Standard', linewidth=2, markersize=8)
    plt.plot(n_regions, optimized_times, 's-', label='Optimized', linewidth=2, markersize=8)
    plt.xlabel('Number of Regions')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Estimation Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(n_regions, speedups, 'o-', color='green', linewidth=2, markersize=8)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
    plt.xlabel('Number of Regions')
    plt.ylabel('Speedup Factor')
    plt.title('Performance Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plots saved to {output_dir}/performance_comparison.png")


def main():
    """Run complete performance optimization demonstration."""
    
    logger.info("Starting performance optimization demonstration")
    
    # 1. Demonstrate caching benefits
    logger.info("\n" + "="*50)
    logger.info("1. CACHING DEMONSTRATION")
    logger.info("="*50)
    caching_results = demonstrate_caching_benefits()
    
    # 2. Demonstrate memory optimization
    logger.info("\n" + "="*50)
    logger.info("2. MEMORY OPTIMIZATION DEMONSTRATION")
    logger.info("="*50)
    memory_results = demonstrate_memory_optimization()
    
    # 3. Run scalability benchmark
    logger.info("\n" + "="*50)
    logger.info("3. SCALABILITY BENCHMARK")
    logger.info("="*50)
    scalability_results = run_scalability_benchmark()
    
    # 4. Create performance plots
    logger.info("\n" + "="*50)
    logger.info("4. CREATING PERFORMANCE PLOTS")
    logger.info("="*50)
    create_performance_plots(scalability_results)
    
    # 5. Summary report
    logger.info("\n" + "="*50)
    logger.info("5. PERFORMANCE SUMMARY")
    logger.info("="*50)
    
    avg_speedup = np.mean([r["speedup"] for r in scalability_results if r["speedup"] > 0])
    max_speedup = max([r["speedup"] for r in scalability_results])
    
    logger.info(f"Average speedup across problem sizes: {avg_speedup:.2f}x")
    logger.info(f"Maximum speedup achieved: {max_speedup:.2f}x")
    logger.info(f"Cache speedup: {caching_results['speedup']:.1f}x")
    logger.info(f"Memory reduction: {memory_results['memory_reduction_percent']:.1f}%")
    
    # Save detailed results
    results_summary = {
        "caching_results": caching_results,
        "memory_results": memory_results,
        "scalability_results": scalability_results,
        "summary": {
            "avg_speedup": avg_speedup,
            "max_speedup": max_speedup,
            "cache_speedup": caching_results['speedup'],
            "memory_reduction_percent": memory_results['memory_reduction_percent']
        }
    }
    
    import json
    with open("output/performance_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info("Performance demonstration completed successfully!")
    logger.info("Results saved to output/performance_results.json")


if __name__ == "__main__":
    main()