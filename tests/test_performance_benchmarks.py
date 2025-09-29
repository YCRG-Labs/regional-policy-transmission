"""
Performance benchmarks and regression testing.

This module tests computational performance, memory usage, and ensures
that performance doesn't regress over time.
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
import warnings

from regional_monetary_policy.data.fred_client import FREDClient
from regional_monetary_policy.data.data_manager import DataManager
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.policy.counterfactual_engine import CounterfactualEngine
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.models import RegionalParameters


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process(os.getpid())
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self):
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'execution_time': end_time - self.start_time,
            'memory_usage': end_memory - self.start_memory,
            'peak_memory': end_memory
        }


class TestDataProcessingPerformance:
    """Test performance of data processing operations."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        n_regions = 10
        n_periods = 500  # ~40 years of monthly data
        
        regions = [f"Region_{i+1:02d}" for i in range(n_regions)]
        dates = pd.date_range('1980-01-01', periods=n_periods, freq='ME')
        
        # Generate correlated regional data
        np.random.seed(42)
        
        # Output gaps with regional correlation
        correlation_matrix = 0.3 * np.ones((n_regions, n_regions)) + 0.7 * np.eye(n_regions)
        output_gaps = np.random.multivariate_normal(
            mean=np.zeros(n_regions),
            cov=0.01 * correlation_matrix,
            size=n_periods
        ).T
        
        # Inflation rates
        inflation_rates = np.random.multivariate_normal(
            mean=0.02 * np.ones(n_regions),
            cov=0.0001 * correlation_matrix,
            size=n_periods
        ).T
        
        # Interest rates
        interest_rates = np.random.normal(0.03, 0.01, n_periods)
        
        return RegionalDataset(
            output_gaps=pd.DataFrame(output_gaps, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(inflation_rates, index=regions, columns=dates),
            interest_rates=pd.Series(interest_rates, index=dates),
            real_time_estimates={},
            metadata={'performance_test': True}
        )
    
    @pytest.mark.slow
    def test_data_loading_performance(self, large_dataset):
        """Test data loading and processing performance."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Simulate data processing operations
        # 1. Data validation
        assert not large_dataset.output_gaps.isnull().any().any()
        assert not large_dataset.inflation_rates.isnull().any().any()
        
        # 2. Data transformations
        output_gaps_standardized = (large_dataset.output_gaps - 
                                   large_dataset.output_gaps.mean()) / large_dataset.output_gaps.std()
        
        # 3. Rolling window calculations
        window_size = 12
        rolling_means = large_dataset.output_gaps.rolling(window=window_size, axis=1).mean()
        
        # 4. Correlation calculations
        correlation_matrix = large_dataset.output_gaps.T.corr()
        
        metrics = monitor.stop()
        
        # Performance benchmarks
        assert metrics['execution_time'] < 5.0, f"Data processing too slow: {metrics['execution_time']:.2f}s"
        assert metrics['memory_usage'] < 100, f"Memory usage too high: {metrics['memory_usage']:.1f}MB"
        
        # Validate results
        assert output_gaps_standardized.shape == large_dataset.output_gaps.shape
        assert rolling_means.shape == large_dataset.output_gaps.shape
        assert correlation_matrix.shape == (10, 10)
    
    @pytest.mark.slow
    def test_spatial_operations_performance(self, large_dataset):
        """Test performance of spatial operations."""
        monitor = PerformanceMonitor()
        
        # Create spatial handler
        regions = large_dataset.regions
        spatial_handler = SpatialModelHandler(regions)
        
        # Create large spatial interaction datasets
        n_regions = len(regions)
        
        # Trade data (all pairs)
        trade_data = []
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    trade_data.append({
                        'origin': regions[i],
                        'destination': regions[j],
                        'trade_flow': np.random.uniform(50, 200)
                    })
        trade_df = pd.DataFrame(trade_data)
        
        # Migration data
        migration_data = []
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    migration_data.append({
                        'origin': regions[i],
                        'destination': regions[j],
                        'migration_flow': np.random.uniform(10, 50)
                    })
        migration_df = pd.DataFrame(migration_data)
        
        # Financial data (sparse)
        financial_data = []
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j and np.random.random() > 0.7:  # 30% connectivity
                    financial_data.append({
                        'origin': regions[i],
                        'destination': regions[j],
                        'financial_flow': np.random.uniform(20, 100)
                    })
        financial_df = pd.DataFrame(financial_data)
        
        # Distance matrix
        distance_matrix = np.random.uniform(100, 2000, (n_regions, n_regions))
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        monitor.start()
        
        # Construct spatial weights
        spatial_weights = spatial_handler.construct_weights(
            trade_df, migration_df, financial_df, distance_matrix,
            weights=(0.4, 0.3, 0.2, 0.1)
        )
        
        # Compute spatial lags for all time periods
        spatial_lags = spatial_handler.compute_spatial_lags(
            large_dataset.output_gaps.T, spatial_weights
        )
        
        # Validate spatial matrix
        validation_report = spatial_handler.validate_spatial_matrix(spatial_weights)
        
        metrics = monitor.stop()
        
        # Performance benchmarks
        assert metrics['execution_time'] < 10.0, f"Spatial operations too slow: {metrics['execution_time']:.2f}s"
        assert metrics['memory_usage'] < 200, f"Memory usage too high: {metrics['memory_usage']:.1f}MB"
        
        # Validate results
        assert spatial_weights.shape == (n_regions, n_regions)
        assert spatial_lags.shape == large_dataset.output_gaps.T.shape
        assert validation_report.is_valid


class TestEstimationPerformance:
    """Test performance of econometric estimation procedures."""
    
    @pytest.fixture
    def estimation_dataset(self):
        """Create dataset optimized for estimation performance testing."""
        n_regions = 8
        n_periods = 300
        
        regions = [f"Region_{i+1:02d}" for i in range(n_regions)]
        dates = pd.date_range('1995-01-01', periods=n_periods, freq='ME')
        
        # Generate well-behaved synthetic data
        np.random.seed(42)
        
        # Regional parameters for data generation
        true_sigma = np.random.uniform(0.5, 1.5, n_regions)
        true_kappa = np.random.uniform(0.05, 0.2, n_regions)
        
        # Generate data with known structure
        output_gaps = np.zeros((n_regions, n_periods))
        inflation_rates = np.zeros((n_regions, n_periods))
        interest_rates = np.random.normal(0.03, 0.01, n_periods)
        
        # Simple data generation for performance testing
        for t in range(1, n_periods):
            for i in range(n_regions):
                # IS curve
                output_gaps[i, t] = (0.7 * output_gaps[i, t-1] - 
                                   true_sigma[i] * (interest_rates[t] - inflation_rates[i, t-1]) +
                                   np.random.normal(0, 0.01))
                
                # Phillips curve
                inflation_rates[i, t] = (0.8 * inflation_rates[i, t-1] +
                                       true_kappa[i] * output_gaps[i, t] +
                                       np.random.normal(0, 0.005))
        
        return RegionalDataset(
            output_gaps=pd.DataFrame(output_gaps, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(inflation_rates, index=regions, columns=dates),
            interest_rates=pd.Series(interest_rates, index=dates),
            real_time_estimates={},
            metadata={'true_sigma': true_sigma, 'true_kappa': true_kappa}
        )
    
    @pytest.mark.slow
    def test_parameter_estimation_performance(self, estimation_dataset):
        """Test performance of parameter estimation procedures."""
        monitor = PerformanceMonitor()
        
        # Create spatial handler
        regions = estimation_dataset.regions
        spatial_handler = SpatialModelHandler(regions)
        
        # Create estimation configuration optimized for performance
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 20  # Reduce for performance testing
        config.max_iterations = 200
        config.convergence_tolerance = 1e-4  # Slightly relaxed for speed
        
        # Create parameter estimator
        estimator = ParameterEstimator(spatial_handler, config)
        
        monitor.start()
        
        # Run full estimation procedure
        results = estimator.estimate_full_model(estimation_dataset)
        
        metrics = monitor.stop()
        
        # Performance benchmarks
        assert metrics['execution_time'] < 60.0, f"Estimation too slow: {metrics['execution_time']:.2f}s"
        assert metrics['memory_usage'] < 500, f"Memory usage too high: {metrics['memory_usage']:.1f}MB"
        
        # Validate estimation completed successfully
        assert results.estimation_time > 0
        assert len(results.regional_parameters.sigma) == 8
        assert results.convergence_info['overall_converged']
    
    @pytest.mark.slow
    def test_bootstrap_performance(self, estimation_dataset):
        """Test performance of bootstrap procedures."""
        monitor = PerformanceMonitor()
        
        # Create simplified regional parameters for bootstrap testing
        regional_params = RegionalParameters(
            sigma=np.random.uniform(0.5, 1.5, 8),
            kappa=np.random.uniform(0.05, 0.2, 8),
            psi=np.random.uniform(-0.1, 0.2, 8),
            phi=np.random.uniform(-0.05, 0.1, 8),
            beta=np.random.uniform(0.98, 0.99, 8),
            standard_errors={},
            confidence_intervals={}
        )
        
        # Create parameter estimator
        regions = estimation_dataset.regions
        spatial_handler = SpatialModelHandler(regions)
        
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 50  # Moderate number for performance testing
        
        estimator = ParameterEstimator(spatial_handler, config)
        
        monitor.start()
        
        # Compute bootstrap standard errors
        bootstrap_se = estimator.compute_standard_errors(estimation_dataset, regional_params)
        
        metrics = monitor.stop()
        
        # Performance benchmarks
        assert metrics['execution_time'] < 30.0, f"Bootstrap too slow: {metrics['execution_time']:.2f}s"
        assert metrics['memory_usage'] < 300, f"Memory usage too high: {metrics['memory_usage']:.1f}MB"
        
        # Validate bootstrap results
        assert 'sigma' in bootstrap_se
        assert len(bootstrap_se['sigma']) == 8
        assert np.all(bootstrap_se['sigma'] >= 0)


class TestCounterfactualPerformance:
    """Test performance of counterfactual analysis."""
    
    @pytest.fixture
    def counterfactual_setup(self):
        """Create setup for counterfactual performance testing."""
        n_regions = 6
        n_periods = 200
        
        # Regional parameters
        regional_params = RegionalParameters(
            sigma=np.random.uniform(0.5, 1.5, n_regions),
            kappa=np.random.uniform(0.05, 0.2, n_regions),
            psi=np.random.uniform(-0.1, 0.2, n_regions),
            phi=np.random.uniform(-0.05, 0.1, n_regions),
            beta=np.random.uniform(0.98, 0.99, n_regions),
            standard_errors={},
            confidence_intervals={}
        )
        
        # Welfare weights
        welfare_weights = np.random.dirichlet(np.ones(n_regions))
        
        # Historical data
        dates = pd.date_range('2000-01-01', periods=n_periods, freq='ME')
        regions = [f"Region_{i+1:02d}" for i in range(n_regions)]
        
        np.random.seed(42)
        historical_data = RegionalDataset(
            output_gaps=pd.DataFrame(
                np.random.normal(0, 0.01, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            inflation_rates=pd.DataFrame(
                np.random.normal(0.02, 0.005, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            interest_rates=pd.Series(
                np.random.normal(0.03, 0.01, n_periods),
                index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        return regional_params, welfare_weights, historical_data
    
    @pytest.mark.slow
    def test_scenario_generation_performance(self, counterfactual_setup):
        """Test performance of counterfactual scenario generation."""
        regional_params, welfare_weights, historical_data = counterfactual_setup
        
        monitor = PerformanceMonitor()
        
        # Create counterfactual engine
        counterfactual_engine = CounterfactualEngine(regional_params, welfare_weights)
        
        monitor.start()
        
        # Generate all scenarios
        baseline = counterfactual_engine.generate_baseline_scenario(historical_data)
        perfect_info = counterfactual_engine.generate_perfect_info_scenario(historical_data)
        optimal_regional = counterfactual_engine.generate_optimal_regional_scenario(historical_data)
        perfect_regional = counterfactual_engine.generate_perfect_regional_scenario(historical_data)
        
        # Compare scenarios
        scenarios = [baseline, perfect_info, optimal_regional, perfect_regional]
        comparison_results = counterfactual_engine.compare_scenarios(scenarios)
        
        metrics = monitor.stop()
        
        # Performance benchmarks
        assert metrics['execution_time'] < 45.0, f"Counterfactual generation too slow: {metrics['execution_time']:.2f}s"
        assert metrics['memory_usage'] < 400, f"Memory usage too high: {metrics['memory_usage']:.1f}MB"
        
        # Validate results
        assert len(scenarios) == 4
        assert all(len(scenario.policy_rates) == 200 for scenario in scenarios)
        assert comparison_results is not None


class TestMemoryUsage:
    """Test memory usage patterns and detect memory leaks."""
    
    def test_memory_usage_scaling(self):
        """Test how memory usage scales with problem size."""
        memory_usage = []
        problem_sizes = [50, 100, 200, 300]
        
        for n_periods in problem_sizes:
            # Create dataset of varying size
            n_regions = 4
            regions = [f"Region_{i+1}" for i in range(n_regions)]
            dates = pd.date_range('2000-01-01', periods=n_periods, freq='ME')
            
            np.random.seed(42)  # Consistent data generation
            
            dataset = RegionalDataset(
                output_gaps=pd.DataFrame(
                    np.random.normal(0, 0.01, (n_regions, n_periods)),
                    index=regions, columns=dates
                ),
                inflation_rates=pd.DataFrame(
                    np.random.normal(0.02, 0.005, (n_regions, n_periods)),
                    index=regions, columns=dates
                ),
                interest_rates=pd.Series(
                    np.random.normal(0.03, 0.01, n_periods),
                    index=dates
                ),
                real_time_estimates={},
                metadata={}
            )
            
            # Measure memory usage
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform some operations
            correlations = dataset.output_gaps.T.corr()
            rolling_stats = dataset.output_gaps.rolling(window=12, axis=1).mean()
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            memory_usage.append(memory_used)
            
            # Clean up
            del dataset, correlations, rolling_stats
        
        # Memory usage should scale reasonably (not exponentially)
        # Check that memory usage doesn't grow too fast
        for i in range(1, len(memory_usage)):
            growth_factor = memory_usage[i] / memory_usage[0]
            size_factor = problem_sizes[i] / problem_sizes[0]
            
            # Memory growth should be roughly linear with data size
            assert growth_factor < size_factor * 2, f"Memory usage growing too fast: {growth_factor:.2f}x for {size_factor:.2f}x data"
    
    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large dataset
        n_regions = 10
        n_periods = 500
        
        regions = [f"Region_{i+1:02d}" for i in range(n_regions)]
        dates = pd.date_range('1980-01-01', periods=n_periods, freq='ME')
        
        large_dataset = RegionalDataset(
            output_gaps=pd.DataFrame(
                np.random.normal(0, 0.01, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            inflation_rates=pd.DataFrame(
                np.random.normal(0.02, 0.005, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            interest_rates=pd.Series(
                np.random.normal(0.03, 0.01, n_periods),
                index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        # Perform memory-intensive operations
        correlations = large_dataset.output_gaps.T.corr()
        covariances = large_dataset.output_gaps.T.cov()
        rolling_means = large_dataset.output_gaps.rolling(window=24, axis=1).mean()
        
        # Clean up explicitly
        del large_dataset, correlations, covariances, rolling_means
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50, f"Memory not properly cleaned up: {memory_increase:.1f}MB increase"


class TestRegressionBenchmarks:
    """Regression tests to ensure performance doesn't degrade over time."""
    
    @pytest.fixture
    def benchmark_data_file(self):
        """Create or load benchmark data file."""
        benchmark_file = Path("tests") / "benchmark_results.json"
        
        if not benchmark_file.exists():
            # Create initial benchmark data
            benchmark_data = {
                'data_processing': {'max_time': 5.0, 'max_memory': 100},
                'spatial_operations': {'max_time': 10.0, 'max_memory': 200},
                'parameter_estimation': {'max_time': 60.0, 'max_memory': 500},
                'counterfactual_analysis': {'max_time': 45.0, 'max_memory': 400}
            }
            
            import json
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
        
        return benchmark_file
    
    def test_performance_regression(self, benchmark_data_file):
        """Test that performance hasn't regressed from benchmarks."""
        import json
        
        with open(benchmark_data_file, 'r') as f:
            benchmarks = json.load(f)
        
        # Run quick performance tests
        monitor = PerformanceMonitor()
        
        # Test 1: Data processing
        monitor.start()
        
        n_regions, n_periods = 5, 100
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        dates = pd.date_range('2010-01-01', periods=n_periods, freq='ME')
        
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(
                np.random.normal(0, 0.01, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            inflation_rates=pd.DataFrame(
                np.random.normal(0.02, 0.005, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            interest_rates=pd.Series(
                np.random.normal(0.03, 0.01, n_periods),
                index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        # Basic operations
        correlations = dataset.output_gaps.T.corr()
        rolling_stats = dataset.output_gaps.rolling(window=6, axis=1).mean()
        
        metrics = monitor.stop()
        
        # Check against benchmarks (allow 20% tolerance for system variations)
        data_benchmark = benchmarks['data_processing']
        assert metrics['execution_time'] < data_benchmark['max_time'] * 1.2, \
            f"Data processing regression: {metrics['execution_time']:.2f}s > {data_benchmark['max_time']}s"
        
        assert metrics['memory_usage'] < data_benchmark['max_memory'] * 1.2, \
            f"Memory usage regression: {metrics['memory_usage']:.1f}MB > {data_benchmark['max_memory']}MB"


if __name__ == '__main__':
    pytest.main([__file__])