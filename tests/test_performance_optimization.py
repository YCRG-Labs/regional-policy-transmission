"""
Tests for performance optimization components.
"""

import pytest
import numpy as np
import pandas as pd
import time
import tempfile
import os
from unittest.mock import Mock, patch

from regional_monetary_policy.performance import (
    PerformanceProfiler, ComputationOptimizer, MemoryManager,
    IntelligentCacheManager, SystemMonitor, PerformanceAlert, AlertLevel
)
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.econometric.models import EstimationConfig


class TestPerformanceProfiler:
    """Test performance profiler functionality."""
    
    def test_function_profiling(self):
        """Test function profiling decorator."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_function("test_function")
        def slow_function(n):
            return sum(i**2 for i in range(n))
        
        result = slow_function(1000)
        assert result == sum(i**2 for i in range(1000))
        
        # Check profiling results
        assert "test_function" in profiler.profile_results
        profile_result = profiler.profile_results["test_function"]
        assert profile_result.execution_time > 0
        assert profile_result.call_count == 1
    
    def test_profile_block_context(self):
        """Test profile block context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.profile_block("test_block"):
            time.sleep(0.01)  # Small delay
        
        # Should have logged the block execution
        # (We can't easily test the logging output, but ensure no errors)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_function("fast_func")
        def fast_function():
            return 42
        
        @profiler.profile_function("slow_func")
        def slow_function():
            time.sleep(0.01)
            return 42
        
        fast_function()
        slow_function()
        
        summary = profiler.get_performance_summary()
        assert "total_execution_time" in summary
        assert "slowest_functions" in summary
        assert len(summary["slowest_functions"]) > 0


class TestComputationOptimizer:
    """Test computation optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = ComputationOptimizer(max_workers=2)
        assert optimizer.max_workers == 2
        assert not optimizer.use_gpu  # Assuming no GPU in test environment
    
    def test_matrix_operations_cpu(self):
        """Test CPU matrix operations."""
        optimizer = ComputationOptimizer(max_workers=2, use_gpu=False)
        
        # Test spatial weights computation
        n_regions = 5
        trade_matrix = np.random.rand(n_regions, n_regions)
        migration_matrix = np.random.rand(n_regions, n_regions)
        financial_matrix = np.random.rand(n_regions, n_regions)
        distance_matrix = np.random.rand(n_regions, n_regions)
        
        matrices = [trade_matrix, migration_matrix, financial_matrix, distance_matrix]
        
        result = optimizer.optimize_matrix_operations(
            'spatial_weights', 
            matrices,
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        
        assert result.shape == (n_regions, n_regions)
        assert np.allclose(np.sum(result, axis=1), 1.0, atol=1e-10)  # Row-normalized
    
    def test_parallel_data_loading(self):
        """Test parallel data loading."""
        optimizer = ComputationOptimizer(max_workers=2)
        
        def mock_data_loader(source):
            time.sleep(0.01)  # Simulate I/O delay
            return f"data_from_{source}"
        
        sources = ["source1", "source2", "source3"]
        
        results = optimizer.optimize_data_loading(mock_data_loader, sources)
        
        assert results["success_rate"] == 1.0
        assert len(results["results"]) == 3
        assert results["results"]["source1"] == "data_from_source1"


class TestMemoryManager:
    """Test memory manager functionality."""
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        memory_manager = MemoryManager()
        stats = memory_manager.get_memory_stats()
        
        assert stats.total_memory > 0
        assert stats.available_memory > 0
        assert 0 <= stats.memory_percent <= 100
    
    def test_dataframe_optimization(self):
        """Test DataFrame memory optimization."""
        memory_manager = MemoryManager()
        
        # Create test DataFrame with suboptimal types
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3, 4, 5], dtype='int64'),
            'float_col': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64'),
            'cat_col': ['A', 'B', 'A', 'B', 'A']
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = memory_manager.optimize_dataframe(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should use less memory (or at least not more)
        assert optimized_memory <= original_memory
        
        # Check that categorical conversion worked
        assert optimized_df['cat_col'].dtype.name == 'category'
    
    def test_memory_mapped_array(self):
        """Test memory-mapped array creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = MemoryManager(temp_dir=temp_dir)
            
            shape = (100, 50)
            mmap_array = memory_manager.create_memory_mapped_array(
                shape, 
                dtype=np.float64,
                cache_key="test_array"
            )
            
            assert mmap_array.shape == shape
            assert mmap_array.dtype == np.float64
            
            # Test writing and reading
            mmap_array[0, 0] = 42.0
            assert mmap_array[0, 0] == 42.0
    
    def test_chunk_processing(self):
        """Test chunked array processing."""
        memory_manager = MemoryManager()
        
        # Create large array
        large_array = np.random.rand(1000, 100)
        
        def sum_processing(chunk):
            return np.sum(chunk, axis=1)
        
        result = memory_manager.chunk_array_processing(
            large_array,
            sum_processing,
            chunk_size=200
        )
        
        # Compare with direct processing
        expected = np.sum(large_array, axis=1)
        np.testing.assert_array_almost_equal(result, expected)


class TestIntelligentCacheManager:
    """Test intelligent cache manager functionality."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = IntelligentCacheManager(cache_dir=temp_dir)
            
            # Test set and get
            test_data = {"key": "value", "number": 42}
            cache_manager.set("test_key", test_data)
            
            retrieved_data = cache_manager.get("test_key")
            assert retrieved_data == test_data
            
            # Test non-existent key
            assert cache_manager.get("non_existent") is None
            assert cache_manager.get("non_existent", "default") == "default"
    
    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = IntelligentCacheManager(cache_dir=temp_dir)
            
            # Set item with short TTL
            cache_manager.set("expire_key", "expire_value", ttl_seconds=1)
            
            # Should be available immediately
            assert cache_manager.get("expire_key") == "expire_value"
            
            # Wait for expiration
            time.sleep(1.1)
            
            # Should be expired now
            assert cache_manager.get("expire_key") is None
    
    def test_api_response_caching(self):
        """Test API response caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = IntelligentCacheManager(cache_dir=temp_dir)
            
            api_response = {"data": [1, 2, 3], "status": "success"}
            params = {"series": "GDP", "start": "2020-01-01"}
            
            # Cache API response
            cache_key = cache_manager.cache_api_response(
                "FRED", "series", params, api_response
            )
            
            # Retrieve cached response
            cached_response = cache_manager.get_cached_api_response(
                "FRED", "series", params
            )
            
            assert cached_response == api_response
    
    def test_computation_caching(self):
        """Test computation result caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = IntelligentCacheManager(cache_dir=temp_dir)
            
            computation_result = np.array([1, 2, 3, 4, 5])
            input_params = {"method": "gmm", "iterations": 100}
            
            # Cache computation result
            cache_key = cache_manager.cache_computation_result(
                "parameter_estimation", input_params, computation_result
            )
            
            # Retrieve cached result
            cached_result = cache_manager.get_cached_computation(
                "parameter_estimation", input_params
            )
            
            np.testing.assert_array_equal(cached_result, computation_result)


class TestSystemMonitor:
    """Test system monitoring functionality."""
    
    def test_system_health_collection(self):
        """Test system health metrics collection."""
        monitor = SystemMonitor(monitoring_interval=1, enable_alerts=False)
        
        health = monitor.collect_system_health()
        
        assert 0 <= health.cpu_percent <= 100
        assert 0 <= health.memory_percent <= 100
        assert 0 <= health.disk_percent <= 100
        assert health.process_count > 0
        assert health.uptime_seconds > 0
    
    def test_custom_metrics(self):
        """Test custom metric recording."""
        monitor = SystemMonitor(enable_alerts=False)
        
        # Record custom metric
        monitor.record_custom_metric(
            "estimation_time", 
            5.2, 
            "seconds",
            warning_threshold=10.0,
            critical_threshold=20.0
        )
        
        # Check metric was recorded
        assert "estimation_time" in monitor.metrics_history
        assert len(monitor.metrics_history["estimation_time"]) == 1
        
        metric = monitor.metrics_history["estimation_time"][0]
        assert metric.value == 5.2
        assert metric.unit == "seconds"
    
    def test_alert_generation(self):
        """Test alert generation."""
        monitor = SystemMonitor(enable_alerts=True)
        
        # Set low threshold for testing
        monitor.set_threshold("test_metric", warning=5.0, critical=10.0)
        
        # Record metric that should trigger warning
        monitor.record_custom_metric("test_metric", 7.0)
        
        # Process any pending alerts
        time.sleep(0.1)
        
        # Check if alert was generated (would be in queue)
        # Note: In real usage, alerts would be processed by handlers


class TestPerformanceIntegration:
    """Test integration of performance components with parameter estimator."""
    
    def test_optimized_parameter_estimator(self):
        """Test parameter estimator with performance optimization."""
        # Create test data
        n_regions = 3
        n_periods = 50
        
        regions = [f"region_{i}" for i in range(n_regions)]
        dates = pd.date_range("2020-01-01", periods=n_periods, freq="M")
        
        # Generate synthetic data
        np.random.seed(42)
        output_gaps = pd.DataFrame(
            np.random.randn(n_regions, n_periods) * 0.02,
            index=regions,
            columns=dates
        ).T
        
        inflation_rates = pd.DataFrame(
            np.random.randn(n_regions, n_periods) * 0.01 + 0.02,
            index=regions,
            columns=dates
        ).T
        
        interest_rates = pd.Series(
            np.random.randn(n_periods) * 0.005 + 0.025,
            index=dates
        )
        
        dataset = RegionalDataset(
            output_gaps=output_gaps,
            inflation_rates=inflation_rates,
            interest_rates=interest_rates,
            real_time_estimates={},
            metadata={"source": "test"}
        )
        
        # Create spatial handler and estimator
        spatial_handler = SpatialModelHandler(regions)
        config = EstimationConfig()
        
        # Test with optimization enabled
        estimator_optimized = ParameterEstimator(
            spatial_handler, 
            config, 
            enable_performance_optimization=True
        )
        
        # Test with optimization disabled
        estimator_standard = ParameterEstimator(
            spatial_handler, 
            config, 
            enable_performance_optimization=False
        )
        
        # Both should work (though we can't easily test performance difference in unit tests)
        assert estimator_optimized.enable_optimization
        assert not estimator_standard.enable_optimization
        
        # Test performance report generation
        if estimator_optimized.enable_optimization:
            report = estimator_optimized.get_performance_report()
            assert "profiling_summary" in report
            assert "memory_stats" in report
            assert "optimization_recommendations" in report
        
        # Cleanup
        estimator_optimized.cleanup_performance_resources()


if __name__ == "__main__":
    pytest.main([__file__])