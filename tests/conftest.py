"""
Pytest configuration and shared fixtures for regional monetary policy tests.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Set test environment variables
os.environ['TESTING'] = '1'


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="rmp_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_time_series():
    """Generate sample time series data for testing."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='M')
    
    def generate_series(name, mean=0, std=1, trend=0):
        """Generate a time series with optional trend."""
        noise = np.random.normal(mean, std, len(dates))
        if trend != 0:
            trend_component = np.linspace(0, trend, len(dates))
            noise += trend_component
        return pd.Series(noise, index=dates, name=name)
    
    return generate_series


@pytest.fixture
def mock_fred_response():
    """Generate mock FRED API response structure."""
    def create_response(series_id, start_date, end_date, values=None):
        dates = pd.date_range(start_date, end_date, freq='M')
        
        if values is None:
            values = np.random.uniform(100, 200, len(dates))
        
        observations = []
        for date, value in zip(dates, values):
            observations.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': str(value) if not pd.isna(value) else '.'
            })
        
        return {
            'realtime_start': start_date,
            'realtime_end': end_date,
            'observations': observations
        }
    
    return create_response


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests as validation tests using synthetic data"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "replication: marks tests that replicate published research results"
    )
    config.addinivalue_line(
        "markers", "mathematical: marks tests for core mathematical operations"
    )
    config.addinivalue_line(
        "markers", "workflow: marks tests for end-to-end workflows"
    )


# Skip integration tests if no API key is available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle integration tests."""
    skip_integration = pytest.mark.skip(reason="FRED_API_KEY not set")
    
    for item in items:
        if "integration" in item.keywords and not os.getenv('FRED_API_KEY'):
            item.add_marker(skip_integration)


@pytest.fixture
def suppress_warnings():
    """Suppress specific warnings during testing."""
    import warnings
    
    # Suppress pandas performance warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    # Suppress numpy warnings about invalid values
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    
    yield
    
    # Reset warning filters
    warnings.resetwarnings()


@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'execution_time': end_time - self.start_time,
                'memory_usage': end_memory - self.start_memory,
                'peak_memory': end_memory
            }
    
    return PerformanceMonitor()


@pytest.fixture
def synthetic_regional_params():
    """Fixture providing synthetic regional parameters for testing."""
    return {
        'sigma': np.array([0.8, 1.0, 0.9, 1.1]),
        'kappa': np.array([0.1, 0.12, 0.08, 0.15]),
        'psi': np.array([0.15, -0.1, 0.2, 0.05]),
        'phi': np.array([0.08, 0.05, -0.03, 0.1]),
        'beta': np.array([0.99, 0.985, 0.99, 0.988])
    }


@pytest.fixture
def benchmark_tolerances():
    """Fixture providing tolerance levels for benchmark tests."""
    return {
        'parameter_recovery': 0.3,  # 30% tolerance for parameter recovery
        'welfare_calculation': 0.1,  # 10% tolerance for welfare calculations
        'policy_coefficient': 0.2,   # 20% tolerance for policy coefficients
        'spatial_autocorr': 0.05,    # 5% tolerance for spatial autocorrelation
        'performance_time': 1.2,     # 20% tolerance for performance benchmarks
        'performance_memory': 1.5    # 50% tolerance for memory usage
    }