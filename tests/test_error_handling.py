"""
Integration tests for error handling and recovery mechanisms.

This module tests the comprehensive error handling system including
logging, recovery strategies, data quality diagnostics, and progress monitoring.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timedelta
import logging
import json

from regional_monetary_policy.exceptions import (
    DataRetrievalError, APIRateLimitError, EstimationError, 
    NumericalError, DataValidationError, InsufficientDataError
)
from regional_monetary_policy.logging_config import (
    setup_logging, get_logger, get_performance_logger, ErrorLogger
)
from regional_monetary_policy.error_recovery import (
    ErrorRecoveryManager, APIRetryStrategy, EstimationRecoveryStrategy,
    DataQualityRecoveryStrategy, with_recovery, RetryConfig
)
from regional_monetary_policy.data_quality import (
    DataQualityAssessor, DataCleaner, validate_data_for_analysis
)
from regional_monetary_policy.progress_monitor import (
    ProgressMonitor, ProgressTracker, SystemMonitor, track_progress
)


class TestLoggingSystem:
    """Test the logging configuration and functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logging_setup(self):
        """Test basic logging setup."""
        config = setup_logging(
            log_level="DEBUG",
            log_dir=self.log_dir,
            enable_console=True,
            enable_file=True
        )
        
        assert config is not None
        assert self.log_dir.exists()
        
        # Test logger creation
        logger = get_logger("test.module")
        assert logger is not None
        assert logger.name == "test.module"
    
    def test_performance_logger(self):
        """Test performance logging functionality."""
        setup_logging(log_dir=self.log_dir)
        perf_logger = get_performance_logger("test.performance")
        
        # Test timer context manager
        with perf_logger.timer("test_operation", param1="value1"):
            time.sleep(0.1)  # Simulate work
        
        # Test progress logging
        perf_logger.log_progress("test_progress", 5, 10, detail="test")
    
    def test_error_logger(self):
        """Test error logging functionality."""
        setup_logging(log_dir=self.log_dir)
        logger = get_logger("test.error")
        error_logger = ErrorLogger(logger)
        
        # Test error logging
        test_error = DataRetrievalError("Test error", series_code="TEST")
        error_logger.log_error(test_error, context={"test": "context"})
        
        # Test recovery logging
        error_logger.log_recovery_attempt(test_error, 1, 3, "retry")
        error_logger.log_recovery_success(test_error, 2, "retry")
        error_logger.log_recovery_failure(test_error, 3, ["retry", "fallback"])
    
    def test_json_logging(self):
        """Test JSON logging format."""
        config = setup_logging(
            log_dir=self.log_dir,
            enable_json=True,
            enable_console=False
        )
        
        logger = get_logger("test.json")
        logger.info("Test message", extra={'extra_fields': {'key': 'value'}})
        
        # Check that log file exists and contains JSON
        log_files = list(self.log_dir.glob("*.log"))
        assert len(log_files) > 0
        
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            # Should be valid JSON
            json.loads(log_content.strip())


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def setup_method(self):
        """Setup test environment."""
        self.recovery_manager = ErrorRecoveryManager()
    
    def test_api_retry_strategy(self):
        """Test API retry strategy."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        strategy = APIRetryStrategy(retry_config)
        
        # Test can_recover
        api_error = DataRetrievalError("API failed")
        assert strategy.can_recover(api_error, {})
        
        estimation_error = EstimationError("Estimation failed")
        assert not strategy.can_recover(estimation_error, {})
        
        # Test successful recovery
        call_count = 0
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DataRetrievalError("Temporary failure")
            return "success"
        
        context = {
            'original_function': mock_api_call,
            'args': (),
            'kwargs': {}
        }
        
        result = strategy.recover(api_error, context)
        assert result == "success"
        assert call_count == 3
    
    def test_estimation_recovery_strategy(self):
        """Test estimation recovery strategy."""
        strategy = EstimationRecoveryStrategy()
        
        # Test numerical error recovery
        numerical_error = NumericalError("Matrix singular")
        
        call_count = 0
        def mock_estimation(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and 'regularization' not in kwargs:
                raise NumericalError("Singular matrix")
            return "success"
        
        context = {
            'original_function': mock_estimation,
            'args': (),
            'kwargs': {}
        }
        
        result = strategy.recover(numerical_error, context)
        assert result == "success"
        assert call_count == 2
    
    def test_data_quality_recovery_strategy(self):
        """Test data quality recovery strategy."""
        strategy = DataQualityRecoveryStrategy()
        
        # Create test data with quality issues
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 100],  # Missing value and outlier
            'col2': [1, 2, 3, 4, 5]
        })
        
        validation_error = DataValidationError("Data quality issues")
        context = {'data': data}
        
        cleaned_data = strategy.recover(validation_error, context)
        
        # Check that data was cleaned
        assert not cleaned_data.isnull().any().any()  # No missing values
        assert cleaned_data['col1'].max() < 100  # Outlier handled
    
    def test_with_recovery_decorator(self):
        """Test the with_recovery decorator."""
        
        @with_recovery
        def failing_function():
            raise DataRetrievalError("API failed")
        
        # Should raise error since no recovery context provided
        with pytest.raises(DataRetrievalError):
            failing_function()
        
        # Test with recovery context
        call_count = 0
        
        @with_recovery(recovery_context={'test': 'context'})
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise DataRetrievalError("Temporary failure")
            return "success"
        
        # Mock the recovery manager to return success
        with patch.object(ErrorRecoveryManager, 'recover', return_value="recovered"):
            result = sometimes_failing_function()
            assert result == "recovered"
    
    def test_recovery_manager_integration(self):
        """Test the complete recovery manager."""
        manager = ErrorRecoveryManager()
        
        # Test with API error
        api_error = DataRetrievalError("API failed")
        
        def mock_api_call():
            return "success"
        
        context = {
            'original_function': mock_api_call,
            'args': (),
            'kwargs': {}
        }
        
        result = manager.recover(api_error, context)
        assert result == "success"


class TestDataQuality:
    """Test data quality assessment and cleaning."""
    
    def setup_method(self):
        """Setup test data."""
        # Create test data with various quality issues
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='M')
        
        self.test_data = pd.DataFrame({
            'good_column': np.random.normal(0, 1, 100),
            'missing_column': [np.nan if i % 10 == 0 else np.random.normal(0, 1) 
                             for i in range(100)],
            'outlier_column': [100 if i == 50 else np.random.normal(0, 1) 
                             for i in range(100)],
            'constant_column': [1.0] * 100,
            'correlated_column': None  # Will set based on good_column
        }, index=dates)
        
        # Create highly correlated column
        self.test_data['correlated_column'] = (
            self.test_data['good_column'] * 0.95 + np.random.normal(0, 0.1, 100)
        )
        
        # Add some duplicate rows
        self.test_data = pd.concat([self.test_data, self.test_data.iloc[:5]])
    
    def test_data_quality_assessment(self):
        """Test comprehensive data quality assessment."""
        assessor = DataQualityAssessor(
            missing_threshold=0.05,
            outlier_threshold=3.0,
            min_observations=50
        )
        
        report = assessor.assess_data_quality(self.test_data, "Test Dataset")
        
        # Check report structure
        assert report.dataset_name == "Test Dataset"
        assert report.total_observations > 0
        assert report.total_variables > 0
        assert len(report.issues) > 0
        
        # Check for expected issues
        issue_types = {issue.issue_type for issue in report.issues}
        assert 'missing_values' in issue_types
        assert 'outliers' in issue_types
        assert 'duplicates' in issue_types
        assert 'multicollinearity' in issue_types
        
        # Check summary
        summary = report.get_summary()
        assert 'total_issues' in summary
        assert 'severity_breakdown' in summary
    
    def test_data_cleaning(self):
        """Test automated data cleaning."""
        cleaner = DataCleaner()
        
        # Test conservative cleaning
        cleaned_data, cleaning_log = cleaner.clean_data(
            self.test_data, 
            cleaning_strategy="conservative"
        )
        
        # Check that data was cleaned
        assert len(cleaned_data) <= len(self.test_data)  # May have removed duplicates
        assert not cleaned_data.isnull().any().any()  # No missing values
        
        # Check cleaning log
        assert 'strategy' in cleaning_log
        assert 'actions_taken' in cleaning_log
        assert len(cleaning_log['actions_taken']) > 0
    
    def test_insufficient_data_validation(self):
        """Test validation with insufficient data."""
        small_data = self.test_data.iloc[:10]  # Only 10 observations
        
        with pytest.raises(DataValidationError):
            validate_data_for_analysis(small_data, min_observations=20)
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        required_columns = ['good_column', 'missing_required_column']
        
        with pytest.raises(DataValidationError):
            validate_data_for_analysis(
                self.test_data, 
                required_columns=required_columns
            )


class TestProgressMonitoring:
    """Test progress monitoring and performance tracking."""
    
    def setup_method(self):
        """Setup test environment."""
        self.monitor = ProgressMonitor(enable_system_monitoring=False)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.monitor.shutdown()
    
    def test_progress_tracker_creation(self):
        """Test creating and using progress trackers."""
        tracker = self.monitor.create_tracker("test_operation", total_steps=10)
        
        assert tracker.metrics.operation_name == "test_operation"
        assert tracker.metrics.total_steps == 10
        assert tracker.metrics.current_step == 0
        assert tracker.metrics.status == "running"
    
    def test_progress_updates(self):
        """Test progress updates and calculations."""
        tracker = self.monitor.create_tracker("test_operation", total_steps=10)
        
        # Test step updates
        tracker.update(step=5, phase="Processing data")
        assert tracker.metrics.current_step == 5
        assert tracker.metrics.current_phase == "Processing data"
        assert tracker.metrics.progress_percentage == 50.0
        
        # Test increment
        tracker.update(increment=True, phase="Next step")
        assert tracker.metrics.current_step == 6
        assert tracker.metrics.progress_percentage == 60.0
    
    def test_progress_completion(self):
        """Test operation completion."""
        tracker = self.monitor.create_tracker("test_operation", total_steps=5)
        
        # Complete successfully
        tracker.complete(success=True)
        assert tracker.metrics.status == "completed"
        assert tracker.metrics.end_time is not None
        
        # Check that operation was moved to completed list
        assert "test_operation" not in self.monitor.active_operations
    
    def test_progress_cancellation(self):
        """Test operation cancellation."""
        tracker = self.monitor.create_tracker("test_operation", total_steps=5)
        
        # Cancel operation
        tracker.cancel()
        assert tracker.is_cancelled()
        assert tracker.metrics.status == "cancelled"
        
        # Should raise error on further updates
        with pytest.raises(InterruptedError):
            tracker.update(step=1)
    
    def test_context_manager(self):
        """Test progress tracking context manager."""
        with self.monitor.track_operation("context_test", total_steps=3) as tracker:
            tracker.update(step=1, phase="Step 1")
            tracker.update(step=2, phase="Step 2")
            tracker.update(step=3, phase="Step 3")
        
        # Should be completed
        assert tracker.metrics.status == "completed"
    
    def test_context_manager_with_error(self):
        """Test context manager with error handling."""
        with pytest.raises(ValueError):
            with self.monitor.track_operation("error_test") as tracker:
                tracker.update(phase="Before error")
                raise ValueError("Test error")
        
        # Should be marked as failed
        assert tracker.metrics.status == "failed"
        assert "Test error" in tracker.metrics.error_message
    
    def test_track_progress_decorator(self):
        """Test the track_progress decorator."""
        
        @track_progress("decorated_operation", total_steps=3)
        def test_function(progress_tracker=None):
            assert progress_tracker is not None
            progress_tracker.update(step=1, phase="Step 1")
            progress_tracker.update(step=2, phase="Step 2")
            progress_tracker.update(step=3, phase="Step 3")
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_system_monitor(self):
        """Test system performance monitoring."""
        system_monitor = SystemMonitor(update_interval=0.1, history_size=5)
        
        # Start monitoring
        system_monitor.start_monitoring()
        
        # Wait for some metrics to be collected
        time.sleep(0.3)
        
        # Check metrics
        current_metrics = system_monitor.get_current_metrics()
        assert current_metrics is not None
        assert current_metrics.cpu_percent >= 0
        assert current_metrics.memory_percent >= 0
        
        # Test average metrics
        avg_metrics = system_monitor.get_average_metrics(minutes=1)
        if avg_metrics:  # May be None if not enough history
            assert 'cpu_percent' in avg_metrics
            assert 'memory_percent' in avg_metrics
        
        # Stop monitoring
        system_monitor.stop_monitoring()
    
    def test_monitor_status_reporting(self):
        """Test status reporting functionality."""
        # Create some operations
        tracker1 = self.monitor.create_tracker("op1", total_steps=10)
        tracker2 = self.monitor.create_tracker("op2", total_steps=5)
        
        tracker1.update(step=5)
        tracker2.update(step=2)
        
        # Test status retrieval
        status = self.monitor.get_operation_status("op1")
        assert status is not None
        assert status.current_step == 5
        
        all_status = self.monitor.get_all_active_operations()
        assert len(all_status) == 2
        assert "op1" in all_status
        assert "op2" in all_status
        
        # Test system status
        system_status = self.monitor.get_system_status()
        assert system_status['active_operations'] == 2


class TestIntegrationScenarios:
    """Test complete integration scenarios combining all error handling components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        
        # Setup logging
        setup_logging(log_dir=self.log_dir, log_level="DEBUG")
        
        # Setup progress monitoring
        self.monitor = ProgressMonitor(enable_system_monitoring=False)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.monitor.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_data_processing_workflow(self):
        """Test a complete data processing workflow with error handling."""
        
        @with_recovery
        def process_data_step(data, step_name, progress_tracker=None):
            """Simulate a data processing step that might fail."""
            if progress_tracker:
                progress_tracker.update(phase=f"Processing {step_name}")
            
            # Simulate potential failure
            if step_name == "validation" and np.random.random() < 0.3:
                raise DataValidationError("Validation failed")
            
            # Simulate processing time
            time.sleep(0.1)
            
            if progress_tracker:
                progress_tracker.update(increment=True)
            
            return f"Processed {step_name}"
        
        # Create test data
        test_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50),
            'col2': np.random.normal(0, 1, 50)
        })
        
        with self.monitor.track_operation("data_workflow", total_steps=4) as tracker:
            steps = ["loading", "validation", "cleaning", "analysis"]
            results = []
            
            for step in steps:
                try:
                    result = process_data_step(
                        test_data, 
                        step, 
                        progress_tracker=tracker
                    )
                    results.append(result)
                except Exception as e:
                    # Log error and continue with next step
                    logger = get_logger("test.workflow")
                    logger.error(f"Step {step} failed: {e}")
                    results.append(f"Failed {step}")
        
        assert len(results) == 4
        assert tracker.metrics.status == "completed"
    
    def test_api_failure_recovery_scenario(self):
        """Test API failure and recovery scenario."""
        
        class MockFREDClient:
            def __init__(self):
                self.call_count = 0
            
            @with_recovery
            def get_series_data(self, series_code):
                self.call_count += 1
                
                # Simulate rate limiting on first few calls
                if self.call_count <= 2:
                    raise APIRateLimitError(
                        f"Rate limit exceeded (attempt {self.call_count})",
                        retry_after=0.1
                    )
                
                # Simulate successful response
                return pd.Series([1, 2, 3, 4, 5], name=series_code)
        
        client = MockFREDClient()
        
        # This should succeed after retries
        with self.monitor.track_operation("api_test") as tracker:
            tracker.update(phase="Fetching data")
            
            # Mock the recovery context
            def mock_get_series():
                return client.get_series_data("TEST_SERIES")
            
            # Patch the recovery manager to handle the API call
            recovery_manager = ErrorRecoveryManager()
            
            try:
                result = mock_get_series()
                tracker.update(phase="Data received")
                assert len(result) == 5
                assert client.call_count >= 3  # Should have retried
            except Exception as e:
                tracker.complete(success=False, error_message=str(e))
                raise
    
    def test_estimation_failure_recovery_scenario(self):
        """Test estimation failure and recovery scenario."""
        
        def mock_estimation_procedure(data, regularization=None, solver='default'):
            """Mock estimation that fails without regularization."""
            if regularization is None:
                raise NumericalError("Matrix is singular")
            
            # Simulate successful estimation with regularization
            return {
                'parameters': np.random.normal(0, 1, 5),
                'standard_errors': np.random.uniform(0.1, 0.5, 5),
                'regularization': regularization
            }
        
        # Create test data
        test_data = pd.DataFrame(np.random.normal(0, 1, (100, 5)))
        
        with self.monitor.track_operation("estimation_test") as tracker:
            tracker.update(phase="Starting estimation")
            
            # Use recovery decorator
            @with_recovery
            def run_estimation():
                return mock_estimation_procedure(test_data)
            
            # Mock the recovery context
            recovery_manager = ErrorRecoveryManager()
            
            try:
                # This should fail first, then succeed with regularization
                context = {
                    'original_function': mock_estimation_procedure,
                    'args': (test_data,),
                    'kwargs': {}
                }
                
                error = NumericalError("Matrix is singular")
                result = recovery_manager.recover(error, context)
                
                tracker.update(phase="Estimation completed")
                assert 'parameters' in result
                assert 'regularization' in result
                
            except Exception as e:
                tracker.complete(success=False, error_message=str(e))
                raise
    
    def test_comprehensive_error_logging(self):
        """Test that all error scenarios are properly logged."""
        logger = get_logger("test.comprehensive")
        error_logger = ErrorLogger(logger)
        
        # Test different error types
        errors_to_test = [
            DataRetrievalError("API failed", series_code="TEST"),
            EstimationError("Convergence failed", estimation_stage="stage_2"),
            DataValidationError("Invalid data", validation_failures=["missing", "outliers"]),
            NumericalError("Singular matrix", numerical_details={"condition_number": 1e-15})
        ]
        
        for error in errors_to_test:
            error_logger.log_error(
                error, 
                context={"test_context": "comprehensive_test"},
                recovery_action="Attempting recovery"
            )
        
        # Check that log files were created
        log_files = list(self.log_dir.glob("*.log"))
        assert len(log_files) > 0
        
        # Check log content contains error information
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            assert "DataRetrievalError" in log_content
            assert "EstimationError" in log_content
            assert "comprehensive_test" in log_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])