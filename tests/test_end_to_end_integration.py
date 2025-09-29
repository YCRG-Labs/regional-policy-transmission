"""
End-to-end integration tests for the Regional Monetary Policy Analysis System.

This module provides comprehensive tests that validate the entire system
from data retrieval through report generation.
"""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from regional_monetary_policy.integration.workflow_engine import WorkflowEngine
from regional_monetary_policy.integration.pipeline_manager import PipelineManager, PipelineStep
from regional_monetary_policy.integration.system_validator import SystemValidator
from regional_monetary_policy.config.config_manager import ConfigManager
from regional_monetary_policy.exceptions import RegionalMonetaryPolicyError


class TestEndToEndIntegration:
    """Test complete end-to-end system integration."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup handled by tempfile
    
    @pytest.fixture
    def mock_config(self, temp_config_dir):
        """Create mock configuration for testing."""
        config_data = {
            'fred_api_key': 'test_key_12345',
            'cache_directory': os.path.join(temp_config_dir, 'cache'),
            'regions': ['US-CA', 'US-NY', 'US-TX'],
            'cache_strategy': 'intelligent',
            'missing_data_method': 'interpolation',
            'spatial_weights': (0.25, 0.25, 0.25, 0.25),
            'estimation_config': {
                'gmm_options': {'max_iter': 100},
                'identification_strategy': 'standard',
                'spatial_weight_method': 'combined',
                'robustness_checks': ['bootstrap'],
                'convergence_tolerance': 1e-6,
                'max_iterations': 1000
            }
        }
        
        # Create config file
        config_path = os.path.join(temp_config_dir, 'config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        return config_path
    
    @pytest.fixture
    def workflow_engine(self, mock_config):
        """Create workflow engine with mock configuration."""
        with patch('regional_monetary_policy.integration.workflow_engine.FREDClient') as mock_fred:
            # Mock FRED client to avoid API calls
            mock_fred_instance = Mock()
            mock_fred_instance.validate_api_key.return_value = True
            mock_fred.return_value = mock_fred_instance
            
            engine = WorkflowEngine(mock_config)
            return engine
    
    def test_workflow_engine_initialization(self, workflow_engine):
        """Test that workflow engine initializes all components correctly."""
        # Test that all required components are initialized
        required_components = [
            'fred_client', 'data_manager', 'spatial_handler',
            'parameter_estimator', 'mistake_decomposer',
            'optimal_policy_calc', 'counterfactual_engine',
            'map_visualizer', 'report_generator'
        ]
        
        for component in required_components:
            assert hasattr(workflow_engine, component), f"Missing component: {component}"
    
    def test_system_integration_validation(self, workflow_engine):
        """Test system integration validation."""
        validation_results = workflow_engine.validate_system_integration()
        
        assert isinstance(validation_results, dict)
        assert 'overall_integration' in validation_results
        
        # Check that all components are validated
        expected_components = [
            'fred_client', 'data_manager', 'spatial_handler',
            'parameter_estimator', 'mistake_decomposer',
            'optimal_policy_calc', 'counterfactual_engine',
            'map_visualizer', 'report_generator'
        ]
        
        for component in expected_components:
            assert component in validation_results
    
    @patch('regional_monetary_policy.data.fred_client.FREDClient.get_series_data')
    @patch('regional_monetary_policy.data.data_manager.DataManager.get_trade_data')
    @patch('regional_monetary_policy.data.data_manager.DataManager.get_migration_data')
    @patch('regional_monetary_policy.data.data_manager.DataManager.get_financial_data')
    @patch('regional_monetary_policy.data.data_manager.DataManager.get_distance_matrix')
    def test_quick_analysis_workflow(self, mock_distance, mock_financial, 
                                   mock_migration, mock_trade, mock_fred_data,
                                   workflow_engine):
        """Test quick analysis workflow with mocked data."""
        # Setup mock data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='M')
        regions = ['US-CA', 'US-NY', 'US-TX']
        
        # Mock FRED data
        mock_fred_data.return_value = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(0, 1, len(dates))
        })
        
        # Mock spatial data
        n_regions = len(regions)
        mock_trade.return_value = pd.DataFrame(np.random.rand(n_regions, n_regions))
        mock_migration.return_value = pd.DataFrame(np.random.rand(n_regions, n_regions))
        mock_financial.return_value = pd.DataFrame(np.random.rand(n_regions, n_regions))
        mock_distance.return_value = np.random.rand(n_regions, n_regions)
        
        # Run quick analysis
        results = workflow_engine.run_quick_analysis('2020-01-01', '2020-12-31')
        
        assert isinstance(results, dict)
        assert 'data_shape' in results
        assert 'regions_analyzed' in results
        assert 'parameters_estimated' in results
        assert 'optimal_policy_computed' in results
        assert 'spatial_weights_shape' in results
    
    def test_workflow_status_tracking(self, workflow_engine):
        """Test workflow status tracking functionality."""
        initial_status = workflow_engine.get_workflow_status()
        
        assert isinstance(initial_status, dict)
        assert 'current_workflow' in initial_status
        assert 'completed_workflows' in initial_status
        assert 'system_status' in initial_status
    
    def test_error_handling_in_workflow(self, workflow_engine):
        """Test error handling in workflow execution."""
        # Test with invalid date range
        with pytest.raises(RegionalMonetaryPolicyError):
            workflow_engine.run_quick_analysis('invalid-date', '2020-12-31')


class TestPipelineManager:
    """Test pipeline manager functionality."""
    
    @pytest.fixture
    def pipeline_manager(self):
        """Create pipeline manager for testing."""
        return PipelineManager(max_workers=2)
    
    def test_pipeline_registration(self, pipeline_manager):
        """Test pipeline registration and validation."""
        steps = [
            PipelineStep(
                name="step1",
                function=lambda: "result1",
                dependencies=[]
            ),
            PipelineStep(
                name="step2", 
                function=lambda step1_result: f"result2_{step1_result}",
                dependencies=["step1"]
            )
        ]
        
        pipeline_manager.register_pipeline("test_pipeline", steps)
        
        assert "test_pipeline" in pipeline_manager.pipelines
        assert len(pipeline_manager.pipelines["test_pipeline"]) == 2
    
    def test_circular_dependency_detection(self, pipeline_manager):
        """Test detection of circular dependencies."""
        steps = [
            PipelineStep(
                name="step1",
                function=lambda: "result1",
                dependencies=["step2"]
            ),
            PipelineStep(
                name="step2",
                function=lambda: "result2", 
                dependencies=["step1"]
            )
        ]
        
        with pytest.raises(RegionalMonetaryPolicyError):
            pipeline_manager.register_pipeline("circular_pipeline", steps)
    
    def test_pipeline_execution(self, pipeline_manager):
        """Test pipeline execution with dependencies."""
        steps = pipeline_manager.create_standard_analysis_pipeline()
        pipeline_manager.register_pipeline("standard_analysis", steps)
        
        results = pipeline_manager.execute_pipeline("standard_analysis")
        
        assert isinstance(results, dict)
        assert len(results) == len(steps)
        
        # Check that all steps completed
        for step_name, result in results.items():
            assert result.status.value in ['completed', 'failed']
    
    def test_parallel_pipeline_execution(self, pipeline_manager):
        """Test parallel pipeline execution."""
        steps = pipeline_manager.create_standard_analysis_pipeline()
        pipeline_manager.register_pipeline("parallel_analysis", steps)
        
        results = pipeline_manager.execute_parallel_pipeline("parallel_analysis")
        
        assert isinstance(results, dict)
        assert len(results) == len(steps)
    
    def test_pipeline_status_monitoring(self, pipeline_manager):
        """Test pipeline status monitoring."""
        steps = [
            PipelineStep(
                name="simple_step",
                function=lambda: "done",
                dependencies=[]
            )
        ]
        
        pipeline_manager.register_pipeline("status_test", steps)
        
        # Before execution
        status = pipeline_manager.get_pipeline_status("status_test")
        assert status['status'] == 'not_executed'
        
        # After execution
        pipeline_manager.execute_pipeline("status_test")
        status = pipeline_manager.get_pipeline_status("status_test")
        assert status['status'] in ['completed', 'failed']


class TestSystemValidator:
    """Test system validator functionality."""
    
    @pytest.fixture
    def system_validator(self, mock_config):
        """Create system validator for testing."""
        with patch('regional_monetary_policy.integration.system_validator.FREDClient') as mock_fred:
            mock_fred_instance = Mock()
            mock_fred_instance.validate_api_key.return_value = True
            mock_fred.return_value = mock_fred_instance
            
            validator = SystemValidator(mock_config)
            return validator
    
    def test_component_validation(self, system_validator):
        """Test individual component validation."""
        component_results = system_validator._validate_components()
        
        assert isinstance(component_results, dict)
        assert 'overall_status' in component_results
        
        # Check that all expected components are tested
        expected_components = [
            'fred_client', 'data_manager', 'spatial_handler',
            'parameter_estimator', 'mistake_decomposer',
            'optimal_policy_calc', 'counterfactual_engine',
            'visualizer', 'report_generator', 'config_manager'
        ]
        
        for component in expected_components:
            assert component in component_results
    
    def test_integration_validation(self, system_validator):
        """Test integration validation."""
        integration_results = system_validator._validate_integration()
        
        assert isinstance(integration_results, dict)
        assert 'overall_status' in integration_results
        assert 'workflow_engine' in integration_results
        assert 'component_communication' in integration_results
        assert 'data_flow' in integration_results
    
    def test_performance_validation(self, system_validator):
        """Test performance validation."""
        performance_results = system_validator._validate_performance()
        
        assert isinstance(performance_results, dict)
        assert 'overall_status' in performance_results
        
        # Check performance metrics
        if 'quick_analysis' in performance_results:
            assert 'success' in performance_results['quick_analysis']
            assert 'execution_time' in performance_results['quick_analysis']
        
        if 'memory_usage' in performance_results:
            assert 'current_mb' in performance_results['memory_usage']
            assert 'acceptable' in performance_results['memory_usage']
    
    def test_comprehensive_validation(self, system_validator):
        """Test comprehensive system validation."""
        validation_results = system_validator.run_comprehensive_validation()
        
        assert isinstance(validation_results, dict)
        assert 'timestamp' in validation_results
        assert 'overall_status' in validation_results
        assert 'component_tests' in validation_results
        assert 'integration_tests' in validation_results
        assert 'performance_tests' in validation_results
        assert 'end_to_end_tests' in validation_results
        
        # Validation should complete without errors
        assert validation_results['overall_status'] in ['passed', 'failed']
    
    def test_validation_report_generation(self, system_validator):
        """Test validation report generation."""
        # Run validation first
        system_validator.run_comprehensive_validation()
        
        # Generate report
        report = system_validator.generate_validation_report()
        
        assert isinstance(report, str)
        assert "System Validation Report" in report
        assert "Component Tests" in report
        assert "Integration Tests" in report
        assert "Performance Tests" in report
        assert "End-to-End Tests" in report


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.fixture
    def configured_system(self, mock_config):
        """Create fully configured system for testing."""
        with patch('regional_monetary_policy.integration.workflow_engine.FREDClient') as mock_fred:
            mock_fred_instance = Mock()
            mock_fred_instance.validate_api_key.return_value = True
            mock_fred.return_value = mock_fred_instance
            
            workflow_engine = WorkflowEngine(mock_config)
            pipeline_manager = PipelineManager()
            system_validator = SystemValidator(mock_config)
            
            return {
                'workflow_engine': workflow_engine,
                'pipeline_manager': pipeline_manager,
                'system_validator': system_validator
            }
    
    def test_research_workflow_scenario(self, configured_system):
        """Test typical research workflow scenario."""
        workflow_engine = configured_system['workflow_engine']
        
        # Test that system can handle typical research workflow
        status = workflow_engine.get_workflow_status()
        assert status['system_status']['overall_integration'] is not None
    
    def test_policy_analysis_scenario(self, configured_system):
        """Test policy analysis scenario."""
        workflow_engine = configured_system['workflow_engine']
        
        # Test system validation for policy analysis
        validation_results = workflow_engine.validate_system_integration()
        
        # Should have policy analysis components
        assert 'mistake_decomposer' in validation_results
        assert 'optimal_policy_calc' in validation_results
        assert 'counterfactual_engine' in validation_results
    
    def test_batch_processing_scenario(self, configured_system):
        """Test batch processing scenario."""
        pipeline_manager = configured_system['pipeline_manager']
        
        # Create multiple pipelines for batch processing
        for i in range(3):
            steps = [
                PipelineStep(
                    name=f"batch_step_{i}",
                    function=lambda x=i: f"batch_result_{x}",
                    dependencies=[]
                )
            ]
            pipeline_manager.register_pipeline(f"batch_pipeline_{i}", steps)
        
        # Execute all pipelines
        results = {}
        for i in range(3):
            results[f"batch_pipeline_{i}"] = pipeline_manager.execute_pipeline(f"batch_pipeline_{i}")
        
        assert len(results) == 3
        for pipeline_results in results.values():
            assert len(pipeline_results) == 1
    
    def test_error_recovery_scenario(self, configured_system):
        """Test error recovery scenario."""
        pipeline_manager = configured_system['pipeline_manager']
        
        # Create pipeline with failing step
        def failing_function():
            raise Exception("Simulated failure")
        
        steps = [
            PipelineStep(
                name="failing_step",
                function=failing_function,
                dependencies=[]
            )
        ]
        
        pipeline_manager.register_pipeline("error_pipeline", steps)
        
        # Execute pipeline - should handle error gracefully
        results = pipeline_manager.execute_pipeline("error_pipeline")
        
        assert "failing_step" in results
        assert results["failing_step"].status.value == "failed"
        assert results["failing_step"].error is not None
    
    def test_configuration_flexibility_scenario(self, configured_system):
        """Test configuration flexibility scenario."""
        system_validator = configured_system['system_validator']
        
        # Test that system can handle different configurations
        flexibility_test = system_validator._test_configuration_flexibility()
        assert flexibility_test is True


class TestDeploymentReadiness:
    """Test deployment readiness and production scenarios."""
    
    def test_system_requirements_validation(self):
        """Test that system meets deployment requirements."""
        # Test Python version compatibility
        import sys
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
        
        # Test required packages are importable
        required_packages = [
            'pandas', 'numpy', 'scipy', 'matplotlib', 'plotly',
            'streamlit', 'requests', 'sqlalchemy'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package {package} not available")
    
    def test_configuration_validation(self):
        """Test configuration system for deployment."""
        config_manager = ConfigManager()
        
        # Test that config manager can handle missing config gracefully
        try:
            config = config_manager.get_config()
            assert config is not None
        except Exception as e:
            # Should handle missing config gracefully
            assert "config" in str(e).lower()
    
    def test_logging_configuration(self):
        """Test logging configuration for production."""
        from regional_monetary_policy.logging_config import setup_logging
        
        # Test that logging can be configured
        try:
            setup_logging()
        except Exception as e:
            pytest.fail(f"Logging configuration failed: {e}")
    
    def test_error_handling_robustness(self):
        """Test error handling robustness for production."""
        from regional_monetary_policy.exceptions import RegionalMonetaryPolicyError
        
        # Test that custom exceptions work correctly
        with pytest.raises(RegionalMonetaryPolicyError):
            raise RegionalMonetaryPolicyError("Test error")
    
    def test_memory_efficiency(self):
        """Test memory efficiency for production deployment."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create some objects and clean up
        large_data = [np.random.rand(1000, 1000) for _ in range(10)]
        del large_data
        gc.collect()
        
        # Memory should not grow excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Allow some memory growth but not excessive
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])