"""
Tests for configuration and extensibility framework.
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from regional_monetary_policy.config.config_manager import ConfigManager
from regional_monetary_policy.config.extensibility import (
    ExtensibilityConfig, ExtensibilityManager, SpatialWeightMethod, 
    IdentificationStrategy, ModelExtension, RobustnessCheck, 
    ParameterRestriction, TradeBasedWeights, DistanceBasedWeights,
    FinancialFrictionsExtension, FiscalPolicyExtension
)
from regional_monetary_policy.config.version_control import VersionControlManager, AnalysisRun, ConfigVersion


class TestExtensibilityConfig:
    """Test extensibility configuration."""
    
    def test_default_config(self):
        """Test default extensibility configuration."""
        config = ExtensibilityConfig()
        
        assert len(config.spatial_weight_methods) == 1
        assert config.spatial_weight_methods[0] == SpatialWeightMethod.TRADE_MIGRATION
        assert len(config.identification_strategies) == 1
        assert config.identification_strategies[0] == IdentificationStrategy.BASELINE
        assert len(config.model_extensions) == 0
        assert len(config.parameter_restrictions) == 0
        assert len(config.robustness_checks) == 0
    
    def test_add_spatial_weight_method(self):
        """Test adding spatial weight methods."""
        config = ExtensibilityConfig()
        
        # Add built-in method
        config.add_spatial_weight_method(SpatialWeightMethod.DISTANCE_ONLY)
        assert SpatialWeightMethod.DISTANCE_ONLY in config.spatial_weight_methods
        
        # Add custom method
        custom_builder = TradeBasedWeights()
        config.add_spatial_weight_method("custom_trade", custom_builder)
        assert "custom_trade" in config.custom_weight_builders
        assert config.custom_weight_builders["custom_trade"] == custom_builder
    
    def test_add_parameter_restriction(self):
        """Test adding parameter restrictions."""
        config = ExtensibilityConfig()
        
        # Add bounds restriction
        config.add_parameter_restriction("sigma", "bounds", lower_bound=0.0, upper_bound=5.0)
        
        assert len(config.parameter_restrictions) == 1
        restriction = config.parameter_restrictions[0]
        assert restriction.parameter_name == "sigma"
        assert restriction.restriction_type == "bounds"
        assert restriction.lower_bound == 0.0
        assert restriction.upper_bound == 5.0
    
    def test_add_robustness_check(self):
        """Test adding robustness checks."""
        config = ExtensibilityConfig()
        
        config.add_robustness_check(
            "subsample_stability",
            "Test parameter stability across subsamples",
            {"split_date": "2008-01-01"}
        )
        
        assert len(config.robustness_checks) == 1
        check = config.robustness_checks[0]
        assert check.name == "subsample_stability"
        assert check.enabled is True
        assert check.parameters["split_date"] == "2008-01-01"
    
    def test_enable_model_extension(self):
        """Test enabling model extensions."""
        config = ExtensibilityConfig()
        
        config.enable_model_extension(
            ModelExtension.FINANCIAL_FRICTIONS,
            {"include_credit_spreads": True}
        )
        
        assert ModelExtension.FINANCIAL_FRICTIONS in config.model_extensions
        assert config.extension_configs["financial_frictions"]["include_credit_spreads"] is True
    
    def test_validation(self):
        """Test configuration validation."""
        config = ExtensibilityConfig()
        
        # Valid configuration
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid bounds restriction
        config.add_parameter_restriction("sigma", "bounds")  # No bounds specified
        errors = config.validate()
        assert len(errors) == 1
        assert "bounds restriction" in errors[0].lower()
        
        # Invalid equality restriction
        config.parameter_restrictions.clear()
        config.add_parameter_restriction("kappa", "equality")  # No value specified
        errors = config.validate()
        assert len(errors) == 1
        assert "equality restriction" in errors[0].lower()


class TestSpatialWeightBuilders:
    """Test spatial weight builders."""
    
    def test_trade_based_weights(self):
        """Test trade-based spatial weight construction."""
        builder = TradeBasedWeights()
        regions = ["region1", "region2", "region3"]
        
        # Create mock trade flows
        trade_flows = {
            ("region1", "region2"): 100,
            ("region1", "region3"): 50,
            ("region2", "region1"): 80,
            ("region2", "region3"): 120,
            ("region3", "region1"): 60,
            ("region3", "region2"): 90
        }
        
        data = {"trade_flows": trade_flows}
        
        # Build weights
        weights = builder.build_weights(regions, data)
        
        assert weights.shape == (3, 3)
        assert np.allclose(weights.sum(axis=1), 1.0)  # Row normalized
        assert np.all(np.diag(weights) == 0)  # No self-weights
    
    def test_distance_based_weights(self):
        """Test distance-based spatial weight construction."""
        builder = DistanceBasedWeights(decay_parameter=1.0)
        regions = ["region1", "region2", "region3"]
        
        # Create distance matrix
        distances = np.array([
            [0, 100, 200],
            [100, 0, 150],
            [200, 150, 0]
        ])
        
        data = {"distances": distances}
        
        # Build weights
        weights = builder.build_weights(regions, data)
        
        assert weights.shape == (3, 3)
        assert np.allclose(weights.sum(axis=1), 1.0)  # Row normalized
        assert np.all(np.diag(weights) == 0)  # No self-weights
    
    def test_validation(self):
        """Test weight builder validation."""
        builder = TradeBasedWeights()
        regions = ["region1", "region2"]
        
        # Missing trade flows
        errors = builder.validate_inputs(regions, {})
        assert len(errors) == 1
        assert "trade flows" in errors[0].lower()
        
        # Invalid trade flows format
        errors = builder.validate_inputs(regions, {"trade_flows": "invalid"})
        assert len(errors) == 1
        assert "dictionary" in errors[0].lower()


class TestModelExtensions:
    """Test model extensions."""
    
    def test_financial_frictions_extension(self):
        """Test financial frictions extension."""
        extension = FinancialFrictionsExtension(
            include_credit_spreads=True,
            include_bank_lending=True
        )
        
        # Test parameter list
        params = extension.get_additional_parameters()
        assert "rho_cs" in params
        assert "beta_cs" in params
        assert "rho_bl" in params
        assert "delta_cs" in params
        
        # Test equation extension
        base_equations = {
            "is_curve": {
                "equation": "x_t = E_t[x_{t+1}] - sigma * (r_t - E_t[pi_{t+1}])",
                "parameters": ["sigma"]
            }
        }
        
        extended = extension.extend_equations(base_equations)
        
        assert "credit_spread" in extended
        assert "bank_lending" in extended
        assert "delta_cs" in extended["is_curve"]["parameters"]
    
    def test_fiscal_policy_extension(self):
        """Test fiscal policy extension."""
        extension = FiscalPolicyExtension(
            include_government_spending=True,
            include_taxes=True
        )
        
        # Test parameter list
        params = extension.get_additional_parameters()
        assert "rho_g" in params
        assert "phi_g" in params
        assert "rho_tau" in params
        assert "alpha_g" in params
        
        # Test equation extension
        base_equations = {
            "is_curve": {
                "equation": "x_t = E_t[x_{t+1}] - sigma * (r_t - E_t[pi_{t+1}])",
                "parameters": ["sigma"]
            }
        }
        
        extended = extension.extend_equations(base_equations)
        
        assert "government_spending" in extended
        assert "tax_rate" in extended
        assert "alpha_g" in extended["is_curve"]["parameters"]
    
    def test_data_validation(self):
        """Test extension data validation."""
        extension = FinancialFrictionsExtension(
            include_credit_spreads=True,
            include_bank_lending=True
        )
        
        # Missing required data
        errors = extension.validate_extension_data({})
        assert len(errors) == 2
        assert any("credit spreads" in error.lower() for error in errors)
        assert any("bank lending" in error.lower() for error in errors)
        
        # Valid data
        data = {
            "credit_spreads": np.random.randn(100),
            "bank_lending": np.random.randn(100)
        }
        errors = extension.validate_extension_data(data)
        assert len(errors) == 0


class TestExtensibilityManager:
    """Test extensibility manager."""
    
    def test_initialization(self):
        """Test extensibility manager initialization."""
        config = ExtensibilityConfig()
        manager = ExtensibilityManager(config)
        
        assert manager.config == config
        assert len(manager._spatial_weight_registry) > 0
        assert len(manager._extension_registry) > 0
    
    def test_get_spatial_weight_builder(self):
        """Test getting spatial weight builders."""
        config = ExtensibilityConfig()
        manager = ExtensibilityManager(config)
        
        # Get built-in builder
        builder = manager.get_spatial_weight_builder(SpatialWeightMethod.TRADE_ONLY)
        assert isinstance(builder, TradeBasedWeights)
        
        # Get builder with parameters
        builder = manager.get_spatial_weight_builder(
            SpatialWeightMethod.DISTANCE_ONLY,
            decay_parameter=2.0
        )
        assert isinstance(builder, DistanceBasedWeights)
        assert builder.decay_parameter == 2.0
    
    def test_apply_parameter_restrictions(self):
        """Test applying parameter restrictions."""
        config = ExtensibilityConfig()
        config.add_parameter_restriction("sigma", "bounds", lower_bound=0.0, upper_bound=5.0)
        config.add_parameter_restriction("kappa", "equality", value=1.0)
        
        manager = ExtensibilityManager(config)
        
        # Test parameters
        parameters = {
            "sigma": np.array([-1.0, 2.0, 6.0]),
            "kappa": np.array([0.5, 1.5, 2.0]),
            "other": np.array([1.0, 2.0, 3.0])
        }
        
        restricted = manager.apply_parameter_restrictions(parameters)
        
        # Check bounds restriction
        assert np.all(restricted["sigma"] >= 0.0)
        assert np.all(restricted["sigma"] <= 5.0)
        
        # Check equality restriction
        assert np.all(restricted["kappa"] == 1.0)
        
        # Check unrestricted parameter unchanged
        assert np.array_equal(restricted["other"], parameters["other"])
    
    def test_get_model_specification(self):
        """Test getting extended model specification."""
        config = ExtensibilityConfig()
        config.enable_model_extension(ModelExtension.FINANCIAL_FRICTIONS)
        
        manager = ExtensibilityManager(config)
        
        base_spec = {
            "is_curve": {
                "equation": "x_t = E_t[x_{t+1}] - sigma * (r_t - E_t[pi_{t+1}])",
                "parameters": ["sigma"]
            }
        }
        
        extended_spec = manager.get_model_specification(base_spec)
        
        # Should have additional equations from financial frictions
        assert "credit_spread" in extended_spec
        assert "delta_cs" in extended_spec["is_curve"]["parameters"]


class TestVersionControl:
    """Test version control system."""
    
    def test_initialization(self):
        """Test version control manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VersionControlManager(temp_dir)
            
            assert manager.base_dir == Path(temp_dir)
            assert manager.config_dir.exists()
            assert manager.runs_dir.exists()
            assert manager.results_dir.exists()
    
    def test_save_and_load_config_version(self):
        """Test saving and loading configuration versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VersionControlManager(temp_dir)
            
            # Save configuration
            config_data = {
                "analysis_name": "test_analysis",
                "data": {"regions": ["region1", "region2"]},
                "estimation": {"method": "gmm"}
            }
            
            version_id = manager.save_config_version(
                config_data,
                description="Test configuration",
                tags=["test", "baseline"]
            )
            
            assert version_id is not None
            assert version_id in manager.config_versions
            
            # Load configuration
            loaded_config = manager.load_config_version(version_id)
            assert loaded_config == config_data
            
            # Check version metadata
            version = manager.config_versions[version_id]
            assert version.description == "Test configuration"
            assert "test" in version.tags
            assert "baseline" in version.tags
    
    def test_analysis_run_lifecycle(self):
        """Test analysis run lifecycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VersionControlManager(temp_dir)
            
            config_data = {"analysis_name": "test"}
            parameters = {"param1": "value1"}
            
            # Start run
            run_id = manager.start_analysis_run(config_data, parameters)
            assert run_id is not None
            assert run_id in manager.analysis_runs
            
            run = manager.analysis_runs[run_id]
            assert run.status == "running"
            assert run.parameters == parameters
            
            # Complete run
            manager.complete_analysis_run(run_id, execution_time=10.5)
            
            run = manager.analysis_runs[run_id]
            assert run.status == "completed"
            assert run.execution_time == 10.5
            
            # Test failed run
            run_id2 = manager.start_analysis_run(config_data)
            manager.fail_analysis_run(run_id2, "Test error")
            
            run2 = manager.analysis_runs[run_id2]
            assert run2.status == "failed"
            assert run2.error_message == "Test error"
    
    def test_config_comparison(self):
        """Test configuration comparison."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VersionControlManager(temp_dir)
            
            # Save two configurations
            config1 = {"param1": "value1", "param2": {"nested": "old"}}
            config2 = {"param1": "value1", "param2": {"nested": "new"}, "param3": "added"}
            
            version1 = manager.save_config_version(config1, "Config 1")
            version2 = manager.save_config_version(config2, "Config 2")
            
            # Compare configurations
            comparison = manager.compare_configs(version1, version2)
            
            assert comparison["summary"]["total_changes"] == 2
            assert comparison["summary"]["changed"] == 1
            assert comparison["summary"]["added"] == 1
            
            differences = comparison["differences"]
            change_paths = [diff["path"] for diff in differences]
            assert "param2.nested" in change_paths
            assert "param3" in change_paths
    
    def test_reproducibility_report(self):
        """Test reproducibility report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VersionControlManager(temp_dir)
            
            config_data = {"analysis_name": "test"}
            run_id = manager.start_analysis_run(config_data)
            manager.complete_analysis_run(run_id)
            
            report = manager.create_reproducibility_report(run_id)
            
            assert "run_info" in report
            assert "configuration" in report
            assert "reproducibility" in report
            assert report["run_info"]["run_id"] == run_id


class TestConfigManager:
    """Test integrated configuration manager."""
    
    def test_initialization_with_extensibility(self):
        """Test config manager initialization with extensibility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager()
            
            assert config_manager.has_version_control
            assert not config_manager.has_extensibility  # Not loaded by default
    
    def test_load_extensibility_config(self):
        """Test loading extensibility configuration."""
        config_manager = ConfigManager(enable_version_control=False)
        
        # Load default extensibility config
        config_manager.load_extensibility_config()
        
        assert config_manager.has_extensibility
        assert config_manager.extensibility_config is not None
        assert config_manager.extensibility_manager is not None
    
    def test_version_control_integration(self):
        """Test version control integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager()
            
            # Load configuration
            config_manager.load_config()
            
            # Save version
            version_id = config_manager.save_config_version("Test version")
            assert version_id is not None
            
            # Start analysis run
            run_id = config_manager.start_analysis_run({"test_param": "value"})
            assert run_id is not None
            
            # Complete run
            config_manager.complete_analysis_run(run_id)
            
            # Get history
            versions = config_manager.get_config_history()
            assert len(versions) >= 1
            
            runs = config_manager.get_run_history()
            assert len(runs) >= 1
    
    def test_extensibility_integration(self):
        """Test extensibility framework integration."""
        config_manager = ConfigManager(enable_version_control=False)
        config_manager.load_config()
        config_manager.load_extensibility_config()
        
        # Test spatial weight builder
        builder = config_manager.get_spatial_weight_builder("trade_only")
        assert isinstance(builder, TradeBasedWeights)
        
        # Test parameter restrictions
        parameters = {"sigma": np.array([1.0, 2.0, 3.0])}
        restricted = config_manager.apply_parameter_restrictions(parameters)
        assert "sigma" in restricted
        
        # Test model specification
        base_spec = {"is_curve": {"equation": "test", "parameters": []}}
        extended_spec = config_manager.get_model_specification(base_spec)
        assert "is_curve" in extended_spec
    
    def test_configuration_template_export(self):
        """Test configuration template export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(enable_version_control=False)
            
            template_path = Path(temp_dir) / "template.json"
            config_manager.export_configuration_template(str(template_path))
            
            assert template_path.exists()
            
            # Load and validate template
            with open(template_path) as f:
                template = json.load(f)
            
            assert "data" in template
            assert "estimation" in template
            assert "policy" in template
            assert "visualization" in template
            assert "extensibility" in template
            assert "_documentation" in template


if __name__ == "__main__":
    pytest.main([__file__])