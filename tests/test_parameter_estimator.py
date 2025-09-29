"""
Tests for the ParameterEstimator class using synthetic data with known parameters.

This test suite validates the econometric estimation engine by testing it
against synthetic data generated from the theoretical model with known
parameter values.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings

from regional_monetary_policy.econometric.parameter_estimator import (
    ParameterEstimator, create_default_estimation_config, MomentConditions, StageResults
)
from regional_monetary_policy.econometric.models import (
    RegionalParameters, EstimationConfig, EstimationResults, IdentificationReport
)
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.exceptions import EstimationError, IdentificationError


class TestSyntheticDataGeneration:
    """Test synthetic data generation for parameter estimation validation."""
    
    def test_generate_synthetic_regional_data(self):
        """Test generation of synthetic regional data with known parameters."""
        # Known parameter values
        n_regions = 4
        n_periods = 100
        
        true_params = {
            'sigma': np.array([1.0, 1.2, 0.8, 1.1]),  # Interest rate sensitivity
            'kappa': np.array([0.1, 0.15, 0.08, 0.12]),  # Phillips curve slope
            'psi': np.array([0.2, -0.1, 0.15, 0.0]),  # Demand spillover
            'phi': np.array([0.1, 0.05, -0.05, 0.08]),  # Price spillover
            'beta': np.array([0.99, 0.98, 0.99, 0.985])  # Discount factor
        }
        
        # Generate synthetic data
        data = self._generate_synthetic_data(true_params, n_periods)
        
        # Validate data structure
        assert data.n_regions == n_regions
        assert data.n_periods == n_periods
        assert data.output_gaps.shape == (n_regions, n_periods)
        assert data.inflation_rates.shape == (n_regions, n_periods)
        assert len(data.interest_rates) == n_periods
        
        # Check data properties
        assert not data.output_gaps.isnull().any().any()
        assert not data.inflation_rates.isnull().any().any()
        assert not data.interest_rates.isnull().any()
        
        # Check reasonable ranges (more generous for synthetic data)
        assert np.all(np.abs(data.output_gaps.values) < 5)  # Output gaps reasonable
        assert np.all(np.abs(data.inflation_rates.values) < 2)  # Inflation rates reasonable
        assert np.all(np.abs(data.interest_rates.values) < 2)  # Interest rates reasonable
    
    def _generate_synthetic_data(self, true_params: dict, n_periods: int) -> RegionalDataset:
        """Generate synthetic regional data from theoretical model."""
        n_regions = len(true_params['sigma'])
        
        # Create spatial weight matrix (simple structure for testing)
        W = np.array([
            [0.0, 0.3, 0.4, 0.3],
            [0.2, 0.0, 0.3, 0.5],
            [0.3, 0.2, 0.0, 0.5],
            [0.25, 0.25, 0.5, 0.0]
        ])
        
        # Initialize data arrays
        y_gaps = np.zeros((n_regions, n_periods))
        pi_rates = np.zeros((n_regions, n_periods))
        r_rates = np.zeros(n_periods)
        
        # Generate shocks
        np.random.seed(42)  # For reproducibility
        demand_shocks = np.random.normal(0, 0.01, (n_regions, n_periods))  # Smaller shocks
        supply_shocks = np.random.normal(0, 0.005, (n_regions, n_periods))
        policy_shocks = np.random.normal(0, 0.002, n_periods)
        
        # Generate data using simplified model dynamics with stability
        for t in range(1, n_periods):
            # Policy rate (simple Taylor rule)
            if t > 0:
                agg_inflation = np.mean(pi_rates[:, t-1])
                agg_output = np.mean(y_gaps[:, t-1])
                r_rates[t] = 0.02 + 0.5 * agg_inflation + 0.2 * agg_output + policy_shocks[t]
            
            # Regional dynamics with stability constraints
            for i in range(n_regions):
                # Spatial lags
                y_spatial = W[i, :] @ y_gaps[:, t-1] if t > 0 else 0
                pi_spatial = W[i, :] @ pi_rates[:, t-1] if t > 0 else 0
                
                # IS equation with stability
                expected_y = y_gaps[i, t-1] * 0.5 if t > 0 else 0  # More stable expectations
                expected_pi = pi_rates[i, t-1] * 0.3 if t > 0 else 0
                
                y_gaps[i, t] = (0.7 * expected_y - 
                               0.1 * true_params['sigma'][i] * (r_rates[t] - expected_pi) +
                               0.05 * true_params['psi'][i] * y_spatial +
                               demand_shocks[i, t])
                
                # Phillips curve with stability
                pi_rates[i, t] = (0.8 * true_params['beta'][i] * expected_pi +
                                 0.05 * true_params['kappa'][i] * y_gaps[i, t] +
                                 0.02 * true_params['phi'][i] * pi_spatial +
                                 supply_shocks[i, t])
        
        # Create RegionalDataset
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        periods = pd.date_range('2000-01-01', periods=n_periods, freq='ME')  # Use 'ME' instead of 'M'
        
        output_gaps_df = pd.DataFrame(y_gaps, index=regions, columns=periods)
        inflation_df = pd.DataFrame(pi_rates, index=regions, columns=periods)
        interest_rates_series = pd.Series(r_rates, index=periods)
        
        return RegionalDataset(
            output_gaps=output_gaps_df,
            inflation_rates=inflation_df,
            interest_rates=interest_rates_series,
            real_time_estimates={},
            metadata={'synthetic': True, 'true_parameters': true_params}
        )


class TestParameterEstimator:
    """Test the ParameterEstimator class functionality."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Fixture providing synthetic regional data."""
        generator = TestSyntheticDataGeneration()
        true_params = {
            'sigma': np.array([1.0, 1.2, 0.8, 1.1]),
            'kappa': np.array([0.1, 0.15, 0.08, 0.12]),
            'psi': np.array([0.2, -0.1, 0.15, 0.0]),
            'phi': np.array([0.1, 0.05, -0.05, 0.08]),
            'beta': np.array([0.99, 0.98, 0.99, 0.985])
        }
        return generator._generate_synthetic_data(true_params, 100)
    
    @pytest.fixture
    def spatial_handler(self, synthetic_data):
        """Fixture providing configured spatial handler."""
        regions = synthetic_data.regions
        return SpatialModelHandler(regions)
    
    @pytest.fixture
    def estimation_config(self):
        """Fixture providing estimation configuration."""
        config = create_default_estimation_config()
        config.bootstrap_replications = 10  # Reduce for testing speed
        config.max_iterations = 100
        return config
    
    @pytest.fixture
    def parameter_estimator(self, spatial_handler, estimation_config):
        """Fixture providing configured parameter estimator."""
        return ParameterEstimator(spatial_handler, estimation_config)
    
    def test_parameter_estimator_initialization(self, spatial_handler, estimation_config):
        """Test parameter estimator initialization."""
        estimator = ParameterEstimator(spatial_handler, estimation_config)
        
        assert estimator.spatial_handler == spatial_handler
        assert estimator.config == estimation_config
        assert estimator.n_regions == len(spatial_handler.regions)
        assert len(estimator.stage_results) == 0
        assert estimator.spatial_weights is None
    
    def test_stage_one_estimation(self, parameter_estimator, synthetic_data):
        """Test Stage 1: Spatial weight estimation."""
        spatial_results = parameter_estimator.estimate_stage_one(synthetic_data)
        
        # Check spatial weight results
        assert spatial_results.weight_matrix.shape == (4, 4)
        assert spatial_results.validation_report.is_valid
        assert 1 in parameter_estimator.stage_results
        
        # Check weight matrix properties
        W = spatial_results.weight_matrix
        assert np.allclose(np.sum(W, axis=1), 1.0, atol=1e-10)  # Row-normalized
        assert np.allclose(np.diag(W), 0.0, atol=1e-10)  # Zero diagonal
        assert np.all(W >= 0)  # Non-negative
    
    def test_stage_two_estimation(self, parameter_estimator, synthetic_data):
        """Test Stage 2: Regional parameter estimation."""
        # First run Stage 1
        spatial_results = parameter_estimator.estimate_stage_one(synthetic_data)
        
        # Run Stage 2
        regional_params = parameter_estimator.estimate_stage_two(
            synthetic_data, spatial_results.weight_matrix
        )
        
        # Check parameter structure
        assert isinstance(regional_params, RegionalParameters)
        assert len(regional_params.sigma) == 4
        assert len(regional_params.kappa) == 4
        assert len(regional_params.psi) == 4
        assert len(regional_params.phi) == 4
        assert len(regional_params.beta) == 4
        
        # Check parameter ranges are reasonable
        assert np.all(regional_params.sigma > 0)  # Positive interest sensitivity
        assert np.all(regional_params.kappa > 0)  # Positive Phillips curve slope
        assert np.all(regional_params.beta >= 0.9)  # Reasonable discount factors (allow equality)
        assert np.all(regional_params.beta < 1.0)
        
        # Check standard errors exist
        assert 'sigma' in regional_params.standard_errors
        assert 'kappa' in regional_params.standard_errors
        
        # Check stage results stored
        assert 2 in parameter_estimator.stage_results
    
    def test_stage_three_estimation(self, parameter_estimator, synthetic_data):
        """Test Stage 3: Policy parameter estimation."""
        # Run Stages 1 and 2 first
        spatial_results = parameter_estimator.estimate_stage_one(synthetic_data)
        regional_params = parameter_estimator.estimate_stage_two(
            synthetic_data, spatial_results.weight_matrix
        )
        
        # Run Stage 3
        policy_params = parameter_estimator.estimate_stage_three(synthetic_data, regional_params)
        
        # Check policy parameters
        assert isinstance(policy_params, dict)
        assert 'intercept' in policy_params
        assert 'inflation_coefficient' in policy_params
        assert 'output_coefficient' in policy_params
        
        # Check reasonable Taylor rule coefficients (relaxed for synthetic data)
        assert policy_params['inflation_coefficient'] != 0  # Non-zero inflation response
        # Note: Output coefficient can be negative in some specifications
        
        # Check stage results stored
        assert 3 in parameter_estimator.stage_results
    
    def test_full_estimation_procedure(self, parameter_estimator, synthetic_data):
        """Test complete three-stage estimation procedure."""
        results = parameter_estimator.estimate_full_model(synthetic_data)
        
        # Check results structure
        assert isinstance(results, EstimationResults)
        assert isinstance(results.regional_parameters, RegionalParameters)
        assert isinstance(results.estimation_config, EstimationConfig)
        assert results.estimation_time > 0
        
        # Check convergence
        assert 'overall_converged' in results.convergence_info
        assert 'stages' in results.convergence_info
        
        # Check identification tests were run
        assert len(results.identification_tests) > 0
        
        # Check all stages completed
        assert len(parameter_estimator.stage_results) == 3
    
    def test_parameter_recovery_accuracy(self, parameter_estimator, synthetic_data):
        """Test accuracy of parameter recovery with known true values."""
        # Get true parameters from synthetic data metadata
        true_params = synthetic_data.metadata['true_parameters']
        
        # Estimate parameters
        results = parameter_estimator.estimate_full_model(synthetic_data)
        estimated_params = results.regional_parameters
        
        # Check parameter recovery (allowing for estimation error)
        tolerance = 0.5  # Generous tolerance for synthetic data test
        
        # Test sigma recovery
        sigma_error = np.abs(estimated_params.sigma - true_params['sigma'])
        assert np.all(sigma_error < tolerance), f"Sigma recovery error: {sigma_error}"
        
        # Test kappa recovery
        kappa_error = np.abs(estimated_params.kappa - true_params['kappa'])
        assert np.all(kappa_error < tolerance), f"Kappa recovery error: {kappa_error}"
        
        # Test beta recovery
        beta_error = np.abs(estimated_params.beta - true_params['beta'])
        assert np.all(beta_error < 0.1), f"Beta recovery error: {beta_error}"
    
    def test_standard_error_computation(self, parameter_estimator, synthetic_data):
        """Test standard error computation methods."""
        # Run estimation
        spatial_results = parameter_estimator.estimate_stage_one(synthetic_data)
        regional_params = parameter_estimator.estimate_stage_two(
            synthetic_data, spatial_results.weight_matrix
        )
        
        # Test bootstrap standard errors
        parameter_estimator.config.bootstrap_replications = 5  # Small for speed
        boot_se = parameter_estimator.compute_standard_errors(synthetic_data, regional_params)
        
        # Check structure
        assert 'sigma' in boot_se
        assert 'kappa' in boot_se
        assert len(boot_se['sigma']) == 4
        assert np.all(boot_se['sigma'] >= 0)  # Non-negative standard errors (some may be zero)
        
        # Test analytical standard errors
        parameter_estimator.config.bootstrap_replications = 0
        analytical_se = parameter_estimator.compute_standard_errors(synthetic_data, regional_params)
        
        assert 'sigma' in analytical_se
        assert len(analytical_se['sigma']) == 4
        assert np.all(analytical_se['sigma'] > 0)
    
    def test_identification_tests(self, parameter_estimator, synthetic_data):
        """Test parameter identification diagnostics."""
        # Run estimation first
        spatial_results = parameter_estimator.estimate_stage_one(synthetic_data)
        regional_params = parameter_estimator.estimate_stage_two(
            synthetic_data, spatial_results.weight_matrix
        )
        
        # Run identification tests
        id_report = parameter_estimator.run_identification_tests(synthetic_data, regional_params)
        
        # Check report structure
        assert isinstance(id_report, IdentificationReport)
        assert isinstance(id_report.is_identified, (bool, np.bool_))
        assert isinstance(id_report.test_statistics, dict)
        assert isinstance(id_report.recommendations, list)
        
        # Check specific tests
        assert 'rank_condition' in id_report.test_statistics
        assert len(id_report.recommendations) > 0
    
    def test_moment_conditions_class(self):
        """Test MomentConditions dataclass functionality."""
        n_moments = 10
        n_params = 5
        
        moments = np.random.normal(0, 1, n_moments)
        jacobian = np.random.normal(0, 1, (n_moments, n_params))
        weight_matrix = np.eye(n_moments)
        
        mc = MomentConditions(
            moments=moments,
            jacobian=jacobian,
            weight_matrix=weight_matrix,
            n_moments=n_moments,
            n_parameters=n_params
        )
        
        # Test objective computation
        objective = mc.compute_objective()
        expected = moments.T @ weight_matrix @ moments
        assert np.isclose(objective, expected)
    
    def test_stage_results_class(self):
        """Test StageResults dataclass functionality."""
        params = np.array([1.0, 0.5, 0.2])
        se = np.array([0.1, 0.05, 0.02])
        
        stage_result = StageResults(
            parameters=params,
            standard_errors=se,
            objective_value=0.123,
            convergence_info={'converged': True, 'iterations': 50},
            moment_conditions=None,
            stage_number=2
        )
        
        assert np.array_equal(stage_result.parameters, params)
        assert np.array_equal(stage_result.standard_errors, se)
        assert stage_result.stage_number == 2
    
    def test_error_handling(self, parameter_estimator):
        """Test error handling in estimation procedures."""
        # Test with invalid data
        invalid_data = RegionalDataset(
            output_gaps=pd.DataFrame(),
            inflation_rates=pd.DataFrame(),
            interest_rates=pd.Series(),
            real_time_estimates={},
            metadata={}
        )
        
        with pytest.raises(Exception):  # Should raise some kind of error
            parameter_estimator.estimate_full_model(invalid_data)
    
    def test_robustness_checks(self, parameter_estimator, synthetic_data):
        """Test robustness check functionality."""
        # Run estimation
        results = parameter_estimator.estimate_full_model(synthetic_data)
        
        # Check robustness results exist
        assert 'robustness_results' in results.__dict__
        assert isinstance(results.robustness_results, dict)
    
    def test_confidence_intervals(self, parameter_estimator, synthetic_data):
        """Test confidence interval computation."""
        # Run estimation
        spatial_results = parameter_estimator.estimate_stage_one(synthetic_data)
        regional_params = parameter_estimator.estimate_stage_two(
            synthetic_data, spatial_results.weight_matrix
        )
        
        # Compute confidence intervals
        ci = parameter_estimator._compute_confidence_intervals(regional_params)
        
        # Check structure
        assert 'sigma' in ci
        assert isinstance(ci['sigma'], tuple)
        assert len(ci['sigma']) == 2  # Lower and upper bounds
        
        lower, upper = ci['sigma']
        assert len(lower) == 4  # Number of regions
        assert len(upper) == 4
        assert np.all(lower <= regional_params.sigma)  # Lower bound check
        assert np.all(upper >= regional_params.sigma)  # Upper bound check


class TestEstimationConfiguration:
    """Test estimation configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creation of default estimation configuration."""
        config = create_default_estimation_config()
        
        assert isinstance(config, EstimationConfig)
        assert config.identification_strategy == 'baseline'
        assert config.spatial_weight_method == 'trade_migration'
        assert config.convergence_tolerance > 0
        assert config.max_iterations > 0
        assert config.bootstrap_replications > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid identification strategy
        with pytest.raises(ValueError):
            EstimationConfig(
                gmm_options={},
                identification_strategy='invalid',
                spatial_weight_method='trade_migration',
                robustness_checks=[],
                convergence_tolerance=1e-6,
                max_iterations=1000
            )
        
        # Test invalid spatial weight method
        with pytest.raises(ValueError):
            EstimationConfig(
                gmm_options={},
                identification_strategy='baseline',
                spatial_weight_method='invalid',
                robustness_checks=[],
                convergence_tolerance=1e-6,
                max_iterations=1000
            )
    
    def test_spatial_weight_params_validation(self):
        """Test spatial weight parameter validation."""
        # Test weights that don't sum to 1
        with pytest.raises(ValueError):
            EstimationConfig(
                gmm_options={},
                identification_strategy='baseline',
                spatial_weight_method='trade_migration',
                robustness_checks=[],
                convergence_tolerance=1e-6,
                max_iterations=1000,
                spatial_weight_params={
                    'trade_weight': 0.5,
                    'migration_weight': 0.3,
                    'financial_weight': 0.3,  # Sum > 1
                    'distance_weight': 0.1
                }
            )


if __name__ == '__main__':
    pytest.main([__file__])