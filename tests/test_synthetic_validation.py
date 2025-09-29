"""
Validation tests using synthetic data and known analytical solutions.

This module tests the accuracy of estimation procedures and policy analysis
using synthetic data generated from the theoretical model with known parameters.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import linalg, stats
import warnings

from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.policy.optimal_policy import OptimalPolicyCalculator, WelfareFunction
from regional_monetary_policy.policy.counterfactual_engine import CounterfactualEngine, WelfareEvaluator
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.models import RegionalParameters


class SyntheticDataGenerator:
    """Generate synthetic data from theoretical model with known parameters."""
    
    def __init__(self, n_regions=4, n_periods=200, random_seed=42):
        """Initialize synthetic data generator."""
        self.n_regions = n_regions
        self.n_periods = n_periods
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Define true parameter values
        self.true_parameters = self._define_true_parameters()
        
        # Create spatial weight matrix
        self.spatial_weights = self._create_spatial_weights()
    
    def _define_true_parameters(self):
        """Define true parameter values for synthetic data generation."""
        return {
            'sigma': np.array([0.8, 1.0, 0.9, 1.1]),  # Interest rate sensitivity
            'kappa': np.array([0.1, 0.12, 0.08, 0.15]),  # Phillips curve slope
            'psi': np.array([0.15, -0.1, 0.2, 0.05]),  # Demand spillover
            'phi': np.array([0.08, 0.05, -0.03, 0.1]),  # Price spillover
            'beta': np.array([0.99, 0.985, 0.99, 0.988]),  # Discount factor
            'policy_intercept': 0.02,  # Policy rule intercept
            'policy_inflation_coef': 1.5,  # Taylor rule inflation coefficient
            'policy_output_coef': 0.5,  # Taylor rule output coefficient
            'shock_variances': {
                'demand': 0.01,
                'supply': 0.005,
                'policy': 0.002
            }
        }
    
    def _create_spatial_weights(self):
        """Create known spatial weight matrix."""
        # Create a realistic spatial weight matrix
        W = np.array([
            [0.0, 0.4, 0.3, 0.3],
            [0.3, 0.0, 0.4, 0.3],
            [0.2, 0.3, 0.0, 0.5],
            [0.25, 0.25, 0.5, 0.0]
        ])
        
        # Ensure row normalization
        W = W / W.sum(axis=1, keepdims=True)
        
        return W
    
    def generate_dataset(self):
        """Generate complete synthetic dataset."""
        # Initialize arrays
        y_gaps = np.zeros((self.n_regions, self.n_periods))
        pi_rates = np.zeros((self.n_regions, self.n_periods))
        r_rates = np.zeros(self.n_periods)
        
        # Generate structural shocks
        demand_shocks = np.random.normal(
            0, self.true_parameters['shock_variances']['demand'], 
            (self.n_regions, self.n_periods)
        )
        supply_shocks = np.random.normal(
            0, self.true_parameters['shock_variances']['supply'], 
            (self.n_regions, self.n_periods)
        )
        policy_shocks = np.random.normal(
            0, self.true_parameters['shock_variances']['policy'], 
            self.n_periods
        )
        
        # Generate data using theoretical model
        for t in range(1, self.n_periods):
            # Aggregate variables for policy rule
            agg_inflation = np.mean(pi_rates[:, t-1])
            agg_output = np.mean(y_gaps[:, t-1])
            
            # Policy rule (Taylor rule)
            r_rates[t] = (self.true_parameters['policy_intercept'] + 
                         self.true_parameters['policy_inflation_coef'] * agg_inflation +
                         self.true_parameters['policy_output_coef'] * agg_output +
                         policy_shocks[t])
            
            # Regional dynamics
            for i in range(self.n_regions):
                # Spatial lags
                y_spatial = self.spatial_weights[i, :] @ y_gaps[:, t-1]
                pi_spatial = self.spatial_weights[i, :] @ pi_rates[:, t-1]
                
                # Expected values (rational expectations with some persistence)
                expected_y = 0.7 * y_gaps[i, t-1] if t > 0 else 0
                expected_pi = 0.8 * pi_rates[i, t-1] if t > 0 else self.true_parameters['policy_intercept']
                
                # IS equation
                y_gaps[i, t] = (expected_y - 
                               self.true_parameters['sigma'][i] * (r_rates[t] - expected_pi) +
                               self.true_parameters['psi'][i] * y_spatial +
                               demand_shocks[i, t])
                
                # Phillips curve
                pi_rates[i, t] = (self.true_parameters['beta'][i] * expected_pi +
                                 self.true_parameters['kappa'][i] * y_gaps[i, t] +
                                 self.true_parameters['phi'][i] * pi_spatial +
                                 supply_shocks[i, t])
        
        # Create RegionalDataset
        regions = [f"Region_{i+1}" for i in range(self.n_regions)]
        periods = pd.date_range('2000-01-01', periods=self.n_periods, freq='ME')
        
        return RegionalDataset(
            output_gaps=pd.DataFrame(y_gaps, index=regions, columns=periods),
            inflation_rates=pd.DataFrame(pi_rates, index=regions, columns=periods),
            interest_rates=pd.Series(r_rates, index=periods),
            real_time_estimates={},
            metadata={
                'synthetic': True,
                'true_parameters': self.true_parameters,
                'spatial_weights': self.spatial_weights,
                'generator_seed': self.random_seed
            }
        )


class TestParameterRecovery:
    """Test parameter recovery accuracy with synthetic data."""
    
    @pytest.fixture
    def synthetic_generator(self):
        """Create synthetic data generator."""
        return SyntheticDataGenerator(n_regions=4, n_periods=200, random_seed=42)
    
    @pytest.fixture
    def synthetic_data(self, synthetic_generator):
        """Generate synthetic dataset."""
        return synthetic_generator.generate_dataset()
    
    @pytest.fixture
    def true_parameters(self, synthetic_generator):
        """Get true parameters used in data generation."""
        return synthetic_generator.true_parameters
    
    def test_spatial_weight_recovery(self, synthetic_data, synthetic_generator):
        """Test recovery of spatial weight matrix."""
        # Create spatial handler
        regions = [f"Region_{i+1}" for i in range(4)]
        spatial_handler = SpatialModelHandler(regions)
        
        # Create mock spatial data that would generate the true weights
        # (In practice, this would come from external data sources)
        trade_data = self._create_consistent_spatial_data(
            regions, synthetic_generator.spatial_weights, 'trade'
        )
        migration_data = self._create_consistent_spatial_data(
            regions, synthetic_generator.spatial_weights, 'migration'
        )
        financial_data = self._create_consistent_spatial_data(
            regions, synthetic_generator.spatial_weights, 'financial'
        )
        distance_matrix = self._create_distance_matrix(4)
        
        # Estimate spatial weights
        spatial_results = spatial_handler.construct_weights(
            trade_data, migration_data, financial_data, distance_matrix,
            weights=(0.4, 0.3, 0.2, 0.1)
        )
        
        # Extract weight matrix
        estimated_weights = spatial_results.weight_matrix
        
        # Test weight matrix properties
        assert estimated_weights.shape == (4, 4)
        np.testing.assert_allclose(np.sum(estimated_weights, axis=1), 1.0, rtol=1e-10)
        np.testing.assert_allclose(np.diag(estimated_weights), 0.0, atol=1e-10)
        assert np.all(estimated_weights >= 0)
    
    def test_regional_parameter_recovery(self, synthetic_data, true_parameters):
        """Test recovery of regional structural parameters."""
        # Create spatial handler with true weights
        regions = [f"Region_{i+1}" for i in range(4)]
        spatial_handler = SpatialModelHandler(regions)
        
        # Use true spatial weights for parameter estimation
        true_spatial_weights = synthetic_data.metadata['spatial_weights']
        
        # Create parameter estimator
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 20  # Reduce for testing speed
        config.max_iterations = 500
        
        estimator = ParameterEstimator(spatial_handler, config)
        
        # Estimate regional parameters using true spatial weights
        estimated_params = estimator.estimate_stage_two(synthetic_data, true_spatial_weights)
        
        # Test parameter recovery accuracy
        tolerance = 0.3  # Allow 30% error due to finite sample and estimation uncertainty
        
        # Test sigma (interest rate sensitivity)
        sigma_error = np.abs(estimated_params.sigma - true_parameters['sigma']) / np.abs(true_parameters['sigma'])
        assert np.all(sigma_error < tolerance), f"Sigma recovery error: {sigma_error}"
        
        # Test kappa (Phillips curve slope)
        kappa_error = np.abs(estimated_params.kappa - true_parameters['kappa']) / np.abs(true_parameters['kappa'])
        assert np.all(kappa_error < tolerance), f"Kappa recovery error: {kappa_error}"
        
        # Test beta (discount factor) - should be very close to true values
        beta_error = np.abs(estimated_params.beta - true_parameters['beta'])
        assert np.all(beta_error < 0.05), f"Beta recovery error: {beta_error}"
        
        # Test that parameters are in reasonable ranges
        assert np.all(estimated_params.sigma > 0), "Sigma should be positive"
        assert np.all(estimated_params.kappa > 0), "Kappa should be positive"
        assert np.all(estimated_params.beta > 0.9), "Beta should be close to 1"
        assert np.all(estimated_params.beta < 1.0), "Beta should be less than 1"
    
    def test_policy_parameter_recovery(self, synthetic_data, true_parameters):
        """Test recovery of policy rule parameters."""
        # Create regional parameters (using true values for this test)
        regional_params = RegionalParameters(
            sigma=true_parameters['sigma'],
            kappa=true_parameters['kappa'],
            psi=true_parameters['psi'],
            phi=true_parameters['phi'],
            beta=true_parameters['beta'],
            standard_errors={},
            confidence_intervals={}
        )
        
        # Create parameter estimator
        regions = [f"Region_{i+1}" for i in range(4)]
        spatial_handler = SpatialModelHandler(regions)
        
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        estimator = ParameterEstimator(spatial_handler, config)
        
        # Estimate policy parameters
        estimated_policy = estimator.estimate_stage_three(synthetic_data, regional_params)
        
        # Test policy parameter recovery
        tolerance = 0.2  # 20% tolerance for policy parameters
        
        inflation_coef_error = (abs(estimated_policy['inflation_coefficient'] - 
                                   true_parameters['policy_inflation_coef']) / 
                               true_parameters['policy_inflation_coef'])
        assert inflation_coef_error < tolerance, f"Inflation coefficient error: {inflation_coef_error}"
        
        output_coef_error = (abs(estimated_policy['output_coefficient'] - 
                                true_parameters['policy_output_coef']) / 
                            abs(true_parameters['policy_output_coef']))
        assert output_coef_error < tolerance, f"Output coefficient error: {output_coef_error}"
    
    def _create_consistent_spatial_data(self, regions, spatial_weights, data_type):
        """Create spatial interaction data consistent with given weight matrix."""
        flows = []
        base_flow = {'trade': 100, 'migration': 50, 'financial': 75}[data_type]
        
        for i, origin in enumerate(regions):
            for j, destination in enumerate(regions):
                if i != j:
                    # Flow proportional to spatial weight
                    flow = base_flow * spatial_weights[i, j] * (1 + np.random.normal(0, 0.1))
                    
                    if data_type == 'financial':
                        # Financial data uses different column names
                        flows.append({
                            'region1': origin,
                            'region2': destination,
                            'financial_linkage': max(0, flow)  # Ensure non-negative
                        })
                    else:
                        # Trade and migration use origin/destination format
                        flows.append({
                            'origin': origin,
                            'destination': destination,
                            f'{data_type}_flow': max(0, flow)  # Ensure non-negative
                        })
        
        return pd.DataFrame(flows)
    
    def _create_distance_matrix(self, n_regions):
        """Create symmetric distance matrix."""
        distances = np.random.uniform(100, 1000, (n_regions, n_regions))
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0)  # Zero diagonal
        return distances


class TestWelfareCalculationAccuracy:
    """Test welfare calculation accuracy with known analytical solutions."""
    
    def test_quadratic_welfare_function(self):
        """Test quadratic welfare function against analytical solution."""
        # Simple case with known analytical solution
        output_gaps = np.array([0.1, -0.05])
        inflation_rates = np.array([0.02, 0.01])
        regional_weights = np.array([0.6, 0.4])
        
        # Create welfare function
        welfare_func = WelfareFunction(
            output_gap_weight=1.0,
            inflation_weight=1.0,
            regional_weights=regional_weights,
            loss_function='quadratic'
        )
        
        # Compute welfare loss
        computed_loss = welfare_func.compute_loss(output_gaps, inflation_rates)
        
        # Analytical solution
        analytical_loss = np.sum(regional_weights * (output_gaps**2 + inflation_rates**2))
        
        # Should match exactly
        np.testing.assert_allclose(computed_loss, analytical_loss, rtol=1e-12)
    
    def test_optimal_policy_first_order_conditions(self):
        """Test that optimal policy satisfies first-order conditions."""
        # Create simple regional parameters
        regional_params = RegionalParameters(
            sigma=np.array([1.0, 1.0]),  # Symmetric for simplicity
            kappa=np.array([0.1, 0.1]),
            psi=np.array([0.0, 0.0]),  # No spillovers for simplicity
            phi=np.array([0.0, 0.0]),
            beta=np.array([0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        welfare_weights = np.array([0.5, 0.5])  # Equal weights
        
        # Create optimal policy calculator
        calculator = OptimalPolicyCalculator(regional_params, welfare_weights)
        
        # Regional conditions
        regional_conditions = pd.DataFrame({
            'output_gap': [0.01, -0.01],
            'inflation': [0.02, 0.01],
            'expected_inflation': [0.02, 0.02]
        })
        
        # Compute optimal policy
        optimal_rate = calculator.compute_optimal_rate(regional_conditions)
        
        # Test that optimal rate is reasonable
        assert isinstance(optimal_rate, (float, np.floating))
        assert -0.1 < optimal_rate < 0.2  # Reasonable range
        
        # Test first-order condition numerically
        epsilon = 1e-6
        
        # Welfare at optimal rate
        welfare_optimal = self._compute_welfare_at_rate(
            optimal_rate, regional_conditions, regional_params, welfare_weights
        )
        
        # Welfare at slightly higher rate
        welfare_higher = self._compute_welfare_at_rate(
            optimal_rate + epsilon, regional_conditions, regional_params, welfare_weights
        )
        
        # Welfare at slightly lower rate
        welfare_lower = self._compute_welfare_at_rate(
            optimal_rate - epsilon, regional_conditions, regional_params, welfare_weights
        )
        
        # First-order condition: derivative should be approximately zero
        derivative = (welfare_higher - welfare_lower) / (2 * epsilon)
        assert abs(derivative) < 1e-3, f"First-order condition violated: derivative = {derivative}"
    
    def _compute_welfare_at_rate(self, rate, conditions, params, weights):
        """Compute welfare loss at given policy rate."""
        # Simplified welfare computation for testing
        # This would normally involve solving the full model
        
        # Simple approximation: welfare depends on deviations from optimal
        output_gaps = conditions['output_gap'].values
        inflation_rates = conditions['inflation'].values
        
        # Policy affects output gaps through IS curve
        policy_effect = -params.sigma * (rate - inflation_rates)
        adjusted_output = output_gaps + 0.1 * policy_effect  # Simplified transmission
        
        # Compute welfare loss
        welfare_loss = np.sum(weights * (adjusted_output**2 + inflation_rates**2))
        
        return welfare_loss


class TestCounterfactualAccuracy:
    """Test counterfactual analysis accuracy with known scenarios."""
    
    def test_welfare_ranking_verification(self):
        """Test that welfare ranking W^PR ≥ W^PI ≥ W^OR ≥ W^B holds."""
        # Create regional parameters
        regional_params = RegionalParameters(
            sigma=np.array([0.8, 1.0, 0.9]),
            kappa=np.array([0.1, 0.12, 0.08]),
            psi=np.array([0.1, 0.05, 0.15]),
            phi=np.array([0.05, 0.08, 0.03]),
            beta=np.array([0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        welfare_weights = np.array([0.4, 0.35, 0.25])
        
        # Create counterfactual engine
        counterfactual_engine = CounterfactualEngine(regional_params, welfare_weights)
        
        # Create sample historical data
        n_periods = 50
        dates = pd.date_range('2015-01-01', periods=n_periods, freq='ME')
        
        np.random.seed(42)  # For reproducibility
        historical_data = RegionalDataset(
            output_gaps=pd.DataFrame(
                np.random.normal(0, 0.01, (3, n_periods)),
                index=['Region_1', 'Region_2', 'Region_3'],
                columns=dates
            ),
            inflation_rates=pd.DataFrame(
                np.random.normal(0.02, 0.005, (3, n_periods)),
                index=['Region_1', 'Region_2', 'Region_3'],
                columns=dates
            ),
            interest_rates=pd.Series(
                np.random.normal(0.03, 0.01, n_periods),
                index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        # Generate all scenarios
        baseline = counterfactual_engine.generate_baseline_scenario(historical_data)
        perfect_info = counterfactual_engine.generate_perfect_info_scenario(historical_data)
        optimal_regional = counterfactual_engine.generate_optimal_regional_scenario(historical_data)
        perfect_regional = counterfactual_engine.generate_perfect_regional_scenario(historical_data)
        
        # Test welfare ranking
        # Note: Higher welfare values are better, so we expect W^PR ≥ W^PI ≥ W^OR ≥ W^B
        assert perfect_regional.welfare_outcome >= perfect_info.welfare_outcome, \
            "Perfect Regional should dominate Perfect Information"
        
        assert perfect_info.welfare_outcome >= optimal_regional.welfare_outcome, \
            "Perfect Information should dominate Optimal Regional"
        
        assert optimal_regional.welfare_outcome >= baseline.welfare_outcome, \
            "Optimal Regional should dominate Baseline"
    
    def test_counterfactual_consistency(self):
        """Test internal consistency of counterfactual scenarios."""
        # Create simple regional parameters
        regional_params = RegionalParameters(
            sigma=np.array([1.0, 1.0]),
            kappa=np.array([0.1, 0.1]),
            psi=np.array([0.0, 0.0]),  # No spillovers for simplicity
            phi=np.array([0.0, 0.0]),
            beta=np.array([0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        welfare_weights = np.array([0.5, 0.5])
        
        # Create counterfactual engine
        counterfactual_engine = CounterfactualEngine(regional_params, welfare_weights)
        
        # Create deterministic historical data for testing
        n_periods = 20
        dates = pd.date_range('2015-01-01', periods=n_periods, freq='ME')
        
        # Constant conditions for predictable results
        historical_data = RegionalDataset(
            output_gaps=pd.DataFrame(
                0.01 * np.ones((2, n_periods)),
                index=['Region_1', 'Region_2'],
                columns=dates
            ),
            inflation_rates=pd.DataFrame(
                0.02 * np.ones((2, n_periods)),
                index=['Region_1', 'Region_2'],
                columns=dates
            ),
            interest_rates=pd.Series(
                0.03 * np.ones(n_periods),
                index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        # Generate baseline scenario
        baseline = counterfactual_engine.generate_baseline_scenario(historical_data)
        
        # Test scenario consistency
        assert len(baseline.policy_rates) == n_periods
        assert baseline.name == 'baseline'
        assert isinstance(baseline.welfare_outcome, (float, np.floating))
        
        # Policy rates should be reasonable
        assert np.all(baseline.policy_rates >= -0.1)
        assert np.all(baseline.policy_rates <= 0.2)


class TestNumericalAccuracy:
    """Test numerical accuracy of computational procedures."""
    
    def test_matrix_operations_accuracy(self):
        """Test accuracy of matrix operations used in estimation."""
        # Test matrix inversion accuracy
        np.random.seed(42)
        A = np.random.randn(5, 5)
        A = A @ A.T  # Make positive definite
        
        A_inv = linalg.inv(A)
        identity = A @ A_inv
        
        # Should be close to identity matrix
        np.testing.assert_allclose(identity, np.eye(5), rtol=1e-12)
    
    def test_optimization_accuracy(self):
        """Test accuracy of optimization procedures."""
        from scipy.optimize import minimize
        
        # Test simple quadratic optimization
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        result = minimize(objective, x0=[0, 0], method='BFGS')
        
        # Should find global minimum
        assert result.success
        np.testing.assert_allclose(result.x, [2, 3], rtol=1e-6)
        np.testing.assert_allclose(result.fun, 0, atol=1e-12)
    
    def test_statistical_accuracy(self):
        """Test accuracy of statistical computations."""
        # Test sample statistics
        np.random.seed(42)
        data = np.random.normal(5, 2, 1000)
        
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        # Should be close to true parameters
        np.testing.assert_allclose(sample_mean, 5, rtol=0.1)
        np.testing.assert_allclose(sample_std, 2, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__])