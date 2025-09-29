"""
Unit tests for core mathematical operations and algorithms.

This module tests the fundamental mathematical operations used throughout
the regional monetary policy analysis system, including matrix operations,
numerical optimization, and statistical computations.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from unittest.mock import Mock, patch
import warnings

from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.policy.optimal_policy import OptimalPolicyCalculator, WelfareFunction
from regional_monetary_policy.policy.counterfactual_engine import CounterfactualEngine, WelfareEvaluator
from regional_monetary_policy.econometric.models import RegionalParameters
from regional_monetary_policy.exceptions import EstimationError, SpatialModelError


class TestMatrixOperations:
    """Test core matrix operations used in spatial modeling and estimation."""
    
    def test_spatial_weight_matrix_properties(self):
        """Test spatial weight matrix construction and properties."""
        n_regions = 4
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        handler = SpatialModelHandler(regions)
        
        # Create sample interaction data
        trade_data = pd.DataFrame({
            'origin': ['Region_1', 'Region_1', 'Region_2', 'Region_2'],
            'destination': ['Region_2', 'Region_3', 'Region_1', 'Region_4'],
            'trade_flow': [100, 80, 90, 70]
        })
        
        migration_data = pd.DataFrame({
            'origin': ['Region_1', 'Region_2', 'Region_3', 'Region_4'],
            'destination': ['Region_2', 'Region_3', 'Region_4', 'Region_1'],
            'migration_flow': [50, 40, 30, 20]
        })
        
        financial_data = pd.DataFrame({
            'region1': ['Region_1', 'Region_2'],
            'region2': ['Region_3', 'Region_4'],
            'financial_linkage': [60, 45]
        })
        
        # Distance matrix (symmetric)
        distance_matrix = np.array([
            [0, 100, 200, 300],
            [100, 0, 150, 250],
            [200, 150, 0, 100],
            [300, 250, 100, 0]
        ])
        
        # Construct spatial weights
        spatial_results = handler.construct_weights(
            trade_data, migration_data, financial_data, distance_matrix,
            weights=(0.4, 0.3, 0.2, 0.1)
        )
        
        # Extract weight matrix
        W = spatial_results.weight_matrix
        
        # Test matrix properties
        assert W.shape == (n_regions, n_regions)
        
        # Row normalization (each row sums to 1)
        row_sums = np.sum(W, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
        
        # Zero diagonal (no self-interaction)
        np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-10)
        
        # Non-negative elements
        assert np.all(W >= 0)
        
        # Test symmetry properties (not necessarily symmetric, but check structure)
        assert not np.allclose(W, W.T)  # Should not be symmetric due to directional flows
    
    def test_spatial_lag_computation(self):
        """Test spatial lag computation accuracy."""
        n_regions = 3
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        handler = SpatialModelHandler(regions)
        
        # Simple known weight matrix
        W = np.array([
            [0.0, 0.6, 0.4],
            [0.3, 0.0, 0.7],
            [0.5, 0.5, 0.0]
        ])
        
        # Test data
        data = pd.DataFrame({
            'Region_1': [1.0, 2.0, 3.0],
            'Region_2': [4.0, 5.0, 6.0],
            'Region_3': [7.0, 8.0, 9.0]
        })
        
        # Compute spatial lags
        spatial_lags = handler.compute_spatial_lags(data, W)
        
        # Manual calculation for verification
        expected_lag_1 = W[0, 1] * data['Region_2'] + W[0, 2] * data['Region_3']
        expected_lag_2 = W[1, 0] * data['Region_1'] + W[1, 2] * data['Region_3']
        expected_lag_3 = W[2, 0] * data['Region_1'] + W[2, 1] * data['Region_2']
        
        np.testing.assert_allclose(spatial_lags['Region_1'], expected_lag_1)
        np.testing.assert_allclose(spatial_lags['Region_2'], expected_lag_2)
        np.testing.assert_allclose(spatial_lags['Region_3'], expected_lag_3)
    
    def test_matrix_inversion_stability(self):
        """Test numerical stability of matrix inversions."""
        # Test with well-conditioned matrix
        A = np.array([[4, 1], [1, 3]])
        A_inv = linalg.inv(A)
        
        # Check inversion accuracy
        identity = A @ A_inv
        np.testing.assert_allclose(identity, np.eye(2), rtol=1e-12)
        
        # Test with ill-conditioned matrix
        B = np.array([[1, 1], [1, 1.0001]])  # Nearly singular
        cond_num = np.linalg.cond(B)
        
        if cond_num < 1e12:  # Only invert if reasonably conditioned
            B_inv = linalg.inv(B)
            identity_B = B @ B_inv
            np.testing.assert_allclose(identity_B, np.eye(2), rtol=1e-8)
    
    def test_eigenvalue_computations(self):
        """Test eigenvalue computations for spatial matrices."""
        # Symmetric matrix
        A = np.array([[2, 1], [1, 2]])
        eigenvals, eigenvecs = linalg.eigh(A)
        
        # Check eigenvalue equation: A * v = λ * v
        for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
            result = A @ vec
            expected = val * vec
            np.testing.assert_allclose(result, expected, rtol=1e-12)
        
        # Test for spatial weight matrix properties
        W = np.array([[0, 0.6, 0.4], [0.3, 0, 0.7], [0.5, 0.5, 0]])
        eigenvals_W = linalg.eigvals(W)
        
        # Largest eigenvalue should be ≤ 1 for row-stochastic matrix
        max_eigenval = np.max(np.real(eigenvals_W))
        assert max_eigenval <= 1.0 + 1e-10  # Allow small numerical error


class TestNumericalOptimization:
    """Test numerical optimization algorithms used in estimation."""
    
    def test_gmm_objective_function(self):
        """Test GMM objective function computation."""
        # Sample moment conditions
        n_moments = 5
        moments = np.array([0.1, -0.05, 0.02, -0.01, 0.03])
        
        # Identity weight matrix
        W = np.eye(n_moments)
        
        # Compute GMM objective
        objective = moments.T @ W @ moments
        expected = np.sum(moments**2)
        
        np.testing.assert_allclose(objective, expected)
        
        # Test with optimal weight matrix (inverse of moment covariance)
        # Simulate moment covariance
        np.random.seed(42)
        moment_samples = np.random.multivariate_normal(moments, 0.01 * np.eye(n_moments), 100)
        S = np.cov(moment_samples.T)
        
        # Optimal weight matrix
        W_opt = linalg.inv(S)
        objective_opt = moments.T @ W_opt @ moments
        
        # Should be positive
        assert objective_opt >= 0
    
    def test_optimization_convergence(self):
        """Test optimization algorithm convergence."""
        # Simple quadratic function: f(x) = (x - 2)^2 + 1
        def objective(x):
            return (x[0] - 2)**2 + 1
        
        def gradient(x):
            return np.array([2 * (x[0] - 2)])
        
        # Test different starting points
        starting_points = [0.0, 5.0, -3.0]
        
        for x0 in starting_points:
            result = optimize.minimize(
                objective, 
                x0=[x0], 
                jac=gradient, 
                method='BFGS',
                options={'gtol': 1e-8}
            )
            
            # Should converge to x = 2
            assert result.success
            np.testing.assert_allclose(result.x, [2.0], rtol=1e-6)
            np.testing.assert_allclose(result.fun, 1.0, rtol=1e-6)
    
    def test_constrained_optimization(self):
        """Test constrained optimization for parameter bounds."""
        # Minimize x^2 + y^2 subject to x + y = 1
        def objective(params):
            x, y = params
            return x**2 + y**2
        
        # Constraint: x + y = 1
        constraint = {'type': 'eq', 'fun': lambda params: params[0] + params[1] - 1}
        
        result = optimize.minimize(
            objective,
            x0=[0.0, 0.0],
            constraints=constraint,
            method='SLSQP'
        )
        
        assert result.success
        # Analytical solution: x = y = 0.5
        np.testing.assert_allclose(result.x, [0.5, 0.5], rtol=1e-6)
        
        # Test inequality constraints (bounds)
        bounds = [(0, 1), (0, 1)]  # 0 ≤ x, y ≤ 1
        
        result_bounded = optimize.minimize(
            objective,
            x0=[0.1, 0.1],
            constraints=constraint,
            bounds=bounds,
            method='SLSQP'
        )
        
        assert result_bounded.success
        np.testing.assert_allclose(result_bounded.x, [0.5, 0.5], rtol=1e-6)


class TestStatisticalComputations:
    """Test statistical computations and inference procedures."""
    
    def test_covariance_matrix_estimation(self):
        """Test covariance matrix estimation methods."""
        # Generate sample data
        np.random.seed(42)
        n_obs = 100
        n_vars = 3
        
        # True covariance matrix
        true_cov = np.array([[1.0, 0.5, 0.2],
                            [0.5, 2.0, 0.3],
                            [0.2, 0.3, 1.5]])
        
        # Generate data
        data = np.random.multivariate_normal(np.zeros(n_vars), true_cov, n_obs)
        
        # Sample covariance
        sample_cov = np.cov(data.T)
        
        # Should be close to true covariance (with sampling error)
        np.testing.assert_allclose(sample_cov, true_cov, rtol=0.3)  # Allow 30% error
        
        # Test positive definiteness
        eigenvals = linalg.eigvals(sample_cov)
        assert np.all(eigenvals > 0)
    
    def test_bootstrap_procedures(self):
        """Test bootstrap resampling procedures."""
        # Sample data
        np.random.seed(42)
        data = np.random.normal(5, 2, 100)
        
        # Bootstrap function: compute mean
        def bootstrap_statistic(sample):
            return np.mean(sample)
        
        # Perform bootstrap
        n_bootstrap = 1000
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            stat = bootstrap_statistic(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bootstrap mean should be close to sample mean
        sample_mean = np.mean(data)
        bootstrap_mean = np.mean(bootstrap_stats)
        np.testing.assert_allclose(bootstrap_mean, sample_mean, rtol=0.1)
        
        # Bootstrap standard error
        bootstrap_se = np.std(bootstrap_stats)
        theoretical_se = np.std(data) / np.sqrt(len(data))
        np.testing.assert_allclose(bootstrap_se, theoretical_se, rtol=0.2)
    
    def test_hypothesis_testing(self):
        """Test statistical hypothesis testing procedures."""
        # Test t-test
        np.random.seed(42)
        sample1 = np.random.normal(0, 1, 50)
        sample2 = np.random.normal(0.5, 1, 50)  # Different mean
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        
        # Should detect difference (p < 0.05 expected)
        assert p_value < 0.1  # Allow some variability
        
        # Test normality (Shapiro-Wilk)
        normal_data = np.random.normal(0, 1, 50)
        shapiro_stat, shapiro_p = stats.shapiro(normal_data)
        
        # Should not reject normality (p > 0.05 expected)
        assert shapiro_p > 0.01  # Conservative threshold
    
    def test_confidence_intervals(self):
        """Test confidence interval construction."""
        # Sample data
        np.random.seed(42)
        data = np.random.normal(10, 3, 100)
        
        # Sample statistics
        sample_mean = np.mean(data)
        sample_se = stats.sem(data)  # Standard error of mean
        
        # 95% confidence interval
        alpha = 0.05
        df = len(data) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        ci_lower = sample_mean - t_critical * sample_se
        ci_upper = sample_mean + t_critical * sample_se
        
        # True mean (10) should be in confidence interval
        assert ci_lower <= 10 <= ci_upper
        
        # Interval should have reasonable width
        ci_width = ci_upper - ci_lower
        assert 0.5 < ci_width < 2.0  # Reasonable for this sample size


class TestWelfareCalculations:
    """Test welfare function computations and optimization."""
    
    def test_quadratic_loss_function(self):
        """Test quadratic welfare loss computation."""
        # Sample regional outcomes
        output_gaps = np.array([0.1, -0.05, 0.02])
        inflation_rates = np.array([0.02, 0.01, 0.03])
        
        # Welfare weights
        regional_weights = np.array([0.4, 0.35, 0.25])
        output_weight = 1.0
        inflation_weight = 1.0
        
        # Create welfare function
        welfare_func = WelfareFunction(
            output_gap_weight=output_weight,
            inflation_weight=inflation_weight,
            regional_weights=regional_weights,
            loss_function='quadratic'
        )
        
        # Compute welfare loss
        welfare_loss = welfare_func.compute_loss(output_gaps, inflation_rates)
        
        # Manual calculation
        expected_loss = np.sum(regional_weights * (
            output_weight * output_gaps**2 + inflation_weight * inflation_rates**2
        ))
        
        np.testing.assert_allclose(welfare_loss, expected_loss)
        
        # Test properties
        assert welfare_loss >= 0  # Loss should be non-negative
        
        # Zero outcomes should give zero loss
        zero_loss = welfare_func.compute_loss(
            np.zeros_like(output_gaps), 
            np.zeros_like(inflation_rates)
        )
        np.testing.assert_allclose(zero_loss, 0.0)
    
    def test_welfare_optimization(self):
        """Test welfare-maximizing policy computation."""
        # Sample regional parameters
        regional_params = RegionalParameters(
            sigma=np.array([0.5, 0.7, 0.6]),
            kappa=np.array([0.3, 0.4, 0.35]),
            psi=np.array([0.1, 0.15, 0.12]),
            phi=np.array([0.05, 0.08, 0.06]),
            beta=np.array([0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        # Welfare weights
        welfare_weights = np.array([0.4, 0.35, 0.25])
        
        # Create optimal policy calculator
        calculator = OptimalPolicyCalculator(regional_params, welfare_weights)
        
        # Sample regional conditions
        regional_conditions = pd.DataFrame({
            'output_gap': [0.1, -0.05, 0.02],
            'inflation': [0.02, 0.01, 0.03],
            'expected_inflation': [0.02, 0.02, 0.02]
        })
        
        # Compute optimal policy rate
        optimal_rate = calculator.compute_optimal_rate(regional_conditions)
        
        # Should be a reasonable policy rate
        assert -0.1 < optimal_rate < 0.2  # Reasonable range
        
        # Test that optimal rate minimizes welfare loss
        # (This is a simplified test - full optimization would require more complex setup)
        assert isinstance(optimal_rate, (float, np.floating))


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_near_singular_matrices(self):
        """Test handling of near-singular matrices."""
        # Create near-singular matrix
        A = np.array([[1, 1], [1, 1 + 1e-15]])
        
        # Check condition number
        cond_num = np.linalg.cond(A)
        assert cond_num > 1e10  # Should be ill-conditioned
        
        # Test regularized inversion
        regularization = 1e-8
        A_reg = A + regularization * np.eye(2)
        
        # Should be invertible with regularization
        A_reg_inv = linalg.inv(A_reg)
        identity = A_reg @ A_reg_inv
        np.testing.assert_allclose(identity, np.eye(2), rtol=1e-6)
    
    def test_overflow_underflow_handling(self):
        """Test handling of numerical overflow and underflow."""
        # Test large numbers
        large_array = np.array([1e100, 1e100])
        
        # Should handle without overflow in sum
        result = np.sum(large_array)
        assert np.isfinite(result)
        
        # Test small numbers
        small_array = np.array([1e-100, 1e-100])
        result_small = np.sum(small_array)
        assert result_small > 0
        assert np.isfinite(result_small)
    
    def test_missing_data_handling(self):
        """Test handling of missing data in computations."""
        # Data with NaN values
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Test nanmean
        mean_result = np.nanmean(data_with_nan)
        expected_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4
        np.testing.assert_allclose(mean_result, expected_mean)
        
        # Test nanstd
        std_result = np.nanstd(data_with_nan)
        assert np.isfinite(std_result)
        assert std_result > 0


if __name__ == '__main__':
    pytest.main([__file__])