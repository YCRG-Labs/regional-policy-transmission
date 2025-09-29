"""
Tests for policy analysis and mistake decomposition system.

This module tests the policy mistake decomposition, optimal policy calculation,
Fed reaction function estimation, and information set reconstruction components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from regional_monetary_policy.econometric.models import RegionalParameters
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.policy.models import PolicyMistakeComponents, PolicyScenario
from regional_monetary_policy.policy.mistake_decomposer import PolicyMistakeDecomposer
from regional_monetary_policy.policy.optimal_policy import OptimalPolicyCalculator, WelfareFunction
from regional_monetary_policy.policy.fed_reaction import FedReactionEstimator, FedReactionResults
from regional_monetary_policy.policy.information_reconstruction import (
    InformationSetReconstructor, InformationSet
)


class TestPolicyMistakeDecomposer:
    """Test policy mistake decomposition functionality."""
    
    @pytest.fixture
    def sample_regional_params(self):
        """Create sample regional parameters for testing."""
        return RegionalParameters(
            sigma=np.array([0.5, 0.7, 0.6]),
            kappa=np.array([0.3, 0.4, 0.35]),
            psi=np.array([0.1, 0.15, 0.12]),
            phi=np.array([0.05, 0.08, 0.06]),
            beta=np.array([0.99, 0.99, 0.99]),
            standard_errors={
                'sigma': np.array([0.05, 0.07, 0.06]),
                'kappa': np.array([0.03, 0.04, 0.035])
            },
            confidence_intervals={}
        )
    
    @pytest.fixture
    def sample_welfare_weights(self):
        """Create sample welfare weights."""
        return np.array([0.4, 0.35, 0.25])
    
    @pytest.fixture
    def decomposer(self, sample_regional_params, sample_welfare_weights):
        """Create PolicyMistakeDecomposer instance."""
        return PolicyMistakeDecomposer(
            regional_params=sample_regional_params,
            social_welfare_weights=sample_welfare_weights
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regional data for testing."""
        dates = pd.date_range('2020-01-01', periods=12, freq='ME')
        
        real_time_data = pd.DataFrame({
            'output_gap_region_1': np.random.normal(0, 1, 12),
            'output_gap_region_2': np.random.normal(0, 1, 12),
            'output_gap_region_3': np.random.normal(0, 1, 12),
            'inflation_region_1': np.random.normal(2, 0.5, 12),
            'inflation_region_2': np.random.normal(2, 0.5, 12),
            'inflation_region_3': np.random.normal(2, 0.5, 12)
        }, index=dates)
        
        # True data with small differences from real-time
        true_data = real_time_data + np.random.normal(0, 0.1, real_time_data.shape)
        
        return real_time_data, true_data
    
    def test_initialization(self, sample_regional_params, sample_welfare_weights):
        """Test decomposer initialization."""
        decomposer = PolicyMistakeDecomposer(
            regional_params=sample_regional_params,
            social_welfare_weights=sample_welfare_weights
        )
        
        assert decomposer.regional_params == sample_regional_params
        assert np.array_equal(decomposer.social_welfare_weights, sample_welfare_weights)
        assert hasattr(decomposer, 'optimal_weights')
        assert hasattr(decomposer, 'optimal_output_coeff')
        assert hasattr(decomposer, 'optimal_inflation_coeff')
    
    def test_invalid_welfare_weights(self, sample_regional_params):
        """Test error handling for invalid welfare weights."""
        # Wrong dimension
        with pytest.raises(ValueError, match="must match number of regions"):
            PolicyMistakeDecomposer(
                regional_params=sample_regional_params,
                social_welfare_weights=np.array([0.5, 0.5])  # Only 2 regions
            )
        
        # Doesn't sum to 1
        with pytest.raises(ValueError, match="must sum to 1"):
            PolicyMistakeDecomposer(
                regional_params=sample_regional_params,
                social_welfare_weights=np.array([0.5, 0.5, 0.5])  # Sums to 1.5
            )
    
    def test_optimal_weights_computation(self, decomposer):
        """Test optimal regional weights computation."""
        weights = decomposer.optimal_weights
        
        # Should sum to 1
        assert np.isclose(np.sum(weights), 1.0)
        
        # Should be positive
        assert np.all(weights >= 0)
        
        # Should have correct dimension
        assert len(weights) == 3
    
    def test_policy_mistake_decomposition(self, decomposer, sample_data):
        """Test complete policy mistake decomposition."""
        real_time_data, true_data = sample_data
        
        # Test decomposition
        decomposition = decomposer.decompose_policy_mistake(
            actual_rate=2.5,
            optimal_rate=2.0,
            real_time_data=real_time_data,
            true_data=true_data
        )
        
        # Check result type
        assert isinstance(decomposition, PolicyMistakeComponents)
        
        # Check total mistake
        assert decomposition.total_mistake == 0.5
        
        # Check components sum to total (within numerical precision)
        component_sum = (
            decomposition.information_effect +
            decomposition.weight_misallocation_effect +
            decomposition.parameter_misspecification_effect +
            decomposition.inflation_response_effect
        )
        assert np.isclose(component_sum, decomposition.total_mistake, atol=1e-6)
        
        # Check additional details are present
        assert decomposition.measurement_errors is not None
        assert decomposition.weight_differences is not None
        assert decomposition.parameter_differences is not None
    
    def test_information_effect_computation(self, decomposer, sample_data):
        """Test information effect computation."""
        real_time_data, true_data = sample_data
        
        fed_coefficients = {'output': 0.5, 'inflation': 1.5}
        
        info_effect = decomposer._compute_information_effect(
            real_time_data, true_data, fed_coefficients
        )
        
        # Should be a scalar
        assert isinstance(info_effect, (int, float, np.number))
        
        # Should be finite
        assert np.isfinite(info_effect)
    
    def test_weight_misallocation_effect(self, decomposer, sample_data):
        """Test weight misallocation effect computation."""
        real_time_data, true_data = sample_data
        fed_weights = np.array([0.5, 0.3, 0.2])
        
        weight_effect = decomposer._compute_weight_misallocation_effect(
            fed_weights, true_data
        )
        
        # Should be a scalar
        assert isinstance(weight_effect, (int, float, np.number))
        
        # Should be finite
        assert np.isfinite(weight_effect)
    
    def test_counterfactual_mistake(self, decomposer, sample_data):
        """Test counterfactual policy mistake computation."""
        real_time_data, true_data = sample_data
        
        counterfactual_weights = np.array([0.4, 0.4, 0.2])
        counterfactual_coefficients = {'output': 0.3, 'inflation': 1.2}
        
        mistake = decomposer.compute_counterfactual_mistake(
            counterfactual_weights, counterfactual_coefficients, true_data
        )
        
        # Should be a scalar
        assert isinstance(mistake, (int, float, np.number))
        
        # Should be finite
        assert np.isfinite(mistake)


class TestOptimalPolicyCalculator:
    """Test optimal policy calculation functionality."""
    
    @pytest.fixture
    def sample_regional_params(self):
        """Create sample regional parameters."""
        return RegionalParameters(
            sigma=np.array([0.5, 0.7, 0.6]),
            kappa=np.array([0.3, 0.4, 0.35]),
            psi=np.array([0.1, 0.15, 0.12]),
            phi=np.array([0.05, 0.08, 0.06]),
            beta=np.array([0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
    
    @pytest.fixture
    def sample_welfare_function(self):
        """Create sample welfare function."""
        return WelfareFunction(
            output_gap_weight=1.0,
            inflation_weight=1.0,
            regional_weights=np.array([0.4, 0.35, 0.25]),
            loss_function='quadratic'
        )
    
    @pytest.fixture
    def calculator(self, sample_regional_params, sample_welfare_function):
        """Create OptimalPolicyCalculator instance."""
        return OptimalPolicyCalculator(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
    
    @pytest.fixture
    def sample_regional_conditions(self):
        """Create sample regional conditions data."""
        return pd.DataFrame({
            'output_gap_region_1': [0.5],
            'output_gap_region_2': [-0.3],
            'output_gap_region_3': [0.1],
            'inflation_region_1': [2.1],
            'inflation_region_2': [1.8],
            'inflation_region_3': [2.0]
        }, index=[pd.Timestamp('2020-01-01')])
    
    def test_initialization(self, sample_regional_params, sample_welfare_function):
        """Test calculator initialization."""
        calculator = OptimalPolicyCalculator(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        assert calculator.regional_params == sample_regional_params
        assert calculator.welfare_function == sample_welfare_function
        assert hasattr(calculator, 'optimal_regional_weights')
        assert hasattr(calculator, 'optimal_coefficients')
    
    def test_welfare_function_validation(self, sample_regional_params):
        """Test welfare function validation."""
        # Invalid regional weights (don't sum to 1)
        with pytest.raises(ValueError, match="must sum to 1"):
            WelfareFunction(regional_weights=np.array([0.5, 0.5, 0.5]))
        
        # Invalid loss function
        with pytest.raises(ValueError, match="must be 'quadratic' or 'asymmetric'"):
            WelfareFunction(loss_function='invalid')
    
    def test_optimal_rate_computation(self, calculator, sample_regional_conditions):
        """Test optimal policy rate computation."""
        optimal_rate = calculator.compute_optimal_rate(sample_regional_conditions)
        
        # Should be a scalar
        assert isinstance(optimal_rate, (int, float, np.number))
        
        # Should be finite
        assert np.isfinite(optimal_rate)
        
        # Should be reasonable (between -10% and 10%)
        assert -10 <= optimal_rate <= 10
    
    def test_optimal_rate_path(self, calculator):
        """Test optimal policy rate path computation."""
        # Create time series data
        dates = pd.date_range('2020-01-01', periods=6, freq='ME')
        regional_data = pd.DataFrame({
            'output_gap_region_1': np.random.normal(0, 0.5, 6),
            'output_gap_region_2': np.random.normal(0, 0.5, 6),
            'output_gap_region_3': np.random.normal(0, 0.5, 6),
            'inflation_region_1': np.random.normal(2, 0.3, 6),
            'inflation_region_2': np.random.normal(2, 0.3, 6),
            'inflation_region_3': np.random.normal(2, 0.3, 6)
        }, index=dates)
        
        optimal_path = calculator.compute_optimal_rate_path(regional_data)
        
        # Should be a Series
        assert isinstance(optimal_path, pd.Series)
        
        # Should have correct length
        assert len(optimal_path) == 6
        
        # Should have correct index
        assert optimal_path.index.equals(dates)
        
        # All values should be finite
        assert np.all(np.isfinite(optimal_path))
    
    def test_welfare_loss_evaluation(self, calculator):
        """Test welfare loss evaluation."""
        # Create sample policy path and outcomes
        dates = pd.date_range('2020-01-01', periods=3, freq='ME')
        policy_path = pd.Series([1.0, 1.5, 2.0], index=dates)
        
        # Regional outcomes (3 regions, 2 variables each, 3 time periods)
        regional_outcomes = pd.DataFrame(
            np.random.normal(0, 0.5, (6, 3)),
            index=[f'output_gap_region_{i+1}' for i in range(3)] + 
                  [f'inflation_region_{i+1}' for i in range(3)],
            columns=dates
        )
        
        welfare_loss = calculator.evaluate_welfare_loss(policy_path, regional_outcomes)
        
        # Should be a positive scalar
        assert isinstance(welfare_loss, (int, float, np.number))
        assert welfare_loss >= 0
        assert np.isfinite(welfare_loss)
    
    def test_policy_tradeoffs_analysis(self, calculator, sample_regional_conditions):
        """Test policy tradeoffs analysis."""
        tradeoffs = calculator.analyze_policy_tradeoffs(sample_regional_conditions)
        
        # Should be a dictionary with expected keys
        expected_keys = [
            'optimal_rate', 'regional_losses', 'regional_impacts',
            'total_welfare_loss', 'cross_regional_variance',
            'most_affected_region', 'least_affected_region'
        ]
        
        for key in expected_keys:
            assert key in tradeoffs
        
        # Check types
        assert isinstance(tradeoffs['optimal_rate'], (int, float, np.number))
        assert isinstance(tradeoffs['regional_losses'], list)
        assert isinstance(tradeoffs['regional_impacts'], dict)
        assert isinstance(tradeoffs['total_welfare_loss'], (int, float, np.number))


class TestFedReactionEstimator:
    """Test Fed reaction function estimation."""
    
    @pytest.fixture
    def sample_policy_data(self):
        """Create sample policy and regional data."""
        dates = pd.date_range('2010-01-01', periods=50, freq='ME')
        
        # Policy rates
        policy_rates = pd.Series(
            2.0 + np.random.normal(0, 0.5, 50),
            index=dates,
            name='fed_funds_rate'
        )
        
        # Regional data
        regional_data = pd.DataFrame({
            'output_gap_region_1': np.random.normal(0, 1, 50),
            'output_gap_region_2': np.random.normal(0, 1, 50),
            'inflation_region_1': np.random.normal(2, 0.5, 50),
            'inflation_region_2': np.random.normal(2, 0.5, 50)
        }, index=dates)
        
        return policy_rates, regional_data
    
    @pytest.fixture
    def estimator(self):
        """Create FedReactionEstimator instance."""
        return FedReactionEstimator(estimation_method='ols')
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = FedReactionEstimator(estimation_method='ols')
        
        assert estimator.estimation_method == 'ols'
        assert estimator.include_smoothing == True
        
        # Test invalid method
        with pytest.raises(ValueError, match="must be one of"):
            FedReactionEstimator(estimation_method='invalid')
    
    def test_reaction_function_estimation(self, estimator, sample_policy_data):
        """Test Fed reaction function estimation."""
        policy_rates, regional_data = sample_policy_data
        
        results = estimator.estimate_reaction_function(policy_rates, regional_data)
        
        # Check result type
        assert isinstance(results, FedReactionResults)
        
        # Check required attributes
        assert hasattr(results, 'estimated_coefficients')
        assert hasattr(results, 'implicit_regional_weights')
        assert hasattr(results, 'model_fit')
        assert hasattr(results, 'standard_errors')
        
        # Check model fit
        assert 'r_squared' in results.model_fit
        assert 0 <= results.model_fit['r_squared'] <= 1
    
    def test_data_alignment(self, estimator, sample_policy_data):
        """Test data alignment functionality."""
        policy_rates, regional_data = sample_policy_data
        
        aligned_data = estimator._align_data(policy_rates, regional_data)
        
        # Should be a DataFrame
        assert isinstance(aligned_data, pd.DataFrame)
        
        # Should have policy_rate column
        assert 'policy_rate' in aligned_data.columns
        
        # Should have regional data columns
        for col in regional_data.columns:
            assert col in aligned_data.columns
        
        # Should have lagged policy rate if smoothing enabled
        if estimator.include_smoothing:
            assert 'lagged_policy_rate' in aligned_data.columns
    
    def test_time_varying_weights(self, estimator, sample_policy_data):
        """Test time-varying regional weights estimation."""
        policy_rates, regional_data = sample_policy_data
        
        weights_df = estimator.estimate_time_varying_weights(
            policy_rates, regional_data, window_size=20
        )
        
        # Should be a DataFrame
        assert isinstance(weights_df, pd.DataFrame)
        
        # Should have regional weight columns
        assert len(weights_df.columns) >= 2  # At least 2 regions
        
        # Weights should sum to approximately 1 for each time period
        weight_sums = weights_df.sum(axis=1)
        assert np.all(np.abs(weight_sums - 1.0) < 0.1)  # Allow some numerical error
    
    def test_taylor_principle_test(self, estimator):
        """Test Taylor principle testing."""
        # Create mock results
        results = FedReactionResults(
            estimated_coefficients={'inflation': 1.5, 'output': 0.5},
            implicit_regional_weights=np.array([0.5, 0.5]),
            model_fit={'r_squared': 0.8},
            standard_errors={'inflation': 0.2, 'output': 0.1},
            confidence_intervals={}
        )
        
        taylor_test = estimator.test_taylor_principle(results)
        
        # Should be a dictionary
        assert isinstance(taylor_test, dict)
        
        # Should have required keys
        expected_keys = [
            'inflation_coefficient', 'satisfies_taylor_principle',
            't_statistic', 'p_value', 'interpretation'
        ]
        
        for key in expected_keys:
            assert key in taylor_test
        
        # Should satisfy Taylor principle (inflation coeff > 1)
        assert taylor_test['satisfies_taylor_principle'] == True


class TestInformationSetReconstructor:
    """Test information set reconstruction functionality."""
    
    @pytest.fixture
    def sample_real_time_data(self):
        """Create sample real-time regional dataset."""
        dates = pd.date_range('2020-01-01', periods=12, freq='ME')
        regions = ['region_1', 'region_2']
        
        # Regional data should have regions as index and time as columns
        output_gaps = pd.DataFrame(
            np.random.normal(0, 1, (len(regions), len(dates))),
            index=regions,
            columns=dates
        )
        
        inflation_rates = pd.DataFrame(
            np.random.normal(2, 0.5, (len(regions), len(dates))),
            index=regions,
            columns=dates
        )
        
        interest_rates = pd.Series(
            np.random.normal(1.5, 0.3, 12),
            index=dates
        )
        
        return RegionalDataset(
            output_gaps=output_gaps,
            inflation_rates=inflation_rates,
            interest_rates=interest_rates,
            real_time_estimates={},
            metadata={}
        )
    
    @pytest.fixture
    def reconstructor(self):
        """Create InformationSetReconstructor instance."""
        return InformationSetReconstructor()
    
    def test_initialization(self):
        """Test reconstructor initialization."""
        reconstructor = InformationSetReconstructor()
        
        assert hasattr(reconstructor, 'publication_lags')
        assert isinstance(reconstructor.publication_lags, dict)
        assert reconstructor.include_forecasts == False
    
    def test_information_set_reconstruction(self, reconstructor, sample_real_time_data):
        """Test information set reconstruction."""
        decision_date = '2020-06-01'
        
        info_set = reconstructor.reconstruct_information_set(
            decision_date, sample_real_time_data
        )
        
        # Check result type
        assert isinstance(info_set, InformationSet)
        
        # Check attributes
        assert info_set.decision_date == decision_date
        assert isinstance(info_set.available_data, pd.DataFrame)
        assert isinstance(info_set.data_vintages, dict)
        assert isinstance(info_set.information_lags, dict)
    
    def test_publication_lags(self, reconstructor, sample_real_time_data):
        """Test publication lag application."""
        decision_date = pd.Timestamp('2020-06-01')
        
        # Test with GDP data (1 month lag)
        gdp_data = pd.DataFrame({
            'gdp': np.random.normal(0, 1, 12)
        }, index=pd.date_range('2020-01-01', periods=12, freq='ME'))
        
        lagged_data = reconstructor._apply_publication_lags(
            gdp_data, decision_date, 'gdp'
        )
        
        # Should exclude data from May 2020 onwards (1 month lag)
        assert lagged_data.index.max() <= pd.Timestamp('2020-04-30')
    
    def test_historical_sequence_reconstruction(self, reconstructor, sample_real_time_data):
        """Test reconstruction of historical sequence."""
        decision_dates = ['2020-03-01', '2020-06-01', '2020-09-01']
        
        info_sets = reconstructor.reconstruct_historical_sequence(
            decision_dates, sample_real_time_data
        )
        
        # Should return list of InformationSet objects
        assert isinstance(info_sets, list)
        assert len(info_sets) <= len(decision_dates)  # Some might fail
        
        for info_set in info_sets:
            assert isinstance(info_set, InformationSet)
    
    def test_information_evolution_analysis(self, reconstructor):
        """Test information evolution analysis."""
        # Create mock information sets
        info_sets = []
        for i in range(3):
            date = f'2020-0{i+1}-01'
            data = pd.DataFrame({
                'series_1': [1.0, 2.0],
                'series_2': [3.0, 4.0]
            }, index=pd.date_range('2020-01-01', periods=2, freq='ME'))
            
            info_set = InformationSet(
                decision_date=date,
                available_data=data,
                data_vintages={'series_1': date, 'series_2': date},
                information_lags={'series_1': 0, 'series_2': 1}
            )
            info_sets.append(info_set)
        
        evolution_df = reconstructor.analyze_information_evolution(info_sets)
        
        # Should be a DataFrame
        assert isinstance(evolution_df, pd.DataFrame)
        
        # Should have expected columns
        expected_cols = ['decision_date', 'n_series', 'avg_data_lag_days', 'data_completeness']
        for col in expected_cols:
            assert col in evolution_df.columns
        
        # Should have correct number of rows
        assert len(evolution_df) == 3


class TestPolicyMistakeComponents:
    """Test PolicyMistakeComponents model."""
    
    def test_initialization_and_validation(self):
        """Test components initialization and validation."""
        # Valid components
        components = PolicyMistakeComponents(
            total_mistake=1.0,
            information_effect=0.3,
            weight_misallocation_effect=0.4,
            parameter_misspecification_effect=0.2,
            inflation_response_effect=0.1
        )
        
        assert components.total_mistake == 1.0
        
        # Invalid components (don't sum to total)
        with pytest.raises(ValueError, match="must sum to total mistake"):
            PolicyMistakeComponents(
                total_mistake=1.0,
                information_effect=0.3,
                weight_misallocation_effect=0.4,
                parameter_misspecification_effect=0.2,
                inflation_response_effect=0.2  # This makes sum = 1.1
            )
    
    def test_relative_contributions(self):
        """Test relative contributions calculation."""
        components = PolicyMistakeComponents(
            total_mistake=1.0,
            information_effect=0.5,
            weight_misallocation_effect=0.3,
            parameter_misspecification_effect=0.1,
            inflation_response_effect=0.1
        )
        
        relative = components.get_relative_contributions()
        
        # Should be percentages
        assert relative['information'] == 50.0
        assert relative['weight_misallocation'] == 30.0
        assert relative['parameter_misspecification'] == 10.0
        assert relative['inflation_response'] == 10.0
    
    def test_zero_mistake_handling(self):
        """Test handling of zero total mistake."""
        components = PolicyMistakeComponents(
            total_mistake=0.0,
            information_effect=0.0,
            weight_misallocation_effect=0.0,
            parameter_misspecification_effect=0.0,
            inflation_response_effect=0.0
        )
        
        relative = components.get_relative_contributions()
        
        # All should be zero
        for value in relative.values():
            assert value == 0.0


if __name__ == '__main__':
    pytest.main([__file__])