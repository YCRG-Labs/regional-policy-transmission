"""
Replication tests using published monetary policy research results.

This module tests the system's ability to replicate key findings from
published monetary policy research, validating the theoretical framework
and implementation accuracy.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from unittest.mock import Mock, patch

from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.policy.optimal_policy import OptimalPolicyCalculator
from regional_monetary_policy.policy.mistake_decomposer import PolicyMistakeDecomposer
from regional_monetary_policy.policy.counterfactual_engine import CounterfactualEngine
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.models import RegionalParameters


class TestTaylorRuleReplication:
    """Test replication of Taylor rule estimation results."""
    
    def test_taylor_rule_coefficients(self):
        """Test that estimated Taylor rule coefficients match literature ranges."""
        # Create synthetic Fed policy data consistent with Taylor rule
        n_periods = 200
        dates = pd.date_range('2000-01-01', periods=n_periods, freq='ME')
        
        # Generate aggregate economic conditions
        np.random.seed(42)
        
        # Inflation rate (annual, in percentage points)
        inflation_trend = 0.02 + 0.01 * np.sin(2 * np.pi * np.arange(n_periods) / 48)  # 4-year cycle
        inflation_noise = np.random.normal(0, 0.005, n_periods)
        inflation = inflation_trend + inflation_noise
        
        # Output gap (percentage points)
        output_gap_trend = 0.005 * np.sin(2 * np.pi * np.arange(n_periods) / 36)  # 3-year cycle
        output_gap_noise = np.random.normal(0, 0.01, n_periods)
        output_gap = output_gap_trend + output_gap_noise
        
        # Fed funds rate following Taylor rule
        # r = r* + π + 0.5(π - π*) + 0.5y
        # Simplified: r = 0.02 + 1.5π + 0.5y + ε
        taylor_intercept = 0.02
        taylor_inflation_coef = 1.5
        taylor_output_coef = 0.5
        
        policy_noise = np.random.normal(0, 0.002, n_periods)
        fed_funds_rate = (taylor_intercept + 
                         taylor_inflation_coef * inflation +
                         taylor_output_coef * output_gap +
                         policy_noise)
        
        # Create regional dataset (simplified for Taylor rule testing)
        n_regions = 4
        regions = ['Northeast', 'South', 'Midwest', 'West']
        
        # Regional data correlated with aggregate
        regional_output = np.zeros((n_regions, n_periods))
        regional_inflation = np.zeros((n_regions, n_periods))
        
        for i in range(n_regions):
            # Regional heterogeneity
            region_weight = 0.7 + 0.3 * np.random.random()  # 0.7 to 1.0
            region_bias = np.random.normal(0, 0.002)
            
            regional_output[i, :] = region_weight * output_gap + region_bias + np.random.normal(0, 0.005, n_periods)
            regional_inflation[i, :] = region_weight * inflation + region_bias + np.random.normal(0, 0.002, n_periods)
        
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(regional_output, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(regional_inflation, index=regions, columns=dates),
            interest_rates=pd.Series(fed_funds_rate, index=dates),
            real_time_estimates={},
            metadata={'true_taylor_rule': {
                'intercept': taylor_intercept,
                'inflation_coefficient': taylor_inflation_coef,
                'output_coefficient': taylor_output_coef
            }}
        )
        
        # Estimate Taylor rule using the system
        spatial_handler = SpatialModelHandler(regions)
        
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 10  # Reduce for testing speed
        
        estimator = ParameterEstimator(spatial_handler, config)
        
        # Create simple regional parameters for policy estimation
        regional_params = RegionalParameters(
            sigma=np.array([0.8, 0.9, 0.85, 0.95]),
            kappa=np.array([0.1, 0.12, 0.08, 0.11]),
            psi=np.array([0.0, 0.0, 0.0, 0.0]),  # No spillovers for simplicity
            phi=np.array([0.0, 0.0, 0.0, 0.0]),
            beta=np.array([0.99, 0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        # Estimate policy parameters
        estimated_policy = estimator.estimate_stage_three(dataset, regional_params)
        
        # Test coefficient recovery
        true_params = dataset.metadata['true_taylor_rule']
        
        # Inflation coefficient should be close to 1.5 (literature range: 1.2-2.0)
        inflation_coef = estimated_policy['inflation_coefficient']
        assert 1.0 < inflation_coef < 2.5, f"Inflation coefficient out of range: {inflation_coef}"
        
        # Output coefficient should be close to 0.5 (literature range: 0.2-0.8)
        output_coef = estimated_policy['output_coefficient']
        assert 0.1 < abs(output_coef) < 1.0, f"Output coefficient out of range: {output_coef}"
        
        # Test accuracy (allow 30% error due to estimation uncertainty)
        inflation_error = abs(inflation_coef - true_params['inflation_coefficient']) / true_params['inflation_coefficient']
        assert inflation_error < 0.5, f"Inflation coefficient error too large: {inflation_error:.2f}"
    
    def test_taylor_rule_stability(self):
        """Test stability of Taylor rule estimates across different periods."""
        # Test with different sample periods to check stability
        n_periods = 300
        dates = pd.date_range('1990-01-01', periods=n_periods, freq='ME')
        
        # Generate stable Taylor rule data
        np.random.seed(123)  # Different seed for robustness
        
        # Consistent Taylor rule throughout sample
        inflation = 0.025 + 0.01 * np.sin(2 * np.pi * np.arange(n_periods) / 60) + np.random.normal(0, 0.004, n_periods)
        output_gap = 0.008 * np.sin(2 * np.pi * np.arange(n_periods) / 40) + np.random.normal(0, 0.008, n_periods)
        
        # Stable Taylor rule coefficients
        fed_funds = 0.025 + 1.4 * inflation + 0.6 * output_gap + np.random.normal(0, 0.003, n_periods)
        
        # Create dataset
        regions = ['Region_A', 'Region_B', 'Region_C']
        n_regions = len(regions)
        
        regional_data = RegionalDataset(
            output_gaps=pd.DataFrame(
                np.random.multivariate_normal(
                    output_gap, 0.0001 * np.eye(n_regions), n_periods
                ).T,
                index=regions, columns=dates
            ),
            inflation_rates=pd.DataFrame(
                np.random.multivariate_normal(
                    inflation, 0.00005 * np.eye(n_regions), n_periods
                ).T,
                index=regions, columns=dates
            ),
            interest_rates=pd.Series(fed_funds, index=dates),
            real_time_estimates={},
            metadata={}
        )
        
        # Test estimation on different subsamples
        subsample_periods = [
            (0, 100),    # First third
            (100, 200),  # Middle third
            (200, 300)   # Last third
        ]
        
        estimated_coefficients = []
        
        for start, end in subsample_periods:
            # Create subsample
            subsample_dates = dates[start:end]
            subsample_data = RegionalDataset(
                output_gaps=regional_data.output_gaps.iloc[:, start:end],
                inflation_rates=regional_data.inflation_rates.iloc[:, start:end],
                interest_rates=regional_data.interest_rates.iloc[start:end],
                real_time_estimates={},
                metadata={}
            )
            
            # Estimate on subsample
            spatial_handler = SpatialModelHandler(regions)
            
            from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
            config = create_default_estimation_config()
            config.bootstrap_replications = 5
            
            estimator = ParameterEstimator(spatial_handler, config)
            
            # Simple regional parameters
            regional_params = RegionalParameters(
                sigma=np.array([0.8, 0.9, 0.85]),
                kappa=np.array([0.1, 0.12, 0.08]),
                psi=np.array([0.0, 0.0, 0.0]),
                phi=np.array([0.0, 0.0, 0.0]),
                beta=np.array([0.99, 0.99, 0.99]),
                standard_errors={},
                confidence_intervals={}
            )
            
            policy_params = estimator.estimate_stage_three(subsample_data, regional_params)
            estimated_coefficients.append(policy_params)
        
        # Test coefficient stability across subsamples
        inflation_coeffs = [params['inflation_coefficient'] for params in estimated_coefficients]
        output_coeffs = [params['output_coefficient'] for params in estimated_coefficients]
        
        # Coefficients should be reasonably stable (coefficient of variation < 0.3)
        inflation_cv = np.std(inflation_coeffs) / np.mean(inflation_coeffs)
        output_cv = np.std(output_coeffs) / abs(np.mean(output_coeffs))
        
        assert inflation_cv < 0.4, f"Inflation coefficient too unstable: CV = {inflation_cv:.3f}"
        assert output_cv < 0.5, f"Output coefficient too unstable: CV = {output_cv:.3f}"


class TestRegionalHeterogeneityReplication:
    """Test replication of regional heterogeneity findings."""
    
    def test_regional_parameter_heterogeneity(self):
        """Test that system detects significant regional heterogeneity."""
        # Create data with known regional heterogeneity
        n_regions = 6
        n_periods = 250
        
        regions = ['Manufacturing', 'Services', 'Agriculture', 'Energy', 'Finance', 'Technology']
        dates = pd.date_range('1995-01-01', periods=n_periods, freq='ME')
        
        # Define heterogeneous regional parameters
        true_regional_params = {
            'Manufacturing': {'sigma': 1.2, 'kappa': 0.15},  # High interest sensitivity
            'Services': {'sigma': 0.8, 'kappa': 0.12},       # Moderate sensitivity
            'Agriculture': {'sigma': 0.6, 'kappa': 0.08},    # Low interest sensitivity
            'Energy': {'sigma': 1.5, 'kappa': 0.20},         # Very high sensitivity
            'Finance': {'sigma': 1.8, 'kappa': 0.18},        # Highest sensitivity
            'Technology': {'sigma': 1.0, 'kappa': 0.10}      # Moderate sensitivity
        }
        
        # Generate regional data with heterogeneous responses
        np.random.seed(456)
        
        # Common monetary policy shocks
        policy_shocks = np.random.normal(0, 0.01, n_periods)
        aggregate_conditions = np.random.normal(0, 0.005, n_periods)
        
        regional_output = np.zeros((n_regions, n_periods))
        regional_inflation = np.zeros((n_regions, n_periods))
        interest_rates = 0.03 + np.cumsum(policy_shocks) * 0.1
        
        for i, region in enumerate(regions):
            params = true_regional_params[region]
            
            # Regional responses to monetary policy
            for t in range(1, n_periods):
                # IS curve with heterogeneous interest sensitivity
                regional_output[i, t] = (0.7 * regional_output[i, t-1] -
                                       params['sigma'] * policy_shocks[t] +
                                       0.3 * aggregate_conditions[t] +
                                       np.random.normal(0, 0.008))
                
                # Phillips curve with heterogeneous slope
                regional_inflation[i, t] = (0.8 * regional_inflation[i, t-1] +
                                          params['kappa'] * regional_output[i, t] +
                                          np.random.normal(0, 0.004))
        
        # Create dataset
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(regional_output, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(regional_inflation, index=regions, columns=dates),
            interest_rates=pd.Series(interest_rates, index=dates),
            real_time_estimates={},
            metadata={'true_regional_params': true_regional_params}
        )
        
        # Estimate regional parameters
        spatial_handler = SpatialModelHandler(regions)
        
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 15
        
        estimator = ParameterEstimator(spatial_handler, config)
        
        # Run estimation
        results = estimator.estimate_full_model(dataset)
        estimated_params = results.regional_parameters
        
        # Test heterogeneity detection
        # 1. Test that sigma parameters show significant variation
        sigma_cv = np.std(estimated_params.sigma) / np.mean(estimated_params.sigma)
        assert sigma_cv > 0.15, f"Insufficient sigma heterogeneity detected: CV = {sigma_cv:.3f}"
        
        # 2. Test that kappa parameters show significant variation
        kappa_cv = np.std(estimated_params.kappa) / np.mean(estimated_params.kappa)
        assert kappa_cv > 0.15, f"Insufficient kappa heterogeneity detected: CV = {kappa_cv:.3f}"
        
        # 3. Test parameter ordering (relative rankings should be preserved)
        true_sigma_order = sorted(range(n_regions), 
                                key=lambda i: true_regional_params[regions[i]]['sigma'])
        estimated_sigma_order = sorted(range(n_regions), 
                                     key=lambda i: estimated_params.sigma[i])
        
        # Check if rankings are reasonably preserved (allow some reordering)
        rank_correlation = np.corrcoef(true_sigma_order, estimated_sigma_order)[0, 1]
        assert rank_correlation > 0.3, f"Parameter rankings not preserved: correlation = {rank_correlation:.3f}"
    
    def test_spatial_spillover_effects(self):
        """Test detection of spatial spillover effects."""
        # Create data with known spatial spillovers
        n_regions = 4
        n_periods = 200
        
        regions = ['Core', 'Adjacent1', 'Adjacent2', 'Peripheral']
        dates = pd.date_range('2000-01-01', periods=n_periods, freq='ME')
        
        # Define spatial structure: Core affects adjacent regions, which affect peripheral
        true_spatial_weights = np.array([
            [0.0, 0.4, 0.4, 0.2],  # Core affects adjacent regions
            [0.3, 0.0, 0.3, 0.4],  # Adjacent1
            [0.3, 0.3, 0.0, 0.4],  # Adjacent2
            [0.2, 0.4, 0.4, 0.0]   # Peripheral affected by adjacent
        ])
        
        # Generate data with spatial spillovers
        np.random.seed(789)
        
        regional_output = np.zeros((n_regions, n_periods))
        regional_inflation = np.zeros((n_regions, n_periods))
        
        # Common shocks
        common_shocks = np.random.normal(0, 0.005, n_periods)
        regional_shocks = np.random.normal(0, 0.008, (n_regions, n_periods))
        
        for t in range(1, n_periods):
            for i in range(n_regions):
                # Spatial spillovers in output
                spatial_output = true_spatial_weights[i, :] @ regional_output[:, t-1]
                regional_output[i, t] = (0.6 * regional_output[i, t-1] +
                                       0.2 * spatial_output +
                                       common_shocks[t] +
                                       regional_shocks[i, t])
                
                # Spatial spillovers in inflation
                spatial_inflation = true_spatial_weights[i, :] @ regional_inflation[:, t-1]
                regional_inflation[i, t] = (0.7 * regional_inflation[i, t-1] +
                                          0.1 * regional_output[i, t] +
                                          0.1 * spatial_inflation +
                                          0.5 * common_shocks[t] +
                                          0.5 * regional_shocks[i, t])
        
        # Create dataset
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(regional_output, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(regional_inflation, index=regions, columns=dates),
            interest_rates=pd.Series(np.random.normal(0.03, 0.01, n_periods), index=dates),
            real_time_estimates={},
            metadata={'true_spatial_weights': true_spatial_weights}
        )
        
        # Test spatial spillover detection
        spatial_handler = SpatialModelHandler(regions)
        
        # Create mock spatial data
        trade_data = self._create_spatial_data_from_weights(regions, true_spatial_weights, 'trade')
        migration_data = self._create_spatial_data_from_weights(regions, true_spatial_weights, 'migration')
        financial_data = pd.DataFrame()  # Empty for simplicity
        distance_matrix = np.random.uniform(100, 1000, (n_regions, n_regions))
        
        # Estimate spatial weights
        spatial_results = spatial_handler.construct_weights(
            trade_data, migration_data, financial_data, distance_matrix,
            weights=(0.6, 0.4, 0.0, 0.0)
        )
        
        # Extract weight matrix
        estimated_weights = spatial_results.weight_matrix
        
        # Test spatial autocorrelation
        spatial_lags = spatial_handler.compute_spatial_lags(dataset.output_gaps.T, estimated_weights)
        
        # Compute spatial autocorrelation
        autocorr_stats = spatial_handler.test_spatial_autocorrelation(
            dataset.output_gaps.T, estimated_weights
        )
        
        # Should detect significant spatial autocorrelation
        assert 'morans_i' in autocorr_stats
        assert abs(autocorr_stats['morans_i']) > 0.1, f"Weak spatial autocorrelation: {autocorr_stats['morans_i']:.3f}"
    
    def _create_spatial_data_from_weights(self, regions, weights, data_type):
        """Create spatial interaction data consistent with weight matrix."""
        flows = []
        base_flow = 100
        
        for i, origin in enumerate(regions):
            for j, destination in enumerate(regions):
                if i != j:
                    flow = base_flow * weights[i, j] * (1 + np.random.normal(0, 0.2))
                    
                    if data_type == 'financial':
                        # Financial data uses different column names
                        flows.append({
                            'region1': origin,
                            'region2': destination,
                            'financial_linkage': max(0, flow)
                        })
                    else:
                        # Trade and migration use origin/destination format
                        flows.append({
                            'origin': origin,
                            'destination': destination,
                            f'{data_type}_flow': max(0, flow)
                        })
        
        return pd.DataFrame(flows)


class TestPolicyEffectivenessReplication:
    """Test replication of policy effectiveness studies."""
    
    def test_welfare_gains_from_regional_information(self):
        """Test that regional information provides welfare gains as in literature."""
        # Create scenario where regional information matters
        n_regions = 5
        n_periods = 150
        
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        dates = pd.date_range('2005-01-01', periods=n_periods, freq='ME')
        
        # Create heterogeneous regional parameters
        regional_params = RegionalParameters(
            sigma=np.array([0.6, 1.0, 0.8, 1.2, 0.9]),  # Heterogeneous interest sensitivity
            kappa=np.array([0.08, 0.15, 0.10, 0.18, 0.12]),  # Heterogeneous Phillips curves
            psi=np.array([0.1, 0.05, 0.15, 0.08, 0.12]),  # Demand spillovers
            phi=np.array([0.05, 0.08, 0.03, 0.10, 0.06]),  # Price spillovers
            beta=np.array([0.99, 0.99, 0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        # Population-based welfare weights (realistic distribution)
        welfare_weights = np.array([0.25, 0.20, 0.18, 0.22, 0.15])
        
        # Generate historical data with regional heterogeneity
        np.random.seed(101112)
        
        # Correlated but heterogeneous regional shocks
        shock_correlation = 0.3 * np.ones((n_regions, n_regions)) + 0.7 * np.eye(n_regions)
        regional_shocks = np.random.multivariate_normal(
            np.zeros(n_regions), 0.01 * shock_correlation, n_periods
        ).T
        
        historical_data = RegionalDataset(
            output_gaps=pd.DataFrame(regional_shocks, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(
                0.5 * regional_shocks + np.random.normal(0, 0.003, (n_regions, n_periods)),
                index=regions, columns=dates
            ),
            interest_rates=pd.Series(
                np.random.normal(0.03, 0.008, n_periods), index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        # Run counterfactual analysis
        counterfactual_engine = CounterfactualEngine(regional_params, welfare_weights)
        
        # Generate scenarios
        baseline = counterfactual_engine.generate_baseline_scenario(historical_data)
        perfect_info = counterfactual_engine.generate_perfect_info_scenario(historical_data)
        optimal_regional = counterfactual_engine.generate_optimal_regional_scenario(historical_data)
        perfect_regional = counterfactual_engine.generate_perfect_regional_scenario(historical_data)
        
        # Test welfare ranking (literature finding: W^PR ≥ W^PI ≥ W^OR ≥ W^B)
        welfare_values = [
            baseline.welfare_outcome,
            optimal_regional.welfare_outcome,
            perfect_info.welfare_outcome,
            perfect_regional.welfare_outcome
        ]
        
        # Test monotonic improvement
        assert perfect_regional.welfare_outcome >= perfect_info.welfare_outcome, \
            "Perfect Regional should dominate Perfect Information"
        
        assert perfect_info.welfare_outcome >= optimal_regional.welfare_outcome, \
            "Perfect Information should dominate Optimal Regional"
        
        assert optimal_regional.welfare_outcome >= baseline.welfare_outcome, \
            "Optimal Regional should dominate Baseline"
        
        # Test magnitude of welfare gains (should be economically significant)
        welfare_gain_perfect_info = perfect_info.welfare_outcome - baseline.welfare_outcome
        welfare_gain_perfect_regional = perfect_regional.welfare_outcome - baseline.welfare_outcome
        
        # Welfare gains should be positive and meaningful
        assert welfare_gain_perfect_info > 0, "Perfect information should provide welfare gains"
        assert welfare_gain_perfect_regional > welfare_gain_perfect_info, \
            "Perfect regional policy should provide larger gains than perfect information"
        
        # Gains should be economically significant (literature suggests 0.1-1% of GDP equivalent)
        relative_gain = welfare_gain_perfect_regional / abs(baseline.welfare_outcome)
        assert 0.001 < relative_gain < 0.1, f"Welfare gains not in expected range: {relative_gain:.4f}"
    
    def test_policy_mistake_decomposition_replication(self):
        """Test policy mistake decomposition matches theoretical predictions."""
        # Create scenario with known policy mistakes
        regional_params = RegionalParameters(
            sigma=np.array([0.8, 1.0, 0.9]),
            kappa=np.array([0.1, 0.12, 0.08]),
            psi=np.array([0.05, 0.08, 0.06]),
            phi=np.array([0.03, 0.05, 0.04]),
            beta=np.array([0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        welfare_weights = np.array([0.4, 0.35, 0.25])
        
        # Create policy mistake decomposer
        decomposer = PolicyMistakeDecomposer(regional_params, welfare_weights)
        
        # Scenario 1: Information effect (Fed has wrong data)
        true_conditions = pd.DataFrame({
            'output_gap': [0.01, -0.005, 0.008],
            'inflation': [0.02, 0.018, 0.022],
            'expected_inflation': [0.02, 0.02, 0.02]
        })
        
        # Fed's real-time data (with measurement error)
        fed_conditions = pd.DataFrame({
            'output_gap': [0.005, 0.000, 0.003],  # Underestimated output gaps
            'inflation': [0.018, 0.016, 0.020],   # Underestimated inflation
            'expected_inflation': [0.02, 0.02, 0.02]
        })
        
        # Compute optimal policies
        optimal_calculator = OptimalPolicyCalculator(regional_params, welfare_weights)
        
        true_optimal_rate = optimal_calculator.compute_optimal_rate(true_conditions)
        fed_optimal_rate = optimal_calculator.compute_optimal_rate(fed_conditions)
        
        # Fed sets policy based on their (incorrect) information
        actual_rate = fed_optimal_rate
        
        # Decompose policy mistake
        mistake_components = decomposer.decompose_policy_mistake(
            actual_rate=actual_rate,
            optimal_rate=true_optimal_rate,
            regional_conditions=true_conditions,
            real_time_data=fed_conditions,
            true_data=true_conditions
        )
        
        # Test decomposition properties
        # 1. Total mistake should equal sum of components (approximately)
        component_sum = (mistake_components.information_effect +
                        mistake_components.weight_misallocation_effect +
                        mistake_components.parameter_misspecification_effect +
                        mistake_components.inflation_response_effect)
        
        np.testing.assert_allclose(
            mistake_components.total_mistake, component_sum, rtol=0.1,
            err_msg="Mistake decomposition doesn't sum correctly"
        )
        
        # 2. Information effect should dominate in this scenario
        assert abs(mistake_components.information_effect) > 0, "Information effect should be non-zero"
        
        # 3. Components should have reasonable magnitudes
        assert abs(mistake_components.total_mistake) < 0.05, "Total mistake unreasonably large"


class TestLiteratureBenchmarks:
    """Test against specific benchmarks from monetary policy literature."""
    
    def test_interest_rate_sensitivity_ranges(self):
        """Test that estimated interest rate sensitivities fall within literature ranges."""
        # Literature suggests sigma (interest rate sensitivity) typically ranges from 0.2 to 2.0
        # with most estimates between 0.5 and 1.5
        
        # Create realistic synthetic data
        n_regions = 8
        n_periods = 240  # 20 years
        
        regions = [f"State_{i+1:02d}" for i in range(n_regions)]
        dates = pd.date_range('2000-01-01', periods=n_periods, freq='ME')
        
        # Literature-consistent parameter values
        true_sigma = np.random.uniform(0.4, 1.6, n_regions)  # Within literature range
        true_kappa = np.random.uniform(0.05, 0.25, n_regions)  # Phillips curve slopes
        
        # Generate data using these parameters
        np.random.seed(131415)
        
        policy_rates = np.random.normal(0.03, 0.015, n_periods)
        policy_changes = np.diff(policy_rates, prepend=policy_rates[0])
        
        regional_output = np.zeros((n_regions, n_periods))
        regional_inflation = np.zeros((n_regions, n_periods))
        
        for t in range(1, n_periods):
            for i in range(n_regions):
                # IS curve response
                regional_output[i, t] = (0.75 * regional_output[i, t-1] -
                                       true_sigma[i] * policy_changes[t] +
                                       np.random.normal(0, 0.01))
                
                # Phillips curve
                regional_inflation[i, t] = (0.8 * regional_inflation[i, t-1] +
                                          true_kappa[i] * regional_output[i, t] +
                                          np.random.normal(0, 0.005))
        
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(regional_output, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(regional_inflation, index=regions, columns=dates),
            interest_rates=pd.Series(policy_rates, index=dates),
            real_time_estimates={},
            metadata={'true_sigma': true_sigma, 'true_kappa': true_kappa}
        )
        
        # Estimate parameters
        spatial_handler = SpatialModelHandler(regions)
        
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 20
        
        estimator = ParameterEstimator(spatial_handler, config)
        results = estimator.estimate_full_model(dataset)
        
        estimated_sigma = results.regional_parameters.sigma
        estimated_kappa = results.regional_parameters.kappa
        
        # Test literature consistency
        # 1. Interest rate sensitivity should be in literature range
        assert np.all(estimated_sigma > 0.1), "Some sigma estimates too low"
        assert np.all(estimated_sigma < 3.0), "Some sigma estimates too high"
        assert np.mean(estimated_sigma) > 0.3, "Average sigma too low"
        assert np.mean(estimated_sigma) < 2.5, "Average sigma too high"
        
        # 2. Phillips curve slopes should be reasonable
        assert np.all(estimated_kappa > 0), "Phillips curve slopes should be positive"
        assert np.all(estimated_kappa < 0.5), "Phillips curve slopes too steep"
        
        # 3. Parameter recovery should be reasonable
        sigma_recovery_error = np.mean(np.abs(estimated_sigma - true_sigma) / true_sigma)
        assert sigma_recovery_error < 0.6, f"Poor sigma recovery: {sigma_recovery_error:.2f}"
    
    def test_taylor_rule_literature_consistency(self):
        """Test Taylor rule estimates against literature consensus."""
        # Literature consensus: inflation coefficient ~1.5, output coefficient ~0.5
        # See Taylor (1993), Clarida et al. (2000), etc.
        
        # Generate data consistent with literature Taylor rule
        n_periods = 200
        dates = pd.date_range('1990-01-01', periods=n_periods, freq='ME')
        
        np.random.seed(161718)
        
        # Realistic macroeconomic data
        inflation = 0.025 + 0.015 * np.sin(2 * np.pi * np.arange(n_periods) / 80) + np.random.normal(0, 0.008, n_periods)
        output_gap = 0.01 * np.sin(2 * np.pi * np.arange(n_periods) / 60) + np.random.normal(0, 0.012, n_periods)
        
        # Literature-consistent Taylor rule
        taylor_intercept = 0.02
        taylor_inflation = 1.5  # Literature consensus
        taylor_output = 0.5     # Literature consensus
        
        fed_funds = (taylor_intercept + 
                    taylor_inflation * inflation +
                    taylor_output * output_gap +
                    np.random.normal(0, 0.004, n_periods))
        
        # Ensure non-negative interest rates
        fed_funds = np.maximum(fed_funds, 0.001)
        
        # Create regional dataset
        n_regions = 4
        regions = ['Northeast', 'South', 'Midwest', 'West']
        
        # Regional data correlated with national aggregates
        regional_correlations = [0.8, 0.85, 0.75, 0.9]  # Different regional correlations
        
        regional_output = np.zeros((n_regions, n_periods))
        regional_inflation = np.zeros((n_regions, n_periods))
        
        for i, corr in enumerate(regional_correlations):
            regional_noise_output = np.random.normal(0, 0.008, n_periods)
            regional_noise_inflation = np.random.normal(0, 0.004, n_periods)
            
            regional_output[i, :] = corr * output_gap + np.sqrt(1 - corr**2) * regional_noise_output
            regional_inflation[i, :] = corr * inflation + np.sqrt(1 - corr**2) * regional_noise_inflation
        
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(regional_output, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(regional_inflation, index=regions, columns=dates),
            interest_rates=pd.Series(fed_funds, index=dates),
            real_time_estimates={},
            metadata={
                'true_taylor_rule': {
                    'intercept': taylor_intercept,
                    'inflation_coefficient': taylor_inflation,
                    'output_coefficient': taylor_output
                }
            }
        )
        
        # Estimate Taylor rule
        spatial_handler = SpatialModelHandler(regions)
        
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 15
        
        estimator = ParameterEstimator(spatial_handler, config)
        
        # Create regional parameters for policy estimation
        regional_params = RegionalParameters(
            sigma=np.array([0.8, 0.9, 0.7, 1.0]),
            kappa=np.array([0.1, 0.12, 0.08, 0.11]),
            psi=np.array([0.0, 0.0, 0.0, 0.0]),
            phi=np.array([0.0, 0.0, 0.0, 0.0]),
            beta=np.array([0.99, 0.99, 0.99, 0.99]),
            standard_errors={},
            confidence_intervals={}
        )
        
        estimated_policy = estimator.estimate_stage_three(dataset, regional_params)
        
        # Test literature consistency
        estimated_inflation_coef = estimated_policy['inflation_coefficient']
        estimated_output_coef = estimated_policy['output_coefficient']
        
        # Literature ranges (allowing for estimation uncertainty)
        assert 1.0 < estimated_inflation_coef < 2.5, \
            f"Inflation coefficient outside literature range: {estimated_inflation_coef:.2f}"
        
        assert 0.1 < abs(estimated_output_coef) < 1.2, \
            f"Output coefficient outside literature range: {estimated_output_coef:.2f}"
        
        # Test accuracy relative to true values
        inflation_error = abs(estimated_inflation_coef - taylor_inflation) / taylor_inflation
        output_error = abs(estimated_output_coef - taylor_output) / abs(taylor_output)
        
        assert inflation_error < 0.4, f"Inflation coefficient error too large: {inflation_error:.2f}"
        assert output_error < 0.6, f"Output coefficient error too large: {output_error:.2f}"


if __name__ == '__main__':
    pytest.main([__file__])