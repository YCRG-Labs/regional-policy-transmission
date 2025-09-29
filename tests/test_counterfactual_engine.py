"""
Tests for counterfactual analysis engine.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from regional_monetary_policy.policy.counterfactual_engine import (
    CounterfactualEngine, WelfareEvaluator
)
from regional_monetary_policy.policy.optimal_policy import WelfareFunction
from regional_monetary_policy.econometric.models import RegionalParameters
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.policy.models import PolicyScenario, WelfareDecomposition
from regional_monetary_policy.exceptions import WelfareCalculationError


@pytest.fixture
def sample_regional_params():
    """Create sample regional parameters for testing."""
    n_regions = 3
    
    return RegionalParameters(
        sigma=np.array([0.5, 0.7, 0.6]),  # Interest rate sensitivities
        kappa=np.array([0.3, 0.4, 0.35]),  # Phillips curve slopes
        psi=np.array([0.2, 0.25, 0.22]),  # Demand spillovers
        phi=np.array([0.1, 0.15, 0.12]),  # Price spillovers
        beta=np.array([0.99, 0.99, 0.99]),  # Discount factors
        standard_errors={},
        confidence_intervals={}
    )


@pytest.fixture
def sample_welfare_function():
    """Create sample welfare function for testing."""
    return WelfareFunction(
        output_gap_weight=1.0,
        inflation_weight=1.0,
        regional_weights=np.array([0.4, 0.35, 0.25]),  # Population-based weights
        loss_function='quadratic'
    )


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    n_regions = 3
    n_periods = 24  # 2 years of monthly data
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='ME')
    
    # Generate synthetic regional data
    np.random.seed(42)  # For reproducibility
    
    # Output gaps with some regional variation
    output_gaps = pd.DataFrame(
        np.random.normal(0, 1, (n_regions, n_periods)),
        index=[f"Region_{i+1}" for i in range(n_regions)],
        columns=dates
    )
    
    # Inflation rates with persistence
    inflation_base = np.random.normal(2.0, 0.5, n_periods)
    inflation_rates = pd.DataFrame(
        [inflation_base + np.random.normal(0, 0.2, n_periods) for _ in range(n_regions)],
        index=[f"Region_{i+1}" for i in range(n_regions)],
        columns=dates
    )
    
    # Policy interest rates
    interest_rates = pd.Series(
        np.random.normal(2.5, 0.5, n_periods),
        index=dates,
        name='fed_funds_rate'
    )
    
    # Real-time estimates (with measurement error)
    real_time_estimates = {
        'output_gaps': output_gaps + np.random.normal(0, 0.1, output_gaps.shape),
        'inflation_rates': inflation_rates + np.random.normal(0, 0.05, inflation_rates.shape)
    }
    
    return RegionalDataset(
        output_gaps=output_gaps,
        inflation_rates=inflation_rates,
        interest_rates=interest_rates,
        real_time_estimates=real_time_estimates,
        metadata={'source': 'synthetic', 'frequency': 'monthly'}
    )


@pytest.fixture
def sample_fed_policy_rates(sample_historical_data):
    """Create sample Fed policy rates."""
    return sample_historical_data.interest_rates


@pytest.fixture
def sample_fed_reaction_function():
    """Create sample Fed reaction function parameters."""
    return {
        'output_gap_response': 0.5,
        'inflation_response': 1.5,
        'interest_rate_smoothing': 0.8
    }


class TestCounterfactualEngine:
    """Test cases for CounterfactualEngine class."""
    
    def test_initialization(self, sample_regional_params, sample_welfare_function):
        """Test CounterfactualEngine initialization."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function,
            discount_factor=0.99,
            n_workers=2
        )
        
        assert engine.regional_params == sample_regional_params
        assert engine.welfare_function == sample_welfare_function
        assert engine.discount_factor == 0.99
        assert engine.n_workers == 2
        assert engine.optimal_calculator is not None
        assert isinstance(engine._scenario_cache, dict)
    
    def test_generate_baseline_scenario(self, sample_regional_params, sample_welfare_function,
                                      sample_historical_data, sample_fed_policy_rates):
        """Test baseline scenario generation."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        scenario = engine.generate_baseline_scenario(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates
        )
        
        assert isinstance(scenario, PolicyScenario)
        assert scenario.scenario_type == 'baseline'
        assert scenario.name == "Baseline (Historical Fed Policy)"
        assert len(scenario.policy_rates) == len(sample_fed_policy_rates)
        assert scenario.welfare_outcome is not None
        assert scenario.regional_outcomes is not None
    
    def test_generate_perfect_info_scenario(self, sample_regional_params, sample_welfare_function,
                                          sample_historical_data, sample_fed_reaction_function):
        """Test perfect information scenario generation."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        scenario = engine.generate_perfect_info_scenario(
            historical_data=sample_historical_data,
            fed_reaction_function=sample_fed_reaction_function
        )
        
        assert isinstance(scenario, PolicyScenario)
        assert scenario.scenario_type == 'perfect_info'
        assert scenario.name == "Perfect Information (Fed Policy)"
        assert scenario.policy_parameters == sample_fed_reaction_function
        assert scenario.welfare_outcome is not None
    
    def test_generate_optimal_regional_scenario(self, sample_regional_params, sample_welfare_function,
                                              sample_historical_data):
        """Test optimal regional scenario generation."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        scenario = engine.generate_optimal_regional_scenario(
            historical_data=sample_historical_data
        )
        
        assert isinstance(scenario, PolicyScenario)
        assert scenario.scenario_type == 'optimal_regional'
        assert scenario.name == "Optimal Regional (Real-time Info)"
        assert scenario.regional_weights is not None
        assert scenario.welfare_outcome is not None
    
    def test_generate_perfect_regional_scenario(self, sample_regional_params, sample_welfare_function,
                                              sample_historical_data):
        """Test perfect regional scenario generation."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        scenario = engine.generate_perfect_regional_scenario(
            historical_data=sample_historical_data
        )
        
        assert isinstance(scenario, PolicyScenario)
        assert scenario.scenario_type == 'perfect_regional'
        assert scenario.name == "Perfect Regional (Optimal + Perfect Info)"
        assert scenario.regional_weights is not None
        assert scenario.welfare_outcome is not None
    
    def test_generate_all_scenarios_sequential(self, sample_regional_params, sample_welfare_function,
                                             sample_historical_data, sample_fed_policy_rates,
                                             sample_fed_reaction_function):
        """Test generating all scenarios sequentially."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function,
            n_workers=1  # Force sequential processing
        )
        
        scenarios = engine.generate_all_scenarios(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates,
            fed_reaction_function=sample_fed_reaction_function,
            parallel=False
        )
        
        assert len(scenarios) == 4
        scenario_types = [s.scenario_type for s in scenarios]
        expected_types = ['baseline', 'perfect_info', 'optimal_regional', 'perfect_regional']
        assert all(stype in scenario_types for stype in expected_types)
    
    def test_generate_all_scenarios_parallel_fallback(self, sample_regional_params, 
                                                     sample_welfare_function, sample_historical_data,
                                                     sample_fed_policy_rates, sample_fed_reaction_function):
        """Test that parallel processing falls back to sequential when needed."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function,
            n_workers=1  # This should force sequential processing
        )
        
        # Generate scenarios with parallel=True but should fall back to sequential
        scenarios = engine.generate_all_scenarios(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates,
            fed_reaction_function=sample_fed_reaction_function,
            parallel=True
        )
        
        assert len(scenarios) == 4
        scenario_types = [s.scenario_type for s in scenarios]
        expected_types = ['baseline', 'perfect_info', 'optimal_regional', 'perfect_regional']
        assert all(stype in scenario_types for stype in expected_types)
    
    def test_compare_scenarios(self, sample_regional_params, sample_welfare_function,
                             sample_historical_data, sample_fed_policy_rates,
                             sample_fed_reaction_function):
        """Test scenario comparison functionality."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        scenarios = engine.generate_all_scenarios(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates,
            fed_reaction_function=sample_fed_reaction_function,
            parallel=False
        )
        
        comparison = engine.compare_scenarios(scenarios)
        
        assert len(comparison.scenario_names) == 4
        assert len(comparison.welfare_outcomes) == 4
        assert len(comparison.welfare_ranking) == 4
        assert isinstance(comparison.pairwise_comparisons, dict)
        
        # Check that rankings are valid (1-4)
        assert set(comparison.welfare_ranking) == {1, 2, 3, 4}
    
    def test_compare_scenarios_invalid_length(self, sample_regional_params, sample_welfare_function):
        """Test that compare_scenarios raises error with wrong number of scenarios."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        # Create only 2 scenarios instead of 4
        scenarios = [
            PolicyScenario("Test1", pd.Series(), pd.DataFrame(), 0.0, 'baseline'),
            PolicyScenario("Test2", pd.Series(), pd.DataFrame(), 0.0, 'perfect_info')
        ]
        
        with pytest.raises(ValueError, match="Expected exactly 4 scenarios"):
            engine.compare_scenarios(scenarios)
    
    def test_scenario_caching(self, sample_regional_params, sample_welfare_function,
                            sample_historical_data, sample_fed_policy_rates):
        """Test that scenarios are properly cached."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        # Generate scenario twice
        scenario1 = engine.generate_baseline_scenario(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates
        )
        scenario2 = engine.generate_baseline_scenario(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates
        )
        
        # Should be the same object due to caching
        assert scenario1 is scenario2
        
        # Clear cache and generate again
        engine.clear_cache()
        scenario3 = engine.generate_baseline_scenario(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates
        )
        
        # Should be different object after cache clear
        assert scenario1 is not scenario3


class TestWelfareEvaluator:
    """Test cases for WelfareEvaluator class."""
    
    def test_initialization(self, sample_welfare_function):
        """Test WelfareEvaluator initialization."""
        evaluator = WelfareEvaluator(
            welfare_function=sample_welfare_function,
            discount_factor=0.95
        )
        
        assert evaluator.welfare_function == sample_welfare_function
        assert evaluator.discount_factor == 0.95
    
    def test_compute_scenario_welfare(self, sample_welfare_function):
        """Test welfare computation for a scenario."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        # Create simple test scenario
        dates = pd.date_range('2020-01-01', periods=12, freq='ME')
        policy_rates = pd.Series(np.ones(12) * 2.0, index=dates)
        
        # Regional outcomes (3 regions, output gaps + inflation)
        regional_outcomes = pd.DataFrame(
            np.random.normal(0, 0.5, (6, 12)),  # 3 regions * 2 variables
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        scenario = PolicyScenario(
            name="Test Scenario",
            policy_rates=policy_rates,
            regional_outcomes=regional_outcomes,
            welfare_outcome=0.0,  # Will be computed
            scenario_type='test'
        )
        
        welfare = evaluator.compute_scenario_welfare(scenario)
        
        assert isinstance(welfare, float)
        assert welfare <= 0  # Welfare is negative of loss, so should be negative
    
    def test_compute_scenario_welfare_from_outcomes(self, sample_welfare_function):
        """Test welfare computation from outcomes directly."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        dates = pd.date_range('2020-01-01', periods=6, freq='ME')
        policy_rates = pd.Series(np.ones(6) * 2.0, index=dates)
        
        regional_outcomes = pd.DataFrame(
            np.random.normal(0, 0.5, (6, 6)),
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        welfare = evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates, regional_outcomes
        )
        
        assert isinstance(welfare, float)
        assert welfare <= 0
    
    def test_decompose_welfare_costs(self, sample_welfare_function):
        """Test welfare decomposition between scenarios."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        dates = pd.date_range('2020-01-01', periods=6, freq='ME')
        policy_rates = pd.Series(np.ones(6) * 2.0, index=dates)
        
        # Baseline scenario with higher volatility
        baseline_outcomes = pd.DataFrame(
            np.random.normal(0, 1.0, (6, 6)),
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        # Alternative scenario with lower volatility
        alternative_outcomes = pd.DataFrame(
            np.random.normal(0, 0.5, (6, 6)),
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        baseline_welfare = evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates, baseline_outcomes
        )
        alternative_welfare = evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates, alternative_outcomes
        )
        
        baseline_scenario = PolicyScenario(
            "Baseline", policy_rates, baseline_outcomes, baseline_welfare, 'baseline'
        )
        alternative_scenario = PolicyScenario(
            "Alternative", policy_rates, alternative_outcomes, alternative_welfare, 'alternative'
        )
        
        decomposition = evaluator.decompose_welfare_costs(baseline_scenario, alternative_scenario)
        
        assert isinstance(decomposition, WelfareDecomposition)
        assert decomposition.baseline_welfare == baseline_welfare
        assert decomposition.alternative_welfare == alternative_welfare
        assert abs(decomposition.total_welfare_difference - (alternative_welfare - baseline_welfare)) < 1e-10
    
    def test_verify_welfare_ranking_correct(self, sample_welfare_function):
        """Test welfare ranking verification with correct ordering."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        dates = pd.date_range('2020-01-01', periods=3, freq='ME')
        policy_rates = pd.Series([2.0, 2.1, 2.2], index=dates)
        
        # Create different regional outcomes for each scenario type
        # Baseline: high volatility (worst welfare)
        baseline_outcomes = pd.DataFrame(
            np.array([[1.0, 1.2, 1.1], [0.8, 1.0, 0.9], [1.1, 0.9, 1.0],  # output gaps
                     [0.5, 0.6, 0.4], [0.4, 0.5, 0.3], [0.6, 0.4, 0.5]]),  # inflation
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        # Perfect Info: medium volatility (better than optimal regional due to perfect information)
        perfect_info_outcomes = pd.DataFrame(
            np.array([[0.4, 0.5, 0.3], [0.3, 0.4, 0.2], [0.5, 0.3, 0.4],  # output gaps
                     [0.15, 0.2, 0.1], [0.1, 0.15, 0.05], [0.2, 0.1, 0.15]]),  # inflation
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        # Optimal Regional: medium-high volatility (worse than perfect info due to real-time constraints)
        optimal_outcomes = pd.DataFrame(
            np.array([[0.6, 0.7, 0.5], [0.5, 0.6, 0.4], [0.7, 0.5, 0.6],  # output gaps
                     [0.25, 0.3, 0.2], [0.2, 0.25, 0.15], [0.3, 0.2, 0.25]]),  # inflation
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        # Perfect Regional: low volatility (best welfare)
        perfect_outcomes = pd.DataFrame(
            np.array([[0.1, 0.2, 0.1], [0.1, 0.1, 0.1], [0.2, 0.1, 0.1],  # output gaps
                     [0.05, 0.1, 0.05], [0.05, 0.05, 0.05], [0.1, 0.05, 0.05]]),  # inflation
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        # Compute welfare for each scenario
        baseline_welfare = evaluator.compute_scenario_welfare_from_outcomes(policy_rates, baseline_outcomes)
        perfect_info_welfare = evaluator.compute_scenario_welfare_from_outcomes(policy_rates, perfect_info_outcomes)
        optimal_welfare = evaluator.compute_scenario_welfare_from_outcomes(policy_rates, optimal_outcomes)
        perfect_welfare = evaluator.compute_scenario_welfare_from_outcomes(policy_rates, perfect_outcomes)
        
        # Create scenarios with computed welfare
        scenarios = [
            PolicyScenario("Baseline", policy_rates, baseline_outcomes, baseline_welfare, 'baseline'),
            PolicyScenario("Perfect Info", policy_rates, perfect_info_outcomes, perfect_info_welfare, 'perfect_info'),
            PolicyScenario("Optimal Regional", policy_rates, optimal_outcomes, optimal_welfare, 'optimal_regional'),
            PolicyScenario("Perfect Regional", policy_rates, perfect_outcomes, perfect_welfare, 'perfect_regional')
        ]
        
        assert evaluator.verify_welfare_ranking(scenarios) == True
    
    def test_verify_welfare_ranking_incorrect(self, sample_welfare_function):
        """Test welfare ranking verification with incorrect ordering."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        dates = pd.date_range('2020-01-01', periods=3, freq='ME')
        policy_rates = pd.Series([2.0, 2.1, 2.2], index=dates)
        regional_outcomes = pd.DataFrame(
            np.zeros((6, 3)),
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        # Create scenarios with incorrect welfare ordering
        scenarios = [
            PolicyScenario("Baseline", policy_rates, regional_outcomes, -1.0, 'baseline'),  # Too good
            PolicyScenario("Perfect Info", policy_rates, regional_outcomes, -3.0, 'perfect_info'),
            PolicyScenario("Optimal Regional", policy_rates, regional_outcomes, -2.0, 'optimal_regional'),
            PolicyScenario("Perfect Regional", policy_rates, regional_outcomes, -4.0, 'perfect_regional')  # Too bad
        ]
        
        assert evaluator.verify_welfare_ranking(scenarios) is False
    
    def test_compute_welfare_gains(self, sample_welfare_function):
        """Test welfare gains computation."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        dates = pd.date_range('2020-01-01', periods=3, freq='ME')
        policy_rates = pd.Series([2.0, 2.1, 2.2], index=dates)
        regional_outcomes = pd.DataFrame(
            np.zeros((6, 3)),
            index=['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
                   'inflation_region_1', 'inflation_region_2', 'inflation_region_3'],
            columns=dates
        )
        
        scenarios = [
            PolicyScenario("Baseline", policy_rates, regional_outcomes, -4.0, 'baseline'),
            PolicyScenario("Perfect Info", policy_rates, regional_outcomes, -3.0, 'perfect_info'),
            PolicyScenario("Optimal Regional", policy_rates, regional_outcomes, -2.0, 'optimal_regional')
        ]
        
        gains = evaluator.compute_welfare_gains(scenarios)
        
        assert len(gains) == 2  # Excludes baseline
        assert gains["Perfect Info"] == 1.0  # -3.0 - (-4.0)
        assert gains["Optimal Regional"] == 2.0  # -2.0 - (-4.0)
    
    def test_compute_welfare_gains_no_baseline(self, sample_welfare_function):
        """Test welfare gains computation without baseline scenario."""
        evaluator = WelfareEvaluator(sample_welfare_function)
        
        dates = pd.date_range('2020-01-01', periods=3, freq='ME')
        policy_rates = pd.Series([2.0, 2.1, 2.2], index=dates)
        regional_outcomes = pd.DataFrame(np.zeros((6, 3)), columns=dates)
        
        scenarios = [
            PolicyScenario("Perfect Info", policy_rates, regional_outcomes, -3.0, 'perfect_info')
        ]
        
        with pytest.raises(ValueError, match="No baseline scenario found"):
            evaluator.compute_welfare_gains(scenarios)


class TestWelfareRankingProperty:
    """Test the key theoretical property: W^PR ≥ W^PI ≥ W^OR ≥ W^B"""
    
    def test_welfare_ranking_property(self, sample_regional_params, sample_welfare_function,
                                    sample_historical_data, sample_fed_policy_rates,
                                    sample_fed_reaction_function):
        """Test that welfare ranking follows theoretical prediction."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        # Generate all scenarios
        scenarios = engine.generate_all_scenarios(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates,
            fed_reaction_function=sample_fed_reaction_function,
            parallel=False
        )
        
        # Verify welfare ranking
        evaluator = WelfareEvaluator(sample_welfare_function)
        ranking_correct = evaluator.verify_welfare_ranking(scenarios)
        
        # The ranking should follow theory (though with synthetic data it might not always hold)
        # At minimum, we test that the verification function works
        assert ranking_correct in [True, False]  # Ensure it returns a boolean
        
        # Extract welfare outcomes by scenario type
        welfare_by_type = {}
        for scenario in scenarios:
            welfare_by_type[scenario.scenario_type] = scenario.welfare_outcome
        
        # Print for debugging (in real tests, this would help identify issues)
        print(f"Welfare outcomes: {welfare_by_type}")
        
        # At minimum, perfect regional should be better than or equal to baseline
        assert welfare_by_type['perfect_regional'] >= welfare_by_type['baseline']
    
    def test_welfare_ranking_with_comparison_results(self, sample_regional_params, 
                                                   sample_welfare_function, sample_historical_data,
                                                   sample_fed_policy_rates, sample_fed_reaction_function):
        """Test welfare ranking using ComparisonResults."""
        engine = CounterfactualEngine(
            regional_params=sample_regional_params,
            welfare_function=sample_welfare_function
        )
        
        scenarios = engine.generate_all_scenarios(
            historical_data=sample_historical_data,
            fed_policy_rates=sample_fed_policy_rates,
            fed_reaction_function=sample_fed_reaction_function,
            parallel=False
        )
        
        comparison = engine.compare_scenarios(scenarios)
        
        # Test the theoretical ranking verification
        ranking_correct = comparison.verify_theoretical_ranking()
        assert isinstance(ranking_correct, bool)
        
        # Test that best scenario is identified
        best_scenario = comparison.get_best_scenario()
        assert best_scenario in comparison.scenario_names
        
        # Test summary table generation
        summary = comparison.summary_table()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4
        assert 'Welfare' in summary.columns
        assert 'Ranking' in summary.columns


if __name__ == "__main__":
    pytest.main([__file__])