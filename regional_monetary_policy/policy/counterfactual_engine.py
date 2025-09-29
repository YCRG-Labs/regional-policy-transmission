"""
Counterfactual analysis engine for regional monetary policy.

This module implements the generation and evaluation of alternative policy scenarios
to assess the welfare implications of different monetary policy approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

from ..econometric.models import RegionalParameters
from ..data.models import RegionalDataset
from .models import PolicyScenario, ComparisonResults, WelfareDecomposition
from .optimal_policy import OptimalPolicyCalculator, WelfareFunction
from ..exceptions import WelfareCalculationError


class CounterfactualEngine:
    """
    Generates and evaluates counterfactual policy scenarios.
    
    This class implements the four key policy scenarios from the theoretical framework:
    1. Baseline (B): Historical Fed policy with real-time information
    2. Perfect Information (PI): Fed policy with perfect information
    3. Optimal Regional (OR): Optimal policy with real-time information
    4. Perfect Regional (PR): Optimal policy with perfect information
    """
    
    def __init__(self,
                 regional_params: RegionalParameters,
                 welfare_function: WelfareFunction,
                 discount_factor: float = 0.99,
                 n_workers: Optional[int] = None):
        """
        Initialize the counterfactual analysis engine.
        
        Args:
            regional_params: Estimated regional structural parameters
            welfare_function: Social welfare function specification
            discount_factor: Discount factor for intertemporal welfare
            n_workers: Number of parallel workers (None for auto-detect)
        """
        self.regional_params = regional_params
        self.welfare_function = welfare_function
        self.discount_factor = discount_factor
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Initialize optimal policy calculator
        self.optimal_calculator = OptimalPolicyCalculator(
            regional_params=regional_params,
            welfare_function=welfare_function,
            discount_factor=discount_factor
        )
        
        # Store scenario cache for efficiency
        self._scenario_cache = {}
    
    def generate_baseline_scenario(self, 
                                 historical_data: RegionalDataset,
                                 fed_policy_rates: pd.Series) -> PolicyScenario:
        """
        Generate baseline scenario using historical Fed policy with real-time information.
        
        Args:
            historical_data: Historical regional economic data
            fed_policy_rates: Actual Fed policy rates
            
        Returns:
            PolicyScenario representing the baseline
        """
        cache_key = f"baseline_{hash(str(fed_policy_rates.values))}"
        if cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]
        
        # Use real-time data estimates for regional outcomes
        regional_outcomes = self._simulate_regional_outcomes_realtime(
            policy_rates=fed_policy_rates,
            historical_data=historical_data,
            use_real_time=True
        )
        
        # Compute welfare outcome
        welfare_evaluator = WelfareEvaluator(self.welfare_function, self.discount_factor)
        welfare_outcome = welfare_evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates=fed_policy_rates,
            regional_outcomes=regional_outcomes
        )
        
        scenario = PolicyScenario(
            name="Baseline (Historical Fed Policy)",
            policy_rates=fed_policy_rates,
            regional_outcomes=regional_outcomes,
            welfare_outcome=welfare_outcome,
            scenario_type='baseline',
            information_set={'type': 'real_time', 'data_vintage': 'historical'}
        )
        
        self._scenario_cache[cache_key] = scenario
        return scenario
    
    def generate_perfect_info_scenario(self, 
                                     historical_data: RegionalDataset,
                                     fed_reaction_function: Dict[str, float]) -> PolicyScenario:
        """
        Generate perfect information scenario using Fed reaction function with perfect data.
        
        Args:
            historical_data: Historical regional economic data
            fed_reaction_function: Estimated Fed reaction function parameters
            
        Returns:
            PolicyScenario with Fed policy under perfect information
        """
        cache_key = f"perfect_info_{hash(str(fed_reaction_function))}"
        if cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]
        
        # Generate Fed policy rates using perfect information
        perfect_info_rates = self._generate_fed_policy_perfect_info(
            historical_data=historical_data,
            reaction_function=fed_reaction_function
        )
        
        # Simulate regional outcomes under perfect information policy
        regional_outcomes = self._simulate_regional_outcomes_realtime(
            policy_rates=perfect_info_rates,
            historical_data=historical_data,
            use_real_time=False  # Use perfect information
        )
        
        # Compute welfare outcome
        welfare_evaluator = WelfareEvaluator(self.welfare_function, self.discount_factor)
        welfare_outcome = welfare_evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates=perfect_info_rates,
            regional_outcomes=regional_outcomes
        )
        
        scenario = PolicyScenario(
            name="Perfect Information (Fed Policy)",
            policy_rates=perfect_info_rates,
            regional_outcomes=regional_outcomes,
            welfare_outcome=welfare_outcome,
            scenario_type='perfect_info',
            policy_parameters=fed_reaction_function,
            information_set={'type': 'perfect_information'}
        )
        
        self._scenario_cache[cache_key] = scenario
        return scenario
    
    def generate_optimal_regional_scenario(self, 
                                         historical_data: RegionalDataset) -> PolicyScenario:
        """
        Generate optimal regional scenario with real-time information constraints.
        
        Args:
            historical_data: Historical regional economic data
            
        Returns:
            PolicyScenario with optimal policy under real-time information
        """
        cache_key = f"optimal_regional_{id(historical_data)}"
        if cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]
        
        # Generate optimal policy rates using real-time information
        optimal_rates_realtime = self._generate_optimal_policy_realtime(historical_data)
        
        # Simulate regional outcomes under optimal policy with real-time constraints
        regional_outcomes = self._simulate_regional_outcomes_realtime(
            policy_rates=optimal_rates_realtime,
            historical_data=historical_data,
            use_real_time=True
        )
        
        # Compute welfare outcome
        welfare_evaluator = WelfareEvaluator(self.welfare_function, self.discount_factor)
        welfare_outcome = welfare_evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates=optimal_rates_realtime,
            regional_outcomes=regional_outcomes
        )
        
        scenario = PolicyScenario(
            name="Optimal Regional (Real-time Info)",
            policy_rates=optimal_rates_realtime,
            regional_outcomes=regional_outcomes,
            welfare_outcome=welfare_outcome,
            scenario_type='optimal_regional',
            policy_parameters=self.optimal_calculator.optimal_coefficients,
            regional_weights=self.optimal_calculator.optimal_regional_weights,
            information_set={'type': 'real_time', 'optimal_weights': True}
        )
        
        self._scenario_cache[cache_key] = scenario
        return scenario
    
    def generate_perfect_regional_scenario(self, 
                                         historical_data: RegionalDataset) -> PolicyScenario:
        """
        Generate perfect regional scenario with optimal policy and perfect information.
        
        Args:
            historical_data: Historical regional economic data
            
        Returns:
            PolicyScenario with optimal policy under perfect information
        """
        cache_key = f"perfect_regional_{id(historical_data)}"
        if cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]
        
        # Generate optimal policy rates using perfect information
        optimal_rates_perfect = self._generate_optimal_policy_perfect_info(historical_data)
        
        # Simulate regional outcomes under optimal policy with perfect information
        regional_outcomes = self._simulate_regional_outcomes_realtime(
            policy_rates=optimal_rates_perfect,
            historical_data=historical_data,
            use_real_time=False  # Use perfect information
        )
        
        # Compute welfare outcome
        welfare_evaluator = WelfareEvaluator(self.welfare_function, self.discount_factor)
        welfare_outcome = welfare_evaluator.compute_scenario_welfare_from_outcomes(
            policy_rates=optimal_rates_perfect,
            regional_outcomes=regional_outcomes
        )
        
        scenario = PolicyScenario(
            name="Perfect Regional (Optimal + Perfect Info)",
            policy_rates=optimal_rates_perfect,
            regional_outcomes=regional_outcomes,
            welfare_outcome=welfare_outcome,
            scenario_type='perfect_regional',
            policy_parameters=self.optimal_calculator.optimal_coefficients,
            regional_weights=self.optimal_calculator.optimal_regional_weights,
            information_set={'type': 'perfect_information', 'optimal_weights': True}
        )
        
        self._scenario_cache[cache_key] = scenario
        return scenario
    
    def generate_all_scenarios(self,
                             historical_data: RegionalDataset,
                             fed_policy_rates: pd.Series,
                             fed_reaction_function: Dict[str, float],
                             parallel: bool = True) -> List[PolicyScenario]:
        """
        Generate all four counterfactual scenarios.
        
        Args:
            historical_data: Historical regional economic data
            fed_policy_rates: Actual Fed policy rates
            fed_reaction_function: Estimated Fed reaction function parameters
            parallel: Whether to use parallel processing
            
        Returns:
            List of all four policy scenarios
        """
        if parallel and self.n_workers > 1:
            return self._generate_scenarios_parallel(
                historical_data, fed_policy_rates, fed_reaction_function
            )
        else:
            return self._generate_scenarios_sequential(
                historical_data, fed_policy_rates, fed_reaction_function
            )
    
    def _generate_scenarios_parallel(self,
                                   historical_data: RegionalDataset,
                                   fed_policy_rates: pd.Series,
                                   fed_reaction_function: Dict[str, float]) -> List[PolicyScenario]:
        """Generate scenarios using parallel processing."""
        
        # Define scenario generation functions
        scenario_functions = [
            partial(self.generate_baseline_scenario, historical_data, fed_policy_rates),
            partial(self.generate_perfect_info_scenario, historical_data, fed_reaction_function),
            partial(self.generate_optimal_regional_scenario, historical_data),
            partial(self.generate_perfect_regional_scenario, historical_data)
        ]
        
        scenarios = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all scenario generation tasks
            future_to_scenario = {
                executor.submit(func): i for i, func in enumerate(scenario_functions)
            }
            
            # Collect results as they complete
            scenario_results = [None] * 4
            for future in as_completed(future_to_scenario):
                scenario_idx = future_to_scenario[future]
                try:
                    scenario_results[scenario_idx] = future.result()
                except Exception as exc:
                    raise WelfareCalculationError(f"Scenario {scenario_idx} generation failed: {exc}")
        
        return scenario_results
    
    def _generate_scenarios_sequential(self,
                                     historical_data: RegionalDataset,
                                     fed_policy_rates: pd.Series,
                                     fed_reaction_function: Dict[str, float]) -> List[PolicyScenario]:
        """Generate scenarios sequentially."""
        
        scenarios = [
            self.generate_baseline_scenario(historical_data, fed_policy_rates),
            self.generate_perfect_info_scenario(historical_data, fed_reaction_function),
            self.generate_optimal_regional_scenario(historical_data),
            self.generate_perfect_regional_scenario(historical_data)
        ]
        
        return scenarios
    
    def compare_scenarios(self, scenarios: List[PolicyScenario]) -> ComparisonResults:
        """
        Compare multiple policy scenarios and verify theoretical welfare ranking.
        
        Args:
            scenarios: List of policy scenarios to compare
            
        Returns:
            ComparisonResults with welfare rankings and comparisons
        """
        if len(scenarios) != 4:
            raise ValueError("Expected exactly 4 scenarios for comparison")
        
        # Extract scenario information
        scenario_names = [s.name for s in scenarios]
        welfare_outcomes = [s.welfare_outcome for s in scenarios]
        
        # Compute welfare ranking (1 = best welfare, i.e., lowest loss)
        welfare_ranking = np.argsort(welfare_outcomes) + 1
        
        # Compute pairwise comparisons
        pairwise_comparisons = {}
        for i, scenario_i in enumerate(scenarios):
            pairwise_comparisons[scenario_i.name] = {}
            for j, scenario_j in enumerate(scenarios):
                if i != j:
                    welfare_diff = scenario_j.welfare_outcome - scenario_i.welfare_outcome
                    pairwise_comparisons[scenario_i.name][scenario_j.name] = welfare_diff
        
        return ComparisonResults(
            scenario_names=scenario_names,
            welfare_outcomes=welfare_outcomes,
            welfare_ranking=welfare_ranking.tolist(),
            pairwise_comparisons=pairwise_comparisons
        )
    
    def _generate_fed_policy_perfect_info(self,
                                        historical_data: RegionalDataset,
                                        reaction_function: Dict[str, float]) -> pd.Series:
        """Generate Fed policy rates using perfect information."""
        
        # Extract perfect information data (final revised estimates)
        output_gaps = historical_data.output_gaps.mean(axis=0)  # Aggregate output gap
        inflation_rates = historical_data.inflation_rates.mean(axis=0)  # Aggregate inflation
        
        # Apply Fed reaction function
        output_coeff = reaction_function.get('output_gap_response', 0.5)
        inflation_coeff = reaction_function.get('inflation_response', 1.5)
        
        fed_rates = output_coeff * output_gaps + inflation_coeff * inflation_rates
        
        return pd.Series(fed_rates, index=output_gaps.index, name='fed_policy_rate')
    
    def _generate_optimal_policy_realtime(self, 
                                        historical_data: RegionalDataset) -> pd.Series:
        """Generate optimal policy using real-time information constraints."""
        
        # Use real-time estimates where available
        if historical_data.real_time_estimates:
            # Construct real-time regional conditions
            realtime_conditions = self._construct_realtime_conditions(historical_data)
        else:
            # Fallback to final data with noise to simulate real-time uncertainty
            realtime_conditions = self._add_realtime_noise(historical_data)
        
        # Compute optimal policy path
        optimal_rates = self.optimal_calculator.compute_optimal_rate_path(realtime_conditions)
        
        return optimal_rates
    
    def _generate_optimal_policy_perfect_info(self, 
                                            historical_data: RegionalDataset) -> pd.Series:
        """Generate optimal policy using perfect information."""
        
        # Construct perfect information regional conditions
        perfect_conditions = pd.DataFrame({
            **{f'output_gap_region_{i+1}': historical_data.output_gaps.iloc[i] 
               for i in range(historical_data.n_regions)},
            **{f'inflation_region_{i+1}': historical_data.inflation_rates.iloc[i] 
               for i in range(historical_data.n_regions)}
        })
        
        # Compute optimal policy path
        optimal_rates = self.optimal_calculator.compute_optimal_rate_path(perfect_conditions)
        
        return optimal_rates
    
    def _simulate_regional_outcomes_realtime(self,
                                           policy_rates: pd.Series,
                                           historical_data: RegionalDataset,
                                           use_real_time: bool = True) -> pd.DataFrame:
        """
        Simulate regional outcomes under given policy path.
        
        This implements a simplified version of the regional equilibrium system.
        A full implementation would solve the complete DSGE model.
        """
        n_regions = historical_data.n_regions
        n_periods = len(policy_rates)
        
        # Initialize outcome arrays
        output_gaps = np.zeros((n_regions, n_periods))
        inflation_rates = np.zeros((n_regions, n_periods))
        
        # Get regional parameters
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        psi = self.regional_params.psi
        phi = self.regional_params.phi
        
        # Simulate period by period
        for t, (date, rate) in enumerate(policy_rates.items()):
            
            # Direct interest rate effects on output gaps
            direct_effects = -sigma * rate
            
            # Add spillover effects from other regions (simplified)
            spillover_effects = np.zeros(n_regions)
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j:
                        spillover_effects[i] += psi[i] * direct_effects[j] / (n_regions - 1)
            
            # Total output gap effects
            output_gaps[:, t] = direct_effects + spillover_effects
            
            # Phillips curve effects on inflation
            inflation_rates[:, t] = -kappa * output_gaps[:, t]
            
            # Add spillover effects on inflation
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j:
                        inflation_rates[i, t] += phi[i] * inflation_rates[j, t] / (n_regions - 1)
        
        # Add measurement noise if using real-time data
        if use_real_time:
            output_gaps += np.random.normal(0, 0.1, output_gaps.shape)
            inflation_rates += np.random.normal(0, 0.05, inflation_rates.shape)
        
        # Combine into single DataFrame
        outcome_data = np.vstack([output_gaps, inflation_rates])
        
        # Create proper index
        region_names = [f"output_gap_region_{i+1}" for i in range(n_regions)]
        region_names += [f"inflation_region_{i+1}" for i in range(n_regions)]
        
        return pd.DataFrame(outcome_data, 
                          index=region_names, 
                          columns=policy_rates.index)
    
    def _construct_realtime_conditions(self, 
                                     historical_data: RegionalDataset) -> pd.DataFrame:
        """Construct regional conditions using real-time data estimates."""
        
        # Use real-time estimates where available
        realtime_output = historical_data.real_time_estimates.get('output_gaps', 
                                                                historical_data.output_gaps)
        realtime_inflation = historical_data.real_time_estimates.get('inflation_rates',
                                                                   historical_data.inflation_rates)
        
        # Construct conditions DataFrame
        conditions = pd.DataFrame({
            **{f'output_gap_region_{i+1}': realtime_output.iloc[i] 
               for i in range(historical_data.n_regions)},
            **{f'inflation_region_{i+1}': realtime_inflation.iloc[i] 
               for i in range(historical_data.n_regions)}
        })
        
        return conditions
    
    def _add_realtime_noise(self, historical_data: RegionalDataset) -> pd.DataFrame:
        """Add noise to simulate real-time data uncertainty."""
        
        # Add measurement noise to simulate real-time uncertainty
        noisy_output = historical_data.output_gaps + np.random.normal(0, 0.2, 
                                                                     historical_data.output_gaps.shape)
        noisy_inflation = historical_data.inflation_rates + np.random.normal(0, 0.1,
                                                                           historical_data.inflation_rates.shape)
        
        # Construct conditions DataFrame
        conditions = pd.DataFrame({
            **{f'output_gap_region_{i+1}': noisy_output.iloc[i] 
               for i in range(historical_data.n_regions)},
            **{f'inflation_region_{i+1}': noisy_inflation.iloc[i] 
               for i in range(historical_data.n_regions)}
        })
        
        return conditions
    
    def clear_cache(self):
        """Clear the scenario cache."""
        self._scenario_cache.clear()


class WelfareEvaluator:
    """
    Evaluates welfare outcomes across policy scenarios.
    
    This class implements the social welfare function and provides methods
    for computing welfare outcomes and decomposing welfare differences.
    """
    
    def __init__(self,
                 welfare_function: WelfareFunction,
                 discount_factor: float = 0.99):
        """
        Initialize the welfare evaluator.
        
        Args:
            welfare_function: Social welfare function specification
            discount_factor: Discount factor for intertemporal welfare
        """
        self.welfare_function = welfare_function
        self.discount_factor = discount_factor
    
    def compute_scenario_welfare(self, scenario: PolicyScenario) -> float:
        """
        Compute total welfare for a policy scenario.
        
        Args:
            scenario: Policy scenario to evaluate
            
        Returns:
            Total discounted welfare (negative of welfare loss)
        """
        return self.compute_scenario_welfare_from_outcomes(
            scenario.policy_rates,
            scenario.regional_outcomes
        )
    
    def compute_scenario_welfare_from_outcomes(self,
                                             policy_rates: pd.Series,
                                             regional_outcomes: pd.DataFrame) -> float:
        """
        Compute welfare from policy rates and regional outcomes.
        
        Args:
            policy_rates: Time series of policy rates
            regional_outcomes: Regional output gaps and inflation outcomes
            
        Returns:
            Total discounted welfare (negative of welfare loss)
        """
        n_periods = len(policy_rates)
        n_regions = len(self.welfare_function.regional_weights)
        
        total_welfare_loss = 0.0
        
        for t, date in enumerate(policy_rates.index):
            # Get period outcomes
            period_outcomes = regional_outcomes.loc[:, date]
            
            # Split into output gaps and inflation
            output_gaps = period_outcomes[:n_regions].values
            inflation_rates = period_outcomes[n_regions:].values
            
            # Compute period welfare loss
            period_loss = self._compute_period_welfare_loss(output_gaps, inflation_rates)
            
            # Add discounted loss
            discount_factor = self.discount_factor ** t
            total_welfare_loss += discount_factor * period_loss
        
        # Return negative of loss (higher welfare is better)
        return -total_welfare_loss
    
    def decompose_welfare_costs(self,
                              baseline: PolicyScenario,
                              alternative: PolicyScenario) -> WelfareDecomposition:
        """
        Decompose welfare differences between two scenarios.
        
        Args:
            baseline: Baseline policy scenario
            alternative: Alternative policy scenario
            
        Returns:
            WelfareDecomposition with component breakdown
        """
        # Compute total welfare difference
        total_difference = alternative.welfare_outcome - baseline.welfare_outcome
        
        # Decompose into output gap and inflation components
        output_component = self._compute_output_gap_component(baseline, alternative)
        inflation_component = self._compute_inflation_component(baseline, alternative)
        
        # Regional distribution component
        regional_component = total_difference - output_component - inflation_component
        
        return WelfareDecomposition(
            total_welfare_difference=total_difference,
            output_gap_component=output_component,
            inflation_component=inflation_component,
            regional_distribution_component=regional_component,
            baseline_welfare=baseline.welfare_outcome,
            alternative_welfare=alternative.welfare_outcome
        )
    
    def verify_welfare_ranking(self, scenarios: List[PolicyScenario]) -> bool:
        """
        Verify that welfare ranking follows theoretical prediction:
        W^PR ≥ W^PI ≥ W^OR ≥ W^B
        
        Args:
            scenarios: List of policy scenarios
            
        Returns:
            True if ranking is consistent with theory
        """
        # Find scenarios by type
        scenario_dict = {s.scenario_type: s for s in scenarios}
        
        required_types = ['baseline', 'perfect_info', 'optimal_regional', 'perfect_regional']
        if not all(stype in scenario_dict for stype in required_types):
            return False
        
        # Check welfare ordering (higher welfare is better, so we expect PR >= PI >= OR >= B)
        w_b = scenario_dict['baseline'].welfare_outcome
        w_pi = scenario_dict['perfect_info'].welfare_outcome
        w_or = scenario_dict['optimal_regional'].welfare_outcome
        w_pr = scenario_dict['perfect_regional'].welfare_outcome
        
        # Allow for small numerical errors
        tolerance = 1e-10
        return (w_pr >= w_pi - tolerance and 
                w_pi >= w_or - tolerance and 
                w_or >= w_b - tolerance)
    
    def compute_welfare_gains(self, scenarios: List[PolicyScenario]) -> Dict[str, float]:
        """
        Compute welfare gains relative to baseline scenario.
        
        Args:
            scenarios: List of policy scenarios
            
        Returns:
            Dictionary with welfare gains for each scenario
        """
        # Find baseline scenario
        baseline = None
        for scenario in scenarios:
            if scenario.scenario_type == 'baseline':
                baseline = scenario
                break
        
        if baseline is None:
            raise ValueError("No baseline scenario found")
        
        # Compute gains relative to baseline
        welfare_gains = {}
        for scenario in scenarios:
            if scenario.scenario_type != 'baseline':
                gain = scenario.welfare_outcome - baseline.welfare_outcome
                welfare_gains[scenario.name] = gain
        
        return welfare_gains
    
    def _compute_period_welfare_loss(self,
                                   output_gaps: np.ndarray,
                                   inflation_rates: np.ndarray) -> float:
        """Compute welfare loss for a single period."""
        
        w = self.welfare_function.regional_weights
        
        if self.welfare_function.loss_function == 'quadratic':
            # Standard quadratic loss function
            output_loss = np.sum(w * output_gaps**2)
            inflation_loss = np.sum(w * inflation_rates**2)
            
            total_loss = (
                self.welfare_function.output_gap_weight * output_loss +
                self.welfare_function.inflation_weight * inflation_loss
            )
        
        elif self.welfare_function.loss_function == 'asymmetric':
            # Asymmetric loss function
            output_loss = np.sum(w * np.where(output_gaps < 0, 2 * output_gaps**2, output_gaps**2))
            inflation_loss = np.sum(w * inflation_rates**2)
            
            total_loss = (
                self.welfare_function.output_gap_weight * output_loss +
                self.welfare_function.inflation_weight * inflation_loss
            )
        
        return total_loss
    
    def _compute_output_gap_component(self,
                                    baseline: PolicyScenario,
                                    alternative: PolicyScenario) -> float:
        """Compute welfare difference due to output gap stabilization."""
        
        n_regions = len(self.welfare_function.regional_weights)
        w = self.welfare_function.regional_weights
        
        baseline_output_loss = 0.0
        alternative_output_loss = 0.0
        
        for t, date in enumerate(baseline.policy_rates.index):
            # Extract output gaps (first n_regions rows)
            baseline_gaps = baseline.regional_outcomes.iloc[:n_regions, t].values
            alternative_gaps = alternative.regional_outcomes.iloc[:n_regions, t].values
            
            # Compute period losses
            baseline_period_loss = np.sum(w * baseline_gaps**2)
            alternative_period_loss = np.sum(w * alternative_gaps**2)
            
            # Add discounted losses
            discount = self.discount_factor ** t
            baseline_output_loss += discount * baseline_period_loss
            alternative_output_loss += discount * alternative_period_loss
        
        return self.welfare_function.output_gap_weight * (alternative_output_loss - baseline_output_loss)
    
    def _compute_inflation_component(self,
                                   baseline: PolicyScenario,
                                   alternative: PolicyScenario) -> float:
        """Compute welfare difference due to inflation stabilization."""
        
        n_regions = len(self.welfare_function.regional_weights)
        w = self.welfare_function.regional_weights
        
        baseline_inflation_loss = 0.0
        alternative_inflation_loss = 0.0
        
        for t, date in enumerate(baseline.policy_rates.index):
            # Extract inflation rates (last n_regions rows)
            baseline_inflation = baseline.regional_outcomes.iloc[n_regions:, t].values
            alternative_inflation = alternative.regional_outcomes.iloc[n_regions:, t].values
            
            # Compute period losses
            baseline_period_loss = np.sum(w * baseline_inflation**2)
            alternative_period_loss = np.sum(w * alternative_inflation**2)
            
            # Add discounted losses
            discount = self.discount_factor ** t
            baseline_inflation_loss += discount * baseline_period_loss
            alternative_inflation_loss += discount * alternative_period_loss
        
        return self.welfare_function.inflation_weight * (alternative_inflation_loss - baseline_inflation_loss)