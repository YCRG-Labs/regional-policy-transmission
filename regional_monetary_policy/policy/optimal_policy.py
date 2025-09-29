"""
Optimal policy calculation for welfare-maximizing monetary policy.

This module implements the computation of optimal monetary policy rates
that maximize social welfare given regional heterogeneity and spillover effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize

from ..econometric.models import RegionalParameters
from .models import PolicyScenario


@dataclass
class WelfareFunction:
    """
    Social welfare function specification.
    """
    
    output_gap_weight: float = 1.0  # Weight on output gap stabilization
    inflation_weight: float = 1.0   # Weight on inflation stabilization
    regional_weights: np.ndarray = None  # Regional welfare weights
    loss_function: str = 'quadratic'  # 'quadratic' or 'asymmetric'
    
    def __post_init__(self):
        """Validate welfare function parameters."""
        if self.regional_weights is not None:
            if not np.isclose(np.sum(self.regional_weights), 1.0):
                raise ValueError("Regional weights must sum to 1")
        
        if self.loss_function not in ['quadratic', 'asymmetric']:
            raise ValueError("Loss function must be 'quadratic' or 'asymmetric'")


class OptimalPolicyCalculator:
    """
    Computes welfare-maximizing monetary policy given regional heterogeneity.
    
    This class implements the theoretical framework for optimal monetary policy
    in a multi-region economy with spillover effects and heterogeneous parameters.
    """
    
    def __init__(self,
                 regional_params: RegionalParameters,
                 welfare_function: WelfareFunction,
                 discount_factor: float = 0.99):
        """
        Initialize the optimal policy calculator.
        
        Args:
            regional_params: Estimated regional structural parameters
            welfare_function: Social welfare function specification
            discount_factor: Discount factor for intertemporal welfare
        """
        self.regional_params = regional_params
        self.welfare_function = welfare_function
        self.discount_factor = discount_factor
        
        # Set default regional weights if not provided
        if self.welfare_function.regional_weights is None:
            n_regions = regional_params.n_regions
            self.welfare_function.regional_weights = np.ones(n_regions) / n_regions
        
        # Validate dimensions
        if len(self.welfare_function.regional_weights) != regional_params.n_regions:
            raise ValueError("Regional weights must match number of regions")
        
        # Precompute optimal policy coefficients
        self._compute_optimal_coefficients()
    
    def _compute_optimal_coefficients(self):
        """
        Compute optimal policy response coefficients.
        
        This solves the welfare maximization problem to derive optimal
        responses to regional output gaps and inflation.
        """
        # Get regional parameters
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        psi = self.regional_params.psi
        phi = self.regional_params.phi
        beta = self.regional_params.beta
        
        # Get welfare weights
        w = self.welfare_function.regional_weights
        
        # Compute optimal regional weights for policy
        # These weights maximize welfare given regional heterogeneity
        self.optimal_regional_weights = self._compute_optimal_regional_weights()
        
        # Compute optimal response coefficients
        # These come from the first-order conditions of the welfare maximization problem
        
        # Denominator terms for normalization
        denom_output = np.sum(w * sigma * kappa / (1 + kappa * phi))
        denom_inflation = np.sum(w * sigma / (1 + kappa * phi))
        
        # Optimal response to aggregate output gap
        self.optimal_output_response = denom_output / np.sum(w * sigma)
        
        # Optimal response to aggregate inflation
        self.optimal_inflation_response = denom_inflation / np.sum(w * sigma)
        
        # Store for external access
        self.optimal_coefficients = {
            'output': self.optimal_output_response,
            'inflation': self.optimal_inflation_response
        }
    
    def _compute_optimal_regional_weights(self) -> np.ndarray:
        """
        Compute optimal regional weights for monetary policy.
        
        Returns:
            Optimal weights that maximize social welfare
        """
        # Get parameters
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        w = self.welfare_function.regional_weights
        
        # Optimal weights combine social preferences with regional sensitivities
        # This reflects both equity concerns (social weights) and efficiency (sensitivities)
        sensitivity_adjusted_weights = w * sigma * kappa
        
        # Normalize to sum to 1
        return sensitivity_adjusted_weights / np.sum(sensitivity_adjusted_weights)
    
    def compute_optimal_rate(self, regional_conditions: pd.DataFrame) -> float:
        """
        Compute optimal policy rate for given regional economic conditions.
        
        Args:
            regional_conditions: DataFrame with regional output gaps and inflation
            
        Returns:
            Optimal policy interest rate
        """
        # Extract regional output gaps and inflation
        output_gaps = self._extract_regional_output_gaps(regional_conditions)
        inflation_rates = self._extract_regional_inflation(regional_conditions)
        
        # Compute weighted aggregates using optimal weights
        weighted_output_gap = np.sum(self.optimal_regional_weights * output_gaps)
        weighted_inflation = np.sum(self.optimal_regional_weights * inflation_rates)
        
        # Apply optimal response coefficients
        optimal_rate = (
            self.optimal_output_response * weighted_output_gap +
            self.optimal_inflation_response * weighted_inflation
        )
        
        return optimal_rate
    
    def compute_optimal_rate_path(self, 
                                regional_data: pd.DataFrame,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.Series:
        """
        Compute optimal policy rate path over time.
        
        Args:
            regional_data: Time series of regional economic conditions
            start_date: Start date for computation (None for full sample)
            end_date: End date for computation (None for full sample)
            
        Returns:
            Time series of optimal policy rates
        """
        # Subset data if dates provided
        if start_date is not None or end_date is not None:
            regional_data = regional_data.loc[start_date:end_date]
        
        optimal_rates = []
        
        for date in regional_data.index:
            period_data = regional_data.loc[[date]]  # Single period as DataFrame
            optimal_rate = self.compute_optimal_rate(period_data)
            optimal_rates.append(optimal_rate)
        
        return pd.Series(optimal_rates, index=regional_data.index, name='optimal_rate')
    
    def evaluate_welfare_loss(self,
                            policy_path: pd.Series,
                            regional_outcomes: pd.DataFrame) -> float:
        """
        Evaluate welfare loss for a given policy path and outcomes.
        
        Args:
            policy_path: Time series of policy rates
            regional_outcomes: Regional output gaps and inflation outcomes
            
        Returns:
            Total discounted welfare loss
        """
        # Extract regional output gaps and inflation over time
        n_periods = len(policy_path)
        n_regions = self.regional_params.n_regions
        
        total_welfare_loss = 0.0
        
        for t, date in enumerate(policy_path.index):
            # Get period outcomes
            period_outcomes = regional_outcomes.loc[:, date]
            
            # Split into output gaps and inflation (assuming first half is output gaps)
            output_gaps = period_outcomes[:n_regions].values
            inflation_rates = period_outcomes[n_regions:].values
            
            # Compute period welfare loss
            period_loss = self._compute_period_welfare_loss(output_gaps, inflation_rates)
            
            # Add discounted loss
            discount_factor = self.discount_factor ** t
            total_welfare_loss += discount_factor * period_loss
        
        return total_welfare_loss
    
    def _compute_period_welfare_loss(self,
                                   output_gaps: np.ndarray,
                                   inflation_rates: np.ndarray) -> float:
        """
        Compute welfare loss for a single period.
        
        Args:
            output_gaps: Regional output gaps
            inflation_rates: Regional inflation rates
            
        Returns:
            Period welfare loss
        """
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
            # Asymmetric loss function (penalize negative output gaps more)
            output_loss = np.sum(w * np.where(output_gaps < 0, 2 * output_gaps**2, output_gaps**2))
            inflation_loss = np.sum(w * inflation_rates**2)
            
            total_loss = (
                self.welfare_function.output_gap_weight * output_loss +
                self.welfare_function.inflation_weight * inflation_loss
            )
        
        return total_loss
    
    def compute_welfare_optimal_weights(self,
                                      constraint_type: str = 'sum_to_one') -> np.ndarray:
        """
        Compute welfare-optimal regional weights subject to constraints.
        
        Args:
            constraint_type: Type of constraint ('sum_to_one', 'population_weighted')
            
        Returns:
            Optimal regional weights
        """
        if constraint_type == 'sum_to_one':
            return self.optimal_regional_weights
        
        elif constraint_type == 'population_weighted':
            # Adjust for population weights while maintaining optimality
            pop_weights = self.welfare_function.regional_weights
            optimal_weights = self.optimal_regional_weights
            
            # Combine population and efficiency considerations
            combined_weights = pop_weights * optimal_weights
            return combined_weights / np.sum(combined_weights)
        
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
    
    def analyze_policy_tradeoffs(self,
                               regional_conditions: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze policy tradeoffs across regions.
        
        Args:
            regional_conditions: Current regional economic conditions
            
        Returns:
            Dictionary with tradeoff analysis
        """
        output_gaps = self._extract_regional_output_gaps(regional_conditions)
        inflation_rates = self._extract_regional_inflation(regional_conditions)
        
        # Compute regional welfare losses under current conditions
        regional_losses = []
        for i in range(self.regional_params.n_regions):
            loss = (
                self.welfare_function.output_gap_weight * output_gaps[i]**2 +
                self.welfare_function.inflation_weight * inflation_rates[i]**2
            )
            regional_losses.append(loss)
        
        # Compute optimal rate and its regional impacts
        optimal_rate = self.compute_optimal_rate(regional_conditions)
        
        # Analyze how optimal policy affects each region
        regional_impacts = self._compute_regional_policy_impacts(optimal_rate, regional_conditions)
        
        return {
            'optimal_rate': optimal_rate,
            'regional_losses': regional_losses,
            'regional_impacts': regional_impacts,
            'total_welfare_loss': np.sum(self.welfare_function.regional_weights * regional_losses),
            'cross_regional_variance': np.var(regional_losses),
            'most_affected_region': np.argmax(regional_losses),
            'least_affected_region': np.argmin(regional_losses)
        }
    
    def _compute_regional_policy_impacts(self,
                                       policy_rate: float,
                                       regional_conditions: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute how policy rate affects each region.
        
        Args:
            policy_rate: Policy interest rate
            regional_conditions: Regional economic conditions
            
        Returns:
            Dictionary with regional impact measures
        """
        # This would involve solving the regional equilibrium system
        # For now, provide simplified impacts based on regional sensitivities
        
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        
        # Direct interest rate effects on output gaps
        output_gap_impacts = -sigma * policy_rate
        
        # Indirect effects on inflation through Phillips curve
        inflation_impacts = -kappa * output_gap_impacts
        
        return {
            'output_gap_impacts': output_gap_impacts,
            'inflation_impacts': inflation_impacts,
            'welfare_impacts': (
                self.welfare_function.output_gap_weight * output_gap_impacts**2 +
                self.welfare_function.inflation_weight * inflation_impacts**2
            )
        }
    
    def _extract_regional_output_gaps(self, data: pd.DataFrame) -> np.ndarray:
        """Extract regional output gaps from data."""
        output_cols = [col for col in data.columns if 'output_gap' in col.lower()]
        if not output_cols:
            raise ValueError("No output gap columns found in data")
        
        return data[output_cols].iloc[-1].values
    
    def _extract_regional_inflation(self, data: pd.DataFrame) -> np.ndarray:
        """Extract regional inflation from data."""
        inflation_cols = [col for col in data.columns if 'inflation' in col.lower()]
        if not inflation_cols:
            raise ValueError("No inflation columns found in data")
        
        return data[inflation_cols].iloc[-1].values
    
    def generate_policy_scenario(self,
                               regional_data: pd.DataFrame,
                               scenario_name: str = "optimal_policy") -> PolicyScenario:
        """
        Generate a complete policy scenario with optimal rates.
        
        Args:
            regional_data: Time series of regional economic data
            scenario_name: Name for the policy scenario
            
        Returns:
            PolicyScenario with optimal policy path and outcomes
        """
        # Compute optimal policy path
        optimal_rates = self.compute_optimal_rate_path(regional_data)
        
        # Simulate regional outcomes under optimal policy
        # This would involve solving the full regional equilibrium system
        # For now, use simplified approach
        regional_outcomes = self._simulate_regional_outcomes(optimal_rates, regional_data)
        
        # Compute welfare outcome
        welfare_outcome = self.evaluate_welfare_loss(optimal_rates, regional_outcomes)
        
        return PolicyScenario(
            name=scenario_name,
            policy_rates=optimal_rates,
            regional_outcomes=regional_outcomes,
            welfare_outcome=welfare_outcome,
            scenario_type='optimal_regional',
            policy_parameters=self.optimal_coefficients,
            regional_weights=self.optimal_regional_weights,
            information_set={'type': 'perfect_information'}
        )
    
    def _simulate_regional_outcomes(self,
                                  policy_rates: pd.Series,
                                  regional_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate regional outcomes under given policy path.
        
        This is a simplified simulation - full implementation would solve
        the complete regional equilibrium system.
        """
        n_regions = self.regional_params.n_regions
        n_periods = len(policy_rates)
        
        # Create outcome matrix (regions x time)
        outcomes = np.zeros((2 * n_regions, n_periods))  # Output gaps + inflation
        
        # Simple simulation based on regional sensitivities
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        
        for t, (date, rate) in enumerate(policy_rates.items()):
            # Direct effects of policy rate
            output_gap_effects = -sigma * rate
            inflation_effects = -kappa * output_gap_effects
            
            # Store outcomes
            outcomes[:n_regions, t] = output_gap_effects
            outcomes[n_regions:, t] = inflation_effects
        
        # Create DataFrame with proper indexing
        region_names = [f"output_gap_region_{i+1}" for i in range(n_regions)]
        region_names += [f"inflation_region_{i+1}" for i in range(n_regions)]
        
        return pd.DataFrame(outcomes, 
                          index=region_names, 
                          columns=policy_rates.index)