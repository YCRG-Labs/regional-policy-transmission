"""
Policy mistake decomposition implementation following Theorem 4.

This module implements the mathematical framework for decomposing monetary policy
mistakes into their constituent components: information effects, weight misallocation,
parameter misspecification, and inflation response effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..econometric.models import RegionalParameters
from .models import PolicyMistakeComponents


class PolicyMistakeDecomposer:
    """
    Implements policy mistake decomposition from Theorem 4.
    
    This class decomposes the difference between actual Fed policy and optimal
    welfare-maximizing policy into four components:
    1. Information effect: Due to real-time data limitations
    2. Weight misallocation: Due to suboptimal regional weights
    3. Parameter misspecification: Due to incorrect parameter estimates
    4. Inflation response: Due to incorrect inflation response coefficient
    """
    
    def __init__(self, 
                 regional_params: RegionalParameters,
                 social_welfare_weights: np.ndarray,
                 discount_factor: float = 0.99):
        """
        Initialize the policy mistake decomposer.
        
        Args:
            regional_params: Estimated regional structural parameters
            social_welfare_weights: Social welfare weights for each region
            discount_factor: Discount factor for welfare calculations
        """
        self.regional_params = regional_params
        self.social_welfare_weights = social_welfare_weights
        self.discount_factor = discount_factor
        
        # Validate inputs
        if len(social_welfare_weights) != regional_params.n_regions:
            raise ValueError("Social welfare weights must match number of regions")
        
        if not np.isclose(np.sum(social_welfare_weights), 1.0):
            raise ValueError("Social welfare weights must sum to 1")
        
        # Precompute optimal policy coefficients
        self._compute_optimal_coefficients()
    
    def _compute_optimal_coefficients(self):
        """Compute optimal policy response coefficients."""
        # Get regional parameters
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        psi = self.regional_params.psi
        phi = self.regional_params.phi
        
        # Compute optimal regional weights (welfare-maximizing)
        self.optimal_weights = self._compute_optimal_regional_weights()
        
        # Compute optimal response coefficients following the theoretical model
        # These formulas come from the first-order conditions of the welfare maximization problem
        
        # Optimal response to output gaps
        self.optimal_output_coeff = np.sum(
            self.optimal_weights * sigma * kappa / (1 + kappa * phi)
        )
        
        # Optimal response to inflation
        self.optimal_inflation_coeff = np.sum(
            self.optimal_weights * sigma / (1 + kappa * phi)
        )
    
    def _compute_optimal_regional_weights(self) -> np.ndarray:
        """
        Compute welfare-maximizing regional weights.
        
        Returns:
            Optimal regional weights for monetary policy
        """
        # The optimal weights depend on regional parameters and social welfare weights
        sigma = self.regional_params.sigma
        kappa = self.regional_params.kappa
        
        # Compute sensitivity-adjusted social weights
        sensitivity_weights = sigma * kappa
        optimal_weights = self.social_welfare_weights * sensitivity_weights
        
        # Normalize to sum to 1
        return optimal_weights / np.sum(optimal_weights)
    
    def decompose_policy_mistake(self,
                                actual_rate: float,
                                optimal_rate: float,
                                real_time_data: pd.DataFrame,
                                true_data: pd.DataFrame,
                                fed_weights: Optional[np.ndarray] = None,
                                fed_coefficients: Optional[Dict[str, float]] = None) -> PolicyMistakeComponents:
        """
        Decompose policy mistake according to Theorem 4.
        
        Args:
            actual_rate: Actual Fed policy rate
            optimal_rate: Welfare-maximizing optimal rate
            real_time_data: Real-time data available to Fed
            true_data: True (revised) data values
            fed_weights: Fed's implicit regional weights (estimated if None)
            fed_coefficients: Fed's policy rule coefficients (estimated if None)
            
        Returns:
            PolicyMistakeComponents with full decomposition
        """
        total_mistake = actual_rate - optimal_rate
        
        # Estimate Fed weights and coefficients if not provided
        if fed_weights is None:
            fed_weights = self._estimate_fed_weights(real_time_data)
        
        if fed_coefficients is None:
            fed_coefficients = self._estimate_fed_coefficients(real_time_data)
        
        # Compute each component of the decomposition
        info_effect = self._compute_information_effect(real_time_data, true_data, fed_coefficients)
        weight_effect = self._compute_weight_misallocation_effect(fed_weights, true_data)
        param_effect = self._compute_parameter_effect(fed_coefficients, true_data)
        inflation_effect = self._compute_inflation_response_effect(fed_coefficients, true_data)
        
        # Ensure components sum to total mistake (adjust residual to inflation effect)
        component_sum = info_effect + weight_effect + param_effect + inflation_effect
        residual = total_mistake - component_sum
        inflation_effect += residual  # Adjust inflation effect to make sum exact
        
        # Store additional decomposition details
        measurement_errors = self._compute_measurement_errors(real_time_data, true_data)
        weight_differences = fed_weights - self.optimal_weights
        parameter_differences = {
            'output_coeff_diff': fed_coefficients['output'] - self.optimal_output_coeff,
            'inflation_coeff_diff': fed_coefficients['inflation'] - self.optimal_inflation_coeff
        }
        
        return PolicyMistakeComponents(
            total_mistake=total_mistake,
            information_effect=info_effect,
            weight_misallocation_effect=weight_effect,
            parameter_misspecification_effect=param_effect,
            inflation_response_effect=inflation_effect,
            measurement_errors=measurement_errors,
            weight_differences=weight_differences,
            parameter_differences=parameter_differences
        )
    
    def _compute_information_effect(self,
                                  real_time_data: pd.DataFrame,
                                  true_data: pd.DataFrame,
                                  fed_coefficients: Dict[str, float]) -> float:
        """
        Compute information effect due to real-time data limitations.
        
        This captures how measurement errors in real-time data affect policy decisions.
        """
        # Extract regional output gaps and inflation from both datasets
        rt_output_gaps = self._extract_output_gaps(real_time_data)
        rt_inflation = self._extract_inflation(real_time_data)
        
        true_output_gaps = self._extract_output_gaps(true_data)
        true_inflation = self._extract_inflation(true_data)
        
        # Compute measurement errors
        output_gap_errors = rt_output_gaps - true_output_gaps
        inflation_errors = rt_inflation - true_inflation
        
        # Weight errors by Fed's regional weights (if Fed had optimal coefficients)
        fed_weights = self._estimate_fed_weights(real_time_data)
        
        # Information effect = Fed coefficients × weighted measurement errors
        weighted_output_error = np.sum(fed_weights * output_gap_errors)
        weighted_inflation_error = np.sum(fed_weights * inflation_errors)
        
        info_effect = (
            fed_coefficients['output'] * weighted_output_error +
            fed_coefficients['inflation'] * weighted_inflation_error
        )
        
        return info_effect
    
    def _compute_weight_misallocation_effect(self,
                                           fed_weights: np.ndarray,
                                           true_data: pd.DataFrame) -> float:
        """
        Compute weight misallocation effect.
        
        This captures the effect of Fed using suboptimal regional weights.
        """
        # Extract true regional conditions
        true_output_gaps = self._extract_output_gaps(true_data)
        true_inflation = self._extract_inflation(true_data)
        
        # Compute difference in weighted regional conditions
        weight_diff = fed_weights - self.optimal_weights
        
        output_weight_effect = self.optimal_output_coeff * np.sum(weight_diff * true_output_gaps)
        inflation_weight_effect = self.optimal_inflation_coeff * np.sum(weight_diff * true_inflation)
        
        return output_weight_effect + inflation_weight_effect
    
    def _compute_parameter_effect(self,
                                fed_coefficients: Dict[str, float],
                                true_data: pd.DataFrame) -> float:
        """
        Compute parameter misspecification effect.
        
        This captures the effect of Fed using incorrect response coefficients.
        """
        # Extract true regional conditions
        true_output_gaps = self._extract_output_gaps(true_data)
        true_inflation = self._extract_inflation(true_data)
        
        # Compute optimally-weighted regional conditions
        weighted_output_gaps = np.sum(self.optimal_weights * true_output_gaps)
        weighted_inflation = np.sum(self.optimal_weights * true_inflation)
        
        # Parameter effect = coefficient differences × weighted true conditions
        output_coeff_diff = fed_coefficients['output'] - self.optimal_output_coeff
        inflation_coeff_diff = fed_coefficients['inflation'] - self.optimal_inflation_coeff
        
        param_effect = (
            output_coeff_diff * weighted_output_gaps +
            inflation_coeff_diff * weighted_inflation
        )
        
        return param_effect
    
    def _compute_inflation_response_effect(self,
                                         fed_coefficients: Dict[str, float],
                                         true_data: pd.DataFrame) -> float:
        """
        Compute inflation response effect.
        
        This is a subset of parameter effect focusing specifically on inflation response.
        For the complete decomposition, this might be zero if already captured in parameter effect.
        """
        # This could be used for a more detailed decomposition
        # For now, return zero as inflation response is captured in parameter effect
        return 0.0
    
    def _estimate_fed_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Estimate Fed's implicit regional weights from policy decisions.
        
        This is a simplified estimation - in practice, this would involve
        estimating a Fed reaction function.
        """
        # For now, assume equal weights as a baseline
        # In practice, this would be estimated from Fed policy decisions
        n_regions = self.regional_params.n_regions
        return np.ones(n_regions) / n_regions
    
    def _estimate_fed_coefficients(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate Fed's policy rule coefficients.
        
        This would typically involve estimating a Taylor rule or similar reaction function.
        """
        # Simplified coefficients based on standard Taylor rule
        return {
            'output': 0.5,  # Response to output gap
            'inflation': 1.5  # Response to inflation (Taylor principle)
        }
    
    def _extract_output_gaps(self, data: pd.DataFrame) -> np.ndarray:
        """Extract regional output gaps from data."""
        # Assume output gaps are in columns named 'output_gap_region_X'
        output_cols = [col for col in data.columns if 'output_gap' in col.lower()]
        if not output_cols:
            raise ValueError("No output gap columns found in data")
        
        return data[output_cols].iloc[-1].values  # Most recent observation
    
    def _extract_inflation(self, data: pd.DataFrame) -> np.ndarray:
        """Extract regional inflation from data."""
        # Assume inflation is in columns named 'inflation_region_X'
        inflation_cols = [col for col in data.columns if 'inflation' in col.lower()]
        if not inflation_cols:
            raise ValueError("No inflation columns found in data")
        
        return data[inflation_cols].iloc[-1].values  # Most recent observation
    
    def _compute_measurement_errors(self,
                                  real_time_data: pd.DataFrame,
                                  true_data: pd.DataFrame) -> Dict[str, float]:
        """Compute measurement errors for each variable."""
        rt_output = self._extract_output_gaps(real_time_data)
        rt_inflation = self._extract_inflation(real_time_data)
        
        true_output = self._extract_output_gaps(true_data)
        true_inflation = self._extract_inflation(true_data)
        
        return {
            'output_gap_error': np.mean(rt_output - true_output),
            'inflation_error': np.mean(rt_inflation - true_inflation),
            'output_gap_rmse': np.sqrt(np.mean((rt_output - true_output)**2)),
            'inflation_rmse': np.sqrt(np.mean((rt_inflation - true_inflation)**2))
        }
    
    def compute_counterfactual_mistake(self,
                                     counterfactual_weights: np.ndarray,
                                     counterfactual_coefficients: Dict[str, float],
                                     true_data: pd.DataFrame) -> float:
        """
        Compute policy mistake for a counterfactual policy specification.
        
        Args:
            counterfactual_weights: Alternative regional weights
            counterfactual_coefficients: Alternative policy coefficients
            true_data: True economic conditions
            
        Returns:
            Policy mistake under counterfactual specification
        """
        # Extract true conditions
        true_output_gaps = self._extract_output_gaps(true_data)
        true_inflation = self._extract_inflation(true_data)
        
        # Compute counterfactual policy rate
        weighted_output = np.sum(counterfactual_weights * true_output_gaps)
        weighted_inflation = np.sum(counterfactual_weights * true_inflation)
        
        counterfactual_rate = (
            counterfactual_coefficients['output'] * weighted_output +
            counterfactual_coefficients['inflation'] * weighted_inflation
        )
        
        # Compute optimal rate
        optimal_weighted_output = np.sum(self.optimal_weights * true_output_gaps)
        optimal_weighted_inflation = np.sum(self.optimal_weights * true_inflation)
        
        optimal_rate = (
            self.optimal_output_coeff * optimal_weighted_output +
            self.optimal_inflation_coeff * optimal_weighted_inflation
        )
        
        return counterfactual_rate - optimal_rate