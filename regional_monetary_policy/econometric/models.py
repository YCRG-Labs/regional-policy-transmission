"""
Core econometric models for regional monetary policy analysis.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class RegionalParameters:
    """
    Container for estimated regional structural parameters.
    
    Stores the key parameters from the regional monetary policy model:
    - sigma: Interest rate sensitivities (intertemporal substitution)
    - kappa: Phillips curve slopes (price adjustment)
    - psi: Demand spillover parameters (regional interactions)
    - phi: Price spillover parameters (inflation spillovers)
    - beta: Discount factors (time preferences)
    """
    
    sigma: np.ndarray  # Interest rate sensitivities by region
    kappa: np.ndarray  # Phillips curve slopes by region
    psi: np.ndarray    # Demand spillover parameters by region
    phi: np.ndarray    # Price spillover parameters by region
    beta: np.ndarray   # Discount factors by region
    
    standard_errors: Dict[str, np.ndarray]  # Standard errors for each parameter
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # CI bounds
    
    def __post_init__(self):
        """Validate parameter dimensions and consistency."""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Ensure all parameter arrays have consistent dimensions."""
        n_regions = len(self.sigma)
        
        for param_name, param_array in [
            ('kappa', self.kappa), ('psi', self.psi), 
            ('phi', self.phi), ('beta', self.beta)
        ]:
            if len(param_array) != n_regions:
                raise ValueError(f"Parameter {param_name} must have {n_regions} elements")
        
        # Validate standard errors
        for param_name in ['sigma', 'kappa', 'psi', 'phi', 'beta']:
            if param_name in self.standard_errors:
                if len(self.standard_errors[param_name]) != n_regions:
                    raise ValueError(f"Standard errors for {param_name} must have {n_regions} elements")
    
    def get_optimal_weights(self, social_weights: np.ndarray) -> np.ndarray:
        """
        Compute optimal regional weights for monetary policy.
        
        Args:
            social_weights: Social welfare weights for each region
            
        Returns:
            Optimal policy weights based on regional parameters
        """
        # Compute weights based on regional sensitivities and social preferences
        # This implements the welfare-maximizing weight formula from the theoretical model
        sensitivity_weights = self.sigma * self.kappa
        optimal_weights = social_weights * sensitivity_weights
        
        # Normalize weights to sum to 1
        return optimal_weights / np.sum(optimal_weights)
    
    def compute_aggregate_parameters(self, population_shares: np.ndarray) -> Dict[str, float]:
        """
        Compute population-weighted aggregate parameters.
        
        Args:
            population_shares: Population share of each region
            
        Returns:
            Dictionary of aggregate parameter values
        """
        if len(population_shares) != len(self.sigma):
            raise ValueError("Population shares must match number of regions")
        
        if not np.isclose(np.sum(population_shares), 1.0):
            raise ValueError("Population shares must sum to 1")
        
        return {
            'sigma_aggregate': np.sum(self.sigma * population_shares),
            'kappa_aggregate': np.sum(self.kappa * population_shares),
            'psi_aggregate': np.sum(self.psi * population_shares),
            'phi_aggregate': np.sum(self.phi * population_shares),
            'beta_aggregate': np.sum(self.beta * population_shares)
        }
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for all parameters.
        
        Returns:
            DataFrame with parameter estimates, standard errors, and confidence intervals
        """
        n_regions = len(self.sigma)
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        
        summary_data = {
            'sigma': self.sigma,
            'kappa': self.kappa,
            'psi': self.psi,
            'phi': self.phi,
            'beta': self.beta
        }
        
        # Add standard errors if available
        for param in ['sigma', 'kappa', 'psi', 'phi', 'beta']:
            if param in self.standard_errors:
                summary_data[f'{param}_se'] = self.standard_errors[param]
        
        # Add confidence intervals if available
        for param in ['sigma', 'kappa', 'psi', 'phi', 'beta']:
            if param in self.confidence_intervals:
                lower, upper = self.confidence_intervals[param]
                summary_data[f'{param}_ci_lower'] = lower
                summary_data[f'{param}_ci_upper'] = upper
        
        return pd.DataFrame(summary_data, index=regions)
    
    @property
    def n_regions(self) -> int:
        """Get number of regions."""
        return len(self.sigma)


@dataclass
class EstimationConfig:
    """
    Configuration parameters for econometric estimation procedures.
    
    This class stores all options and settings needed for the three-stage
    estimation procedure and robustness checks.
    """
    
    # GMM estimation options
    gmm_options: Dict[str, Any]
    
    # Identification strategy
    identification_strategy: str  # 'baseline', 'alternative', 'robust'
    
    # Spatial weight construction method
    spatial_weight_method: str  # 'trade_migration', 'distance_only', 'financial'
    
    # Robustness checks to perform
    robustness_checks: List[str]
    
    # Numerical optimization settings
    convergence_tolerance: float
    max_iterations: int
    
    # Bootstrap settings for standard errors
    bootstrap_replications: int = 1000
    bootstrap_method: str = 'block'  # 'block', 'stationary', 'circular'
    
    # Spatial weight parameters
    spatial_weight_params: Dict[str, float] = None
    
    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.spatial_weight_params is None:
            self.spatial_weight_params = {
                'trade_weight': 0.4,
                'migration_weight': 0.3,
                'financial_weight': 0.2,
                'distance_weight': 0.1
            }
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_strategies = ['baseline', 'alternative', 'robust']
        if self.identification_strategy not in valid_strategies:
            raise ValueError(f"Identification strategy must be one of {valid_strategies}")
        
        valid_spatial_methods = ['trade_migration', 'distance_only', 'financial']
        if self.spatial_weight_method not in valid_spatial_methods:
            raise ValueError(f"Spatial weight method must be one of {valid_spatial_methods}")
        
        if self.convergence_tolerance <= 0:
            raise ValueError("Convergence tolerance must be positive")
        
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        
        # Validate spatial weight parameters sum to 1
        if abs(sum(self.spatial_weight_params.values()) - 1.0) > 1e-6:
            raise ValueError("Spatial weight parameters must sum to 1")


@dataclass
class EstimationResults:
    """
    Container for complete estimation results.
    """
    
    regional_parameters: RegionalParameters
    estimation_config: EstimationConfig
    convergence_info: Dict[str, Any]
    identification_tests: Dict[str, float]
    robustness_results: Dict[str, Any]
    estimation_time: float
    
    def summary_report(self) -> str:
        """Generate a comprehensive estimation summary."""
        return f"""
Regional Parameter Estimation Results
===================================

Estimation Method: {self.estimation_config.identification_strategy}
Spatial Weights: {self.estimation_config.spatial_weight_method}
Estimation Time: {self.estimation_time:.2f} seconds

Convergence: {'SUCCESS' if self.convergence_info.get('converged', False) else 'FAILED'}
Iterations: {self.convergence_info.get('iterations', 'N/A')}

Number of Regions: {self.regional_parameters.n_regions}

Parameter Ranges:
  Sigma (interest sensitivity): [{np.min(self.regional_parameters.sigma):.3f}, {np.max(self.regional_parameters.sigma):.3f}]
  Kappa (Phillips curve): [{np.min(self.regional_parameters.kappa):.3f}, {np.max(self.regional_parameters.kappa):.3f}]
  Psi (demand spillover): [{np.min(self.regional_parameters.psi):.3f}, {np.max(self.regional_parameters.psi):.3f}]
  Phi (price spillover): [{np.min(self.regional_parameters.phi):.3f}, {np.max(self.regional_parameters.phi):.3f}]

Identification Tests:
{self._format_test_results()}
        """.strip()
    
    def _format_test_results(self) -> str:
        """Format identification test results."""
        if not self.identification_tests:
            return "  No tests performed"
        
        return "\n".join(f"  {test}: {value:.4f}" for test, value in self.identification_tests.items())


@dataclass
class IdentificationReport:
    """
    Results from parameter identification tests.
    """
    
    is_identified: bool
    weak_identification_warning: bool
    test_statistics: Dict[str, float]
    critical_values: Dict[str, float]
    recommendations: List[str]
    
    def summary(self) -> str:
        """Generate identification test summary."""
        status = "IDENTIFIED" if self.is_identified else "NOT IDENTIFIED"
        weak_warning = " (WEAK IDENTIFICATION WARNING)" if self.weak_identification_warning else ""
        
        return f"""
Parameter Identification Report
==============================
Status: {status}{weak_warning}

Test Statistics:
{self._format_tests()}

Recommendations:
{self._format_recommendations()}
        """.strip()
    
    def _format_tests(self) -> str:
        """Format test statistics with critical values."""
        if not self.test_statistics:
            return "  No tests performed"
        
        lines = []
        for test, stat in self.test_statistics.items():
            critical = self.critical_values.get(test, "N/A")
            lines.append(f"  {test}: {stat:.4f} (critical: {critical})")
        
        return "\n".join(lines)
    
    def _format_recommendations(self) -> str:
        """Format recommendations list."""
        if not self.recommendations:
            return "  None"
        
        return "\n".join(f"  - {rec}" for rec in self.recommendations)