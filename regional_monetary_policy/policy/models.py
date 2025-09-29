"""
Core policy analysis models for regional monetary policy.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PolicyScenario:
    """
    Container for counterfactual policy scenario analysis.
    
    Represents a complete policy scenario including policy rates,
    resulting regional outcomes, and welfare calculations.
    """
    
    name: str  # Descriptive name for the scenario
    policy_rates: pd.Series  # Time series of policy interest rates
    regional_outcomes: pd.DataFrame  # Regional output gaps and inflation outcomes
    welfare_outcome: float  # Aggregate welfare measure for this scenario
    scenario_type: str  # Type: 'baseline', 'perfect_info', 'optimal_regional', 'perfect_regional'
    
    # Additional scenario metadata
    policy_parameters: Dict[str, float] = None
    regional_weights: np.ndarray = None
    information_set: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate scenario data consistency."""
        self._validate_scenario()
    
    def _validate_scenario(self):
        """Ensure scenario data is consistent and complete."""
        valid_types = ['baseline', 'perfect_info', 'optimal_regional', 'perfect_regional', 'test', 'alternative']
        if self.scenario_type not in valid_types:
            raise ValueError(f"Scenario type must be one of {valid_types}")
        
        # Only validate time indices if both have data
        if len(self.policy_rates) > 0 and len(self.regional_outcomes.columns) > 0:
            if not self.policy_rates.index.equals(self.regional_outcomes.columns):
                raise ValueError("Policy rates and regional outcomes must have matching time indices")
    
    def compute_policy_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics for the policy path.
        
        Returns:
            Dictionary with policy rate statistics
        """
        return {
            'mean_rate': self.policy_rates.mean(),
            'std_rate': self.policy_rates.std(),
            'min_rate': self.policy_rates.min(),
            'max_rate': self.policy_rates.max(),
            'rate_volatility': self.policy_rates.diff().std(),
            'rate_persistence': self.policy_rates.autocorr(lag=1)
        }
    
    def get_regional_impacts(self) -> pd.DataFrame:
        """
        Compute regional-specific impact measures.
        
        Returns:
            DataFrame with regional welfare and volatility measures
        """
        n_regions = len(self.regional_outcomes.index) // 2  # Assuming output gap + inflation per region
        
        impacts = []
        for i in range(n_regions):
            region_name = f"Region_{i+1}"
            
            # Extract regional output gap and inflation
            output_gap = self.regional_outcomes.iloc[i]
            inflation = self.regional_outcomes.iloc[i + n_regions]
            
            # Compute regional welfare loss (quadratic loss function)
            welfare_loss = np.mean(output_gap**2 + inflation**2)
            
            # Compute volatilities
            output_volatility = output_gap.std()
            inflation_volatility = inflation.std()
            
            impacts.append({
                'region': region_name,
                'welfare_loss': welfare_loss,
                'output_volatility': output_volatility,
                'inflation_volatility': inflation_volatility,
                'mean_output_gap': output_gap.mean(),
                'mean_inflation': inflation.mean()
            })
        
        return pd.DataFrame(impacts)
    
    def compare_to_baseline(self, baseline_scenario: 'PolicyScenario') -> Dict[str, float]:
        """
        Compare this scenario to a baseline scenario.
        
        Args:
            baseline_scenario: Reference scenario for comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        welfare_improvement = self.welfare_outcome - baseline_scenario.welfare_outcome
        
        # Compare policy rate characteristics
        rate_diff = self.policy_rates - baseline_scenario.policy_rates
        
        return {
            'welfare_improvement': welfare_improvement,
            'welfare_improvement_pct': (welfare_improvement / abs(baseline_scenario.welfare_outcome)) * 100,
            'mean_rate_difference': rate_diff.mean(),
            'rate_difference_volatility': rate_diff.std(),
            'max_rate_difference': rate_diff.abs().max()
        }
    
    @property
    def time_periods(self) -> pd.Index:
        """Get time periods covered by the scenario."""
        return self.policy_rates.index
    
    @property
    def n_periods(self) -> int:
        """Get number of time periods in the scenario."""
        return len(self.policy_rates)


@dataclass
class PolicyMistakeComponents:
    """
    Decomposition of monetary policy mistakes according to Theorem 4.
    
    Breaks down the total policy mistake into its constituent components:
    information effects, weight misallocation, parameter misspecification,
    and inflation response effects.
    """
    
    total_mistake: float  # Total policy rate deviation from optimal
    information_effect: float  # Effect of imperfect information
    weight_misallocation_effect: float  # Effect of suboptimal regional weights
    parameter_misspecification_effect: float  # Effect of parameter errors
    inflation_response_effect: float  # Effect of inflation response coefficient errors
    
    # Additional decomposition details
    measurement_errors: Dict[str, float] = None
    weight_differences: np.ndarray = None
    parameter_differences: Dict[str, float] = None
    
    def __post_init__(self):
        """Validate decomposition consistency."""
        self._validate_decomposition()
    
    def _validate_decomposition(self):
        """Ensure decomposition components sum to total mistake."""
        component_sum = (
            self.information_effect + 
            self.weight_misallocation_effect + 
            self.parameter_misspecification_effect + 
            self.inflation_response_effect
        )
        
        if abs(component_sum - self.total_mistake) > 1e-6:
            raise ValueError("Decomposition components must sum to total mistake")
    
    def get_relative_contributions(self) -> Dict[str, float]:
        """
        Compute relative contribution of each component to total mistake.
        
        Returns:
            Dictionary with percentage contributions
        """
        if abs(self.total_mistake) < 1e-10:
            return {
                'information': 0.0,
                'weight_misallocation': 0.0,
                'parameter_misspecification': 0.0,
                'inflation_response': 0.0
            }
        
        return {
            'information': (self.information_effect / self.total_mistake) * 100,
            'weight_misallocation': (self.weight_misallocation_effect / self.total_mistake) * 100,
            'parameter_misspecification': (self.parameter_misspecification_effect / self.total_mistake) * 100,
            'inflation_response': (self.inflation_response_effect / self.total_mistake) * 100
        }
    
    def plot_decomposition(self) -> plt.Figure:
        """
        Create a visualization of the policy mistake decomposition.
        
        Returns:
            Matplotlib figure with decomposition chart
        """
        components = [
            'Information\nEffect',
            'Weight\nMisallocation', 
            'Parameter\nMisspecification',
            'Inflation\nResponse'
        ]
        
        values = [
            self.information_effect,
            self.weight_misallocation_effect,
            self.parameter_misspecification_effect,
            self.inflation_response_effect
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of absolute contributions
        bars = ax1.bar(components, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax1.set_title('Policy Mistake Decomposition\n(Absolute Contributions)')
        ax1.set_ylabel('Contribution to Policy Mistake')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + np.sign(height)*0.01,
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Pie chart of relative contributions (absolute values)
        abs_values = [abs(v) for v in values]
        if sum(abs_values) > 0:
            ax2.pie(abs_values, labels=components, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Relative Contributions\n(Absolute Values)')
        else:
            ax2.text(0.5, 0.5, 'No Policy Mistakes', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Relative Contributions')
        
        plt.tight_layout()
        return fig
    
    def summary_report(self) -> str:
        """
        Generate a text summary of the policy mistake decomposition.
        
        Returns:
            Formatted string with decomposition results
        """
        relative_contribs = self.get_relative_contributions()
        
        return f"""
Policy Mistake Decomposition Report
==================================

Total Policy Mistake: {self.total_mistake:.4f} percentage points

Component Breakdown:
  Information Effect:           {self.information_effect:.4f} ({relative_contribs['information']:+.1f}%)
  Weight Misallocation:         {self.weight_misallocation_effect:.4f} ({relative_contribs['weight_misallocation']:+.1f}%)
  Parameter Misspecification:   {self.parameter_misspecification_effect:.4f} ({relative_contribs['parameter_misspecification']:+.1f}%)
  Inflation Response Effect:    {self.inflation_response_effect:.4f} ({relative_contribs['inflation_response']:+.1f}%)

Interpretation:
- Positive values indicate the component contributed to overly tight policy
- Negative values indicate the component contributed to overly loose policy
- The sum of all components equals the total policy mistake
        """.strip()


@dataclass
class WelfareDecomposition:
    """
    Decomposition of welfare differences between policy scenarios.
    """
    
    total_welfare_difference: float
    output_gap_component: float
    inflation_component: float
    regional_distribution_component: float
    
    baseline_welfare: float
    alternative_welfare: float
    
    def get_welfare_improvement_pct(self) -> float:
        """Calculate percentage welfare improvement."""
        if abs(self.baseline_welfare) < 1e-10:
            return 0.0
        return (self.total_welfare_difference / abs(self.baseline_welfare)) * 100
    
    def summary(self) -> str:
        """Generate welfare decomposition summary."""
        improvement_pct = self.get_welfare_improvement_pct()
        
        return f"""
Welfare Decomposition Analysis
=============================

Total Welfare Difference: {self.total_welfare_difference:.6f}
Welfare Improvement: {improvement_pct:+.2f}%

Component Breakdown:
  Output Gap Stabilization:     {self.output_gap_component:.6f}
  Inflation Stabilization:      {self.inflation_component:.6f}
  Regional Distribution:        {self.regional_distribution_component:.6f}

Baseline Welfare:     {self.baseline_welfare:.6f}
Alternative Welfare:  {self.alternative_welfare:.6f}
        """.strip()


@dataclass
class ComparisonResults:
    """
    Results from comparing multiple policy scenarios.
    """
    
    scenario_names: List[str]
    welfare_outcomes: List[float]
    welfare_ranking: List[int]  # Ranking by welfare (1 = best)
    pairwise_comparisons: Dict[str, Dict[str, float]]
    
    def verify_theoretical_ranking(self) -> bool:
        """
        Verify that welfare ranking follows theoretical prediction:
        W^PR ≥ W^PI ≥ W^OR ≥ W^B
        """
        expected_order = ['perfect_regional', 'perfect_info', 'optimal_regional', 'baseline']
        
        # Find scenarios in expected order
        scenario_indices = {}
        for i, name in enumerate(self.scenario_names):
            for expected in expected_order:
                if expected in name.lower():
                    scenario_indices[expected] = i
                    break
        
        # Check if we have all expected scenarios
        if len(scenario_indices) != 4:
            return False
        
        # Verify welfare ordering
        welfare_by_type = {
            scenario_type: self.welfare_outcomes[idx] 
            for scenario_type, idx in scenario_indices.items()
        }
        
        return (
            welfare_by_type['perfect_regional'] >= welfare_by_type['perfect_info'] and
            welfare_by_type['perfect_info'] >= welfare_by_type['optimal_regional'] and
            welfare_by_type['optimal_regional'] >= welfare_by_type['baseline']
        )
    
    def get_best_scenario(self) -> str:
        """Get name of scenario with highest welfare."""
        best_idx = np.argmax(self.welfare_outcomes)
        return self.scenario_names[best_idx]
    
    def summary_table(self) -> pd.DataFrame:
        """Generate summary table of all scenarios."""
        return pd.DataFrame({
            'Scenario': self.scenario_names,
            'Welfare': self.welfare_outcomes,
            'Ranking': self.welfare_ranking
        }).sort_values('Ranking')