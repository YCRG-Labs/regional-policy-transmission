#!/usr/bin/env python3
"""
Demonstration of the visualization and reporting system for regional monetary policy analysis.

This script shows how to use the visualization components and report generator
to create comprehensive analysis reports with interactive charts, maps, and tables.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.models import RegionalParameters, EstimationResults, EstimationConfig
from regional_monetary_policy.policy.models import (
    PolicyScenario, PolicyMistakeComponents, ComparisonResults, WelfareDecomposition
)
from regional_monetary_policy.presentation.visualizers import (
    RegionalMapVisualizer, TimeSeriesVisualizer, ParameterVisualizer,
    PolicyAnalysisVisualizer, CounterfactualVisualizer
)
from regional_monetary_policy.presentation.report_generator import ReportGenerator


def create_sample_regional_data(n_regions: int = 8, n_periods: int = 120) -> RegionalDataset:
    """Create sample regional economic data for demonstration."""
    print("Creating sample regional economic data...")
    
    # Create time index (10 years of monthly data)
    start_date = datetime(2014, 1, 1)
    dates = pd.date_range(start_date, periods=n_periods, freq='M')
    
    # Create region codes (using state-like codes)
    region_codes = [f"R{i+1:02d}" for i in range(n_regions)]
    
    # Generate synthetic economic data with regional heterogeneity
    np.random.seed(42)
    
    # Output gaps with different regional sensitivities
    output_gaps = np.zeros((n_regions, n_periods))
    inflation_rates = np.zeros((n_regions, n_periods))
    
    # Common monetary policy shock
    policy_shock = np.random.normal(0, 0.5, n_periods)
    policy_rates = 2.0 + np.cumsum(policy_shock * 0.1)
    policy_rates = np.clip(policy_rates, 0.1, 8.0)  # Realistic bounds
    
    for i in range(n_regions):
        # Regional heterogeneity parameters
        sensitivity = 0.5 + 0.5 * np.random.random()  # Interest rate sensitivity
        persistence = 0.7 + 0.2 * np.random.random()  # Output gap persistence
        
        # Generate output gaps
        for t in range(n_periods):
            if t == 0:
                output_gaps[i, t] = np.random.normal(0, 0.5)
            else:
                # AR(1) process with monetary policy effects
                output_gaps[i, t] = (persistence * output_gaps[i, t-1] - 
                                   sensitivity * (policy_rates[t] - 2.0) + 
                                   np.random.normal(0, 0.3))
        
        # Generate inflation rates (Phillips curve relationship)
        phillips_slope = 0.1 + 0.1 * np.random.random()
        for t in range(n_periods):
            inflation_base = 2.0  # Target inflation
            if t == 0:
                inflation_rates[i, t] = inflation_base + np.random.normal(0, 0.2)
            else:
                inflation_rates[i, t] = (0.8 * inflation_rates[i, t-1] + 
                                       phillips_slope * output_gaps[i, t] + 
                                       np.random.normal(0, 0.2))
    
    # Create DataFrames
    output_gaps_df = pd.DataFrame(output_gaps, index=region_codes, columns=dates)
    inflation_rates_df = pd.DataFrame(inflation_rates, index=region_codes, columns=dates)
    policy_rates_series = pd.Series(policy_rates, index=dates)
    
    # Create real-time estimates (simplified)
    real_time_estimates = {
        'output_gap_rt': output_gaps_df + np.random.normal(0, 0.1, output_gaps_df.shape),
        'inflation_rt': inflation_rates_df + np.random.normal(0, 0.05, inflation_rates_df.shape)
    }
    
    metadata = {
        'source': 'Synthetic data for demonstration',
        'created_at': datetime.now().isoformat(),
        'n_regions': n_regions,
        'n_periods': n_periods
    }
    
    return RegionalDataset(
        output_gaps=output_gaps_df,
        inflation_rates=inflation_rates_df,
        interest_rates=policy_rates_series,
        real_time_estimates=real_time_estimates,
        metadata=metadata
    )


def create_sample_parameters(n_regions: int = 8) -> RegionalParameters:
    """Create sample regional parameters for demonstration."""
    print("Creating sample regional parameters...")
    
    np.random.seed(123)
    
    # Generate heterogeneous parameters
    sigma = 0.5 + 0.5 * np.random.random(n_regions)  # Interest rate sensitivity
    kappa = 0.05 + 0.1 * np.random.random(n_regions)  # Phillips curve slope
    psi = 0.1 + 0.2 * np.random.random(n_regions)  # Demand spillover
    phi = 0.05 + 0.1 * np.random.random(n_regions)  # Price spillover
    beta = 0.95 + 0.04 * np.random.random(n_regions)  # Discount factor
    
    # Generate standard errors (10% of parameter values)
    standard_errors = {
        'sigma': sigma * 0.1,
        'kappa': kappa * 0.1,
        'psi': psi * 0.1,
        'phi': phi * 0.1,
        'beta': beta * 0.01
    }
    
    # Generate confidence intervals
    confidence_intervals = {}
    for param_name, param_values in [('sigma', sigma), ('kappa', kappa), 
                                   ('psi', psi), ('phi', phi), ('beta', beta)]:
        se = standard_errors[param_name]
        lower = param_values - 1.96 * se
        upper = param_values + 1.96 * se
        confidence_intervals[param_name] = (lower, upper)
    
    return RegionalParameters(
        sigma=sigma,
        kappa=kappa,
        psi=psi,
        phi=phi,
        beta=beta,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals
    )


def create_sample_policy_mistakes() -> PolicyMistakeComponents:
    """Create sample policy mistake decomposition for demonstration."""
    print("Creating sample policy mistake components...")
    
    # Sample decomposition values (in percentage points)
    total_mistake = 0.25
    information_effect = 0.10
    weight_misallocation_effect = 0.08
    parameter_misspecification_effect = 0.05
    inflation_response_effect = 0.02
    
    return PolicyMistakeComponents(
        total_mistake=total_mistake,
        information_effect=information_effect,
        weight_misallocation_effect=weight_misallocation_effect,
        parameter_misspecification_effect=parameter_misspecification_effect,
        inflation_response_effect=inflation_response_effect
    )


def create_sample_scenarios(regional_data: RegionalDataset) -> List[PolicyScenario]:
    """Create sample counterfactual scenarios for demonstration."""
    print("Creating sample counterfactual scenarios...")
    
    scenarios = []
    n_regions = regional_data.n_regions
    n_periods = regional_data.n_periods
    
    # Baseline scenario (actual policy)
    baseline_rates = regional_data.interest_rates
    baseline_outcomes = pd.concat([regional_data.output_gaps, regional_data.inflation_rates])
    baseline_welfare = -0.0150  # Negative welfare (loss function)
    
    scenarios.append(PolicyScenario(
        name="Baseline",
        policy_rates=baseline_rates,
        regional_outcomes=baseline_outcomes,
        welfare_outcome=baseline_welfare,
        scenario_type="baseline"
    ))
    
    # Perfect Information scenario
    pi_rates = baseline_rates * 0.95  # Slightly more responsive policy
    pi_outcomes = baseline_outcomes * 0.9  # Better outcomes
    pi_welfare = -0.0120
    
    scenarios.append(PolicyScenario(
        name="Perfect Information",
        policy_rates=pi_rates,
        regional_outcomes=pi_outcomes,
        welfare_outcome=pi_welfare,
        scenario_type="perfect_info"
    ))
    
    # Optimal Regional scenario
    or_rates = baseline_rates * 0.92
    or_outcomes = baseline_outcomes * 0.85
    or_welfare = -0.0100
    
    scenarios.append(PolicyScenario(
        name="Optimal Regional",
        policy_rates=or_rates,
        regional_outcomes=or_outcomes,
        welfare_outcome=or_welfare,
        scenario_type="optimal_regional"
    ))
    
    # Perfect Regional scenario
    pr_rates = baseline_rates * 0.90
    pr_outcomes = baseline_outcomes * 0.80
    pr_welfare = -0.0080
    
    scenarios.append(PolicyScenario(
        name="Perfect Regional",
        policy_rates=pr_rates,
        regional_outcomes=pr_outcomes,
        welfare_outcome=pr_welfare,
        scenario_type="perfect_regional"
    ))
    
    return scenarios


def demonstrate_visualizations():
    """Demonstrate all visualization components."""
    print("\n" + "="*60)
    print("REGIONAL MONETARY POLICY VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    regional_data = create_sample_regional_data(n_regions=8, n_periods=120)
    regional_params = create_sample_parameters(n_regions=8)
    policy_mistakes = create_sample_policy_mistakes()
    scenarios = create_sample_scenarios(regional_data)
    
    # Create comparison results
    comparison_results = ComparisonResults(
        scenario_names=[s.name for s in scenarios],
        welfare_outcomes=[s.welfare_outcome for s in scenarios],
        welfare_ranking=[4, 3, 2, 1],  # Perfect Regional is best
        pairwise_comparisons={}
    )
    
    print(f"\nGenerated sample data:")
    print(f"- {regional_data.n_regions} regions")
    print(f"- {regional_data.n_periods} time periods")
    print(f"- {len(scenarios)} counterfactual scenarios")
    
    # Initialize visualizers
    print("\nInitializing visualizers...")
    map_viz = RegionalMapVisualizer(
        region_codes=regional_data.regions,
        region_names=[f"Region {i+1}" for i in range(regional_data.n_regions)]
    )
    ts_viz = TimeSeriesVisualizer()
    param_viz = ParameterVisualizer()
    policy_viz = PolicyAnalysisVisualizer()
    counterfactual_viz = CounterfactualVisualizer()
    
    # Create output directory
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations (saved to {output_dir}/)...")
    
    # 1. Regional maps for economic indicators
    print("1. Creating regional economic indicator maps...")
    
    # Output gap map
    avg_output_gaps = regional_data.output_gaps.mean(axis=1)
    output_gap_map = map_viz.create_indicator_map(
        avg_output_gaps,
        "Average Output Gaps by Region",
        colorscale='RdBu_r'
    )
    output_gap_map.write_html(f"{output_dir}/output_gap_map.html")
    
    # Inflation map
    avg_inflation = regional_data.inflation_rates.mean(axis=1)
    inflation_map = map_viz.create_indicator_map(
        avg_inflation,
        "Average Inflation Rates by Region",
        colorscale='Reds'
    )
    inflation_map.write_html(f"{output_dir}/inflation_map.html")
    
    # Multi-indicator dashboard
    indicators = {
        'Output Gap': avg_output_gaps,
        'Inflation Rate': avg_inflation,
        'Output Volatility': regional_data.output_gaps.std(axis=1),
        'Inflation Volatility': regional_data.inflation_rates.std(axis=1)
    }
    dashboard = map_viz.create_multi_indicator_map(
        indicators,
        "Regional Economic Indicators Dashboard"
    )
    dashboard.write_html(f"{output_dir}/economic_dashboard.html")
    
    # 2. Time series plots
    print("2. Creating time series visualizations...")
    
    # Policy transmission plot
    transmission_plot = ts_viz.create_policy_transmission_plot(
        regional_data.interest_rates,
        pd.concat([regional_data.output_gaps, regional_data.inflation_rates]),
        "Monetary Policy Transmission Analysis"
    )
    transmission_plot.write_html(f"{output_dir}/policy_transmission.html")
    
    # Spillover effects (create sample spatial weights)
    spatial_weights = np.random.random((regional_data.n_regions, regional_data.n_regions))
    np.fill_diagonal(spatial_weights, 0)
    spatial_weights = spatial_weights / spatial_weights.sum(axis=1, keepdims=True)
    
    spillover_plot = ts_viz.create_spillover_effects_plot(
        regional_data,
        spatial_weights,
        "Regional Spillover Effects Analysis"
    )
    spillover_plot.write_html(f"{output_dir}/spillover_effects.html")
    
    # 3. Parameter estimation visualizations
    print("3. Creating parameter estimation visualizations...")
    
    # Parameter estimates plot
    param_plot = param_viz.create_parameter_estimates_plot(
        regional_params,
        "Regional Parameter Estimates with Confidence Intervals"
    )
    param_plot.write_html(f"{output_dir}/parameter_estimates.html")
    
    # Parameter summary table
    param_table = param_viz.create_parameter_comparison_table(
        regional_params,
        "Parameter Estimates Summary Table"
    )
    param_table.write_html(f"{output_dir}/parameter_table.html")
    
    # 4. Policy analysis visualizations
    print("4. Creating policy analysis visualizations...")
    
    # Policy mistake decomposition
    mistake_plot = policy_viz.create_mistake_decomposition_plot(
        policy_mistakes,
        "Policy Mistake Decomposition Analysis"
    )
    mistake_plot.write_html(f"{output_dir}/mistake_decomposition.html")
    
    # Create sample time series of mistakes
    dates = regional_data.time_periods[-24:]  # Last 2 years
    mistake_history = pd.DataFrame({
        'total_mistake': np.random.normal(0.15, 0.05, len(dates)),
        'information_effect': np.random.normal(0.06, 0.02, len(dates)),
        'weight_misallocation_effect': np.random.normal(0.04, 0.015, len(dates)),
        'parameter_misspecification_effect': np.random.normal(0.03, 0.01, len(dates)),
        'inflation_response_effect': np.random.normal(0.02, 0.005, len(dates))
    }, index=dates)
    
    mistake_ts_plot = policy_viz.create_mistake_time_series(
        mistake_history,
        "Policy Mistakes Over Time"
    )
    mistake_ts_plot.write_html(f"{output_dir}/mistake_time_series.html")
    
    # 5. Counterfactual analysis visualizations
    print("5. Creating counterfactual analysis visualizations...")
    
    # Scenario comparison
    scenario_plot = counterfactual_viz.create_scenario_comparison_plot(
        scenarios,
        "Counterfactual Policy Scenarios Comparison"
    )
    scenario_plot.write_html(f"{output_dir}/scenario_comparison.html")
    
    # Welfare comparison table
    welfare_table = counterfactual_viz.create_welfare_comparison_table(
        comparison_results,
        "Welfare Outcomes Comparison"
    )
    welfare_table.write_html(f"{output_dir}/welfare_comparison.html")
    
    # Regional impact heatmap
    regional_heatmap = counterfactual_viz.create_regional_impact_heatmap(
        scenarios,
        "Regional Welfare Impact Comparison"
    )
    regional_heatmap.write_html(f"{output_dir}/regional_impacts.html")
    
    print("\n" + "="*60)
    print("VISUALIZATION DEMONSTRATION COMPLETE")
    print("="*60)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        if file.endswith('.html'):
            print(f"  - {file}")


def demonstrate_report_generation():
    """Demonstrate comprehensive report generation."""
    print("\n" + "="*60)
    print("REPORT GENERATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    regional_data = create_sample_regional_data(n_regions=6, n_periods=100)
    regional_params = create_sample_parameters(n_regions=6)
    policy_mistakes = create_sample_policy_mistakes()
    scenarios = create_sample_scenarios(regional_data)
    
    # Create estimation results
    estimation_config = EstimationConfig(
        gmm_options={'max_iter': 1000, 'tol': 1e-6},
        identification_strategy='baseline',
        spatial_weight_method='trade_migration',
        robustness_checks=['bootstrap', 'alternative_iv'],
        convergence_tolerance=1e-6,
        max_iterations=1000
    )
    
    estimation_results = EstimationResults(
        regional_parameters=regional_params,
        estimation_config=estimation_config,
        convergence_info={'converged': True, 'iterations': 45, 'final_criterion': 1e-7},
        identification_tests={'weak_iv_test': 15.2, 'overid_test': 8.3},
        robustness_results={'bootstrap_se': 'computed', 'alternative_iv': 'similar'},
        estimation_time=12.5
    )
    
    # Create comparison results
    comparison_results = ComparisonResults(
        scenario_names=[s.name for s in scenarios],
        welfare_outcomes=[s.welfare_outcome for s in scenarios],
        welfare_ranking=[4, 3, 2, 1],
        pairwise_comparisons={}
    )
    
    # Initialize report generator
    report_generator = ReportGenerator(output_dir="reports_output")
    
    print("\nGenerating comprehensive analysis report...")
    
    # Generate comprehensive report
    comprehensive_report = report_generator.generate_comprehensive_report(
        estimation_results=estimation_results,
        policy_analysis=policy_mistakes,
        counterfactual_scenarios=scenarios,
        comparison_results=comparison_results,
        regional_data=regional_data,
        report_name="demo_comprehensive_analysis"
    )
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"\nComprehensive report generated with components:")
    
    def print_report_structure(report_dict, indent=0):
        """Recursively print report structure."""
        for key, value in report_dict.items():
            if isinstance(value, dict):
                print("  " * indent + f"- {key}:")
                print_report_structure(value, indent + 1)
            elif isinstance(value, str) and (value.endswith('.html') or value.endswith('.txt') or value.endswith('.csv') or value.endswith('.json')):
                print("  " * indent + f"- {key}: {os.path.basename(value)}")
            else:
                print("  " * indent + f"- {key}: {value}")
    
    print_report_structure(comprehensive_report)
    
    print(f"\nAll report files saved to: reports_output/")


if __name__ == "__main__":
    print("Regional Monetary Policy Analysis - Visualization and Reporting Demo")
    print("This demonstration shows the complete visualization and reporting system.")
    
    try:
        # Run visualization demonstration
        demonstrate_visualizations()
        
        # Run report generation demonstration
        demonstrate_report_generation()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nTo view the generated visualizations:")
        print("1. Open the HTML files in visualization_output/ with a web browser")
        print("2. Review the comprehensive reports in reports_output/")
        print("\nThe visualization system provides:")
        print("- Interactive regional maps for economic indicators")
        print("- Time series plots for policy transmission analysis")
        print("- Parameter estimation results with confidence intervals")
        print("- Policy mistake decomposition visualizations")
        print("- Counterfactual scenario comparison charts")
        print("- Comprehensive analysis reports combining all components")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()