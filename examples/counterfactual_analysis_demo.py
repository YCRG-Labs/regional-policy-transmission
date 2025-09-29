"""
Demonstration of counterfactual analysis engine for regional monetary policy.

This script shows how to use the CounterfactualEngine to generate and compare
the four key policy scenarios: Baseline, Perfect Information, Optimal Regional,
and Perfect Regional.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from regional_monetary_policy.policy.counterfactual_engine import (
    CounterfactualEngine, WelfareEvaluator
)
from regional_monetary_policy.policy.optimal_policy import WelfareFunction
from regional_monetary_policy.econometric.models import RegionalParameters
from regional_monetary_policy.data.models import RegionalDataset


def create_synthetic_data(n_regions=4, n_periods=60, seed=42):
    """
    Create synthetic regional economic data for demonstration.
    
    Args:
        n_regions: Number of regions
        n_periods: Number of time periods (months)
        seed: Random seed for reproducibility
        
    Returns:
        RegionalDataset with synthetic data
    """
    np.random.seed(seed)
    
    # Create time index (5 years of monthly data)
    dates = pd.date_range('2018-01-01', periods=n_periods, freq='ME')
    
    # Generate correlated regional shocks
    correlation_matrix = np.eye(n_regions) * 0.7 + np.ones((n_regions, n_regions)) * 0.3
    regional_shocks = np.random.multivariate_normal(
        mean=np.zeros(n_regions),
        cov=correlation_matrix,
        size=n_periods
    ).T
    
    # Generate output gaps with regional heterogeneity
    output_gaps = pd.DataFrame(
        regional_shocks * np.array([1.2, 0.8, 1.0, 0.9])[:, np.newaxis],  # Different volatilities
        index=[f"Region_{i+1}" for i in range(n_regions)],
        columns=dates
    )
    
    # Generate inflation with persistence and regional variation
    inflation_base = 2.0 + 0.5 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)  # Seasonal pattern
    inflation_rates = pd.DataFrame(
        [inflation_base + 0.3 * regional_shocks[i] + np.random.normal(0, 0.1, n_periods) 
         for i in range(n_regions)],
        index=[f"Region_{i+1}" for i in range(n_regions)],
        columns=dates
    )
    
    # Generate Fed policy rates (Taylor rule-like)
    aggregate_output_gap = output_gaps.mean(axis=0)
    aggregate_inflation = inflation_rates.mean(axis=0)
    
    fed_rates = (2.0 +  # Neutral rate
                1.5 * (aggregate_inflation - 2.0) +  # Inflation response
                0.5 * aggregate_output_gap +  # Output gap response
                np.random.normal(0, 0.2, n_periods))  # Policy noise
    
    interest_rates = pd.Series(fed_rates, index=dates, name='fed_funds_rate')
    
    # Create real-time estimates with measurement error
    real_time_estimates = {
        'output_gaps': output_gaps + np.random.normal(0, 0.3, output_gaps.shape),
        'inflation_rates': inflation_rates + np.random.normal(0, 0.15, inflation_rates.shape)
    }
    
    return RegionalDataset(
        output_gaps=output_gaps,
        inflation_rates=inflation_rates,
        interest_rates=interest_rates,
        real_time_estimates=real_time_estimates,
        metadata={
            'source': 'synthetic_demo',
            'n_regions': n_regions,
            'frequency': 'monthly',
            'description': 'Synthetic data for counterfactual analysis demonstration'
        }
    )


def create_regional_parameters(n_regions=4, seed=42):
    """
    Create realistic regional structural parameters.
    
    Args:
        n_regions: Number of regions
        seed: Random seed for reproducibility
        
    Returns:
        RegionalParameters with synthetic estimates
    """
    np.random.seed(seed)
    
    # Interest rate sensitivities (intertemporal substitution elasticity)
    sigma = np.random.uniform(0.3, 0.8, n_regions)
    
    # Phillips curve slopes (price stickiness varies by region)
    kappa = np.random.uniform(0.2, 0.5, n_regions)
    
    # Demand spillover parameters (trade linkages)
    psi = np.random.uniform(0.1, 0.3, n_regions)
    
    # Price spillover parameters (inflation spillovers)
    phi = np.random.uniform(0.05, 0.2, n_regions)
    
    # Discount factors (close to 1)
    beta = np.random.uniform(0.98, 0.995, n_regions)
    
    # Generate standard errors (10-20% of parameter values)
    standard_errors = {
        'sigma': sigma * np.random.uniform(0.1, 0.2, n_regions),
        'kappa': kappa * np.random.uniform(0.1, 0.2, n_regions),
        'psi': psi * np.random.uniform(0.15, 0.25, n_regions),
        'phi': phi * np.random.uniform(0.15, 0.25, n_regions),
        'beta': beta * np.random.uniform(0.01, 0.02, n_regions)
    }
    
    return RegionalParameters(
        sigma=sigma,
        kappa=kappa,
        psi=psi,
        phi=phi,
        beta=beta,
        standard_errors=standard_errors,
        confidence_intervals={}
    )


def create_welfare_function(n_regions=4):
    """
    Create social welfare function with population-based regional weights.
    
    Args:
        n_regions: Number of regions
        
    Returns:
        WelfareFunction specification
    """
    # Population-based weights (larger regions get higher weight)
    population_shares = np.array([0.35, 0.25, 0.25, 0.15])[:n_regions]
    population_shares = population_shares / np.sum(population_shares)  # Normalize
    
    return WelfareFunction(
        output_gap_weight=1.0,
        inflation_weight=1.0,
        regional_weights=population_shares,
        loss_function='quadratic'
    )


def run_counterfactual_analysis():
    """
    Run complete counterfactual analysis demonstration.
    """
    print("Regional Monetary Policy - Counterfactual Analysis Demo")
    print("=" * 60)
    
    # 1. Create synthetic data and parameters
    print("\n1. Creating synthetic regional data and parameters...")
    
    n_regions = 4
    historical_data = create_synthetic_data(n_regions=n_regions, n_periods=60)
    regional_params = create_regional_parameters(n_regions=n_regions)
    welfare_function = create_welfare_function(n_regions=n_regions)
    
    print(f"   - Created data for {n_regions} regions over {historical_data.n_periods} periods")
    print(f"   - Regional parameter ranges:")
    print(f"     * Sigma (interest sensitivity): [{regional_params.sigma.min():.3f}, {regional_params.sigma.max():.3f}]")
    print(f"     * Kappa (Phillips curve): [{regional_params.kappa.min():.3f}, {regional_params.kappa.max():.3f}]")
    print(f"   - Regional welfare weights: {welfare_function.regional_weights}")
    
    # 2. Initialize counterfactual engine
    print("\n2. Initializing counterfactual analysis engine...")
    
    engine = CounterfactualEngine(
        regional_params=regional_params,
        welfare_function=welfare_function,
        discount_factor=0.99,
        n_workers=2  # Use parallel processing
    )
    
    print("   - Engine initialized with optimal policy calculator")
    print(f"   - Using {engine.n_workers} workers for parallel processing")
    
    # 3. Estimate Fed reaction function (simplified)
    print("\n3. Estimating Fed reaction function...")
    
    # Simple OLS estimation of Taylor rule
    y = historical_data.interest_rates.values
    X = np.column_stack([
        historical_data.output_gaps.mean(axis=0).values,  # Aggregate output gap
        historical_data.inflation_rates.mean(axis=0).values - 2.0  # Inflation gap
    ])
    
    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    
    # OLS estimation
    fed_coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    
    fed_reaction_function = {
        'constant': fed_coeffs[0],
        'output_gap_response': fed_coeffs[1],
        'inflation_response': fed_coeffs[2]
    }
    
    print(f"   - Estimated Fed reaction function:")
    print(f"     * Constant: {fed_coeffs[0]:.3f}")
    print(f"     * Output gap response: {fed_coeffs[1]:.3f}")
    print(f"     * Inflation response: {fed_coeffs[2]:.3f}")
    
    # 4. Generate all counterfactual scenarios
    print("\n4. Generating counterfactual scenarios...")
    
    scenarios = engine.generate_all_scenarios(
        historical_data=historical_data,
        fed_policy_rates=historical_data.interest_rates,
        fed_reaction_function=fed_reaction_function,
        parallel=True
    )
    
    print(f"   - Generated {len(scenarios)} scenarios:")
    for scenario in scenarios:
        print(f"     * {scenario.name} (Type: {scenario.scenario_type})")
        print(f"       Welfare outcome: {scenario.welfare_outcome:.6f}")
    
    # 5. Compare scenarios and verify welfare ranking
    print("\n5. Comparing scenarios and verifying welfare ranking...")
    
    comparison = engine.compare_scenarios(scenarios)
    evaluator = WelfareEvaluator(welfare_function)
    
    ranking_correct = evaluator.verify_welfare_ranking(scenarios)
    
    print(f"   - Theoretical welfare ranking verified: {ranking_correct}")
    print(f"   - Best scenario: {comparison.get_best_scenario()}")
    
    # Display welfare ranking
    print("\n   Welfare Ranking (1 = best):")
    summary_table = comparison.summary_table()
    for _, row in summary_table.iterrows():
        print(f"     {row['Ranking']}. {row['Scenario']}: {row['Welfare']:.6f}")
    
    # 6. Compute welfare gains
    print("\n6. Computing welfare gains relative to baseline...")
    
    welfare_gains = evaluator.compute_welfare_gains(scenarios)
    
    print("   Welfare gains (relative to baseline):")
    for scenario_name, gain in welfare_gains.items():
        gain_pct = (gain / abs(scenarios[0].welfare_outcome)) * 100
        print(f"     * {scenario_name}: {gain:.6f} ({gain_pct:+.2f}%)")
    
    # 7. Welfare decomposition analysis
    print("\n7. Analyzing welfare decomposition...")
    
    baseline_scenario = next(s for s in scenarios if s.scenario_type == 'baseline')
    perfect_regional_scenario = next(s for s in scenarios if s.scenario_type == 'perfect_regional')
    
    decomposition = evaluator.decompose_welfare_costs(baseline_scenario, perfect_regional_scenario)
    
    print("   Welfare decomposition (Baseline vs Perfect Regional):")
    print(f"     * Total welfare difference: {decomposition.total_welfare_difference:.6f}")
    print(f"     * Output gap component: {decomposition.output_gap_component:.6f}")
    print(f"     * Inflation component: {decomposition.inflation_component:.6f}")
    print(f"     * Regional distribution component: {decomposition.regional_distribution_component:.6f}")
    print(f"     * Welfare improvement: {decomposition.get_welfare_improvement_pct():.2f}%")
    
    # 8. Create visualizations
    print("\n8. Creating visualizations...")
    
    create_visualizations(scenarios, historical_data, comparison)
    
    print("   - Saved visualization plots")
    
    return scenarios, comparison, evaluator


def create_visualizations(scenarios, historical_data, comparison):
    """
    Create comprehensive visualizations of counterfactual analysis results.
    
    Args:
        scenarios: List of policy scenarios
        historical_data: Regional economic data
        comparison: ComparisonResults object
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Policy rate paths comparison
    ax1 = plt.subplot(2, 3, 1)
    for scenario in scenarios:
        plt.plot(scenario.policy_rates.index, scenario.policy_rates.values, 
                label=scenario.scenario_type.replace('_', ' ').title(), linewidth=2)
    
    plt.title('Policy Rate Paths Across Scenarios', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Policy Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Welfare outcomes comparison
    ax2 = plt.subplot(2, 3, 2)
    scenario_names = [s.scenario_type.replace('_', ' ').title() for s in scenarios]
    welfare_outcomes = [s.welfare_outcome for s in scenarios]
    
    bars = plt.bar(scenario_names, welfare_outcomes, 
                   color=['red', 'orange', 'lightblue', 'green'])
    plt.title('Welfare Outcomes by Scenario', fontsize=12, fontweight='bold')
    plt.ylabel('Welfare (negative of loss)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, welfare_outcomes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Regional output gap volatility
    ax3 = plt.subplot(2, 3, 3)
    
    baseline_scenario = next(s for s in scenarios if s.scenario_type == 'baseline')
    perfect_scenario = next(s for s in scenarios if s.scenario_type == 'perfect_regional')
    
    n_regions = len(historical_data.regions)
    regions = [f"Region {i+1}" for i in range(n_regions)]
    
    # Compute output gap volatilities
    baseline_volatility = []
    perfect_volatility = []
    
    for i in range(n_regions):
        baseline_og = baseline_scenario.regional_outcomes.iloc[i]
        perfect_og = perfect_scenario.regional_outcomes.iloc[i]
        
        baseline_volatility.append(baseline_og.std())
        perfect_volatility.append(perfect_og.std())
    
    x = np.arange(len(regions))
    width = 0.35
    
    plt.bar(x - width/2, baseline_volatility, width, label='Baseline', alpha=0.8)
    plt.bar(x + width/2, perfect_volatility, width, label='Perfect Regional', alpha=0.8)
    
    plt.title('Output Gap Volatility by Region', fontsize=12, fontweight='bold')
    plt.xlabel('Region')
    plt.ylabel('Standard Deviation')
    plt.xticks(x, regions)
    plt.legend()
    
    # 4. Welfare ranking verification
    ax4 = plt.subplot(2, 3, 4)
    
    ranking_data = comparison.summary_table().sort_values('Ranking')
    
    plt.barh(range(len(ranking_data)), ranking_data['Welfare'], 
             color=['red', 'orange', 'lightblue', 'green'])
    plt.yticks(range(len(ranking_data)), ranking_data['Scenario'])
    plt.xlabel('Welfare Outcome')
    plt.title('Welfare Ranking\n(Theory: PR ≥ PI ≥ OR ≥ B)', fontsize=12, fontweight='bold')
    
    # Add ranking numbers
    for i, (_, row) in enumerate(ranking_data.iterrows()):
        plt.text(row['Welfare'] + 0.001, i, f"#{int(row['Ranking'])}", 
                va='center', fontweight='bold')
    
    # 5. Policy rate volatility comparison
    ax5 = plt.subplot(2, 3, 5)
    
    rate_volatilities = [scenario.policy_rates.std() for scenario in scenarios]
    scenario_labels = [s.scenario_type.replace('_', ' ').title() for s in scenarios]
    
    plt.bar(scenario_labels, rate_volatilities, 
            color=['red', 'orange', 'lightblue', 'green'], alpha=0.7)
    plt.title('Policy Rate Volatility', fontsize=12, fontweight='bold')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    
    # 6. Regional welfare distribution
    ax6 = plt.subplot(2, 3, 6)
    
    # Compute regional welfare losses for baseline vs perfect regional
    regional_welfare_baseline = []
    regional_welfare_perfect = []
    
    for i in range(n_regions):
        # Extract regional outcomes
        baseline_og = baseline_scenario.regional_outcomes.iloc[i]
        baseline_inf = baseline_scenario.regional_outcomes.iloc[i + n_regions]
        perfect_og = perfect_scenario.regional_outcomes.iloc[i]
        perfect_inf = perfect_scenario.regional_outcomes.iloc[i + n_regions]
        
        # Compute regional welfare losses
        baseline_loss = np.mean(baseline_og**2 + baseline_inf**2)
        perfect_loss = np.mean(perfect_og**2 + perfect_inf**2)
        
        regional_welfare_baseline.append(baseline_loss)
        regional_welfare_perfect.append(perfect_loss)
    
    x = np.arange(len(regions))
    width = 0.35
    
    plt.bar(x - width/2, regional_welfare_baseline, width, label='Baseline', alpha=0.8)
    plt.bar(x + width/2, regional_welfare_perfect, width, label='Perfect Regional', alpha=0.8)
    
    plt.title('Regional Welfare Losses', fontsize=12, fontweight='bold')
    plt.xlabel('Region')
    plt.ylabel('Average Loss')
    plt.xticks(x, regions)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('counterfactual_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional time series plot
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Regional output gaps over time
    axes[0, 0].plot(historical_data.output_gaps.T)
    axes[0, 0].set_title('Regional Output Gaps (Historical Data)')
    axes[0, 0].set_ylabel('Output Gap')
    axes[0, 0].legend([f'Region {i+1}' for i in range(n_regions)])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Regional inflation rates over time
    axes[0, 1].plot(historical_data.inflation_rates.T)
    axes[0, 1].set_title('Regional Inflation Rates (Historical Data)')
    axes[0, 1].set_ylabel('Inflation Rate (%)')
    axes[0, 1].legend([f'Region {i+1}' for i in range(n_regions)])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy rate differences from baseline
    for i, scenario in enumerate(scenarios[1:]):
        rate_diff = scenario.policy_rates - baseline_scenario.policy_rates
        axes[1, 0].plot(baseline_scenario.policy_rates.index, rate_diff, 
                       label=scenario.scenario_type.replace('_', ' ').title())
    axes[1, 0].set_title('Policy Rate Differences from Baseline')
    axes[1, 0].set_ylabel('Rate Difference (pp)')
    axes[1, 0].legend([s.scenario_type.replace('_', ' ').title() for s in scenarios[1:]])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative welfare gains over time (simplified as constant gains)
    for i, scenario in enumerate(scenarios[1:]):
        welfare_gain = scenario.welfare_outcome - baseline_scenario.welfare_outcome
        constant_gains = np.full(len(baseline_scenario.policy_rates), welfare_gain)
        axes[1, 1].plot(baseline_scenario.policy_rates.index, constant_gains,
                       label=scenario.scenario_type.replace('_', ' ').title())
    axes[1, 1].set_title('Cumulative Welfare Gains vs Baseline')
    axes[1, 1].set_ylabel('Cumulative Welfare Gain')
    axes[1, 1].legend([s.scenario_type.replace('_', ' ').title() for s in scenarios[1:]])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('counterfactual_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the complete demonstration
    scenarios, comparison, evaluator = run_counterfactual_analysis()
    
    print("\n" + "=" * 60)
    print("Counterfactual Analysis Demo Complete!")
    print("\nKey Results:")
    print(f"- Generated {len(scenarios)} policy scenarios")
    print(f"- Welfare ranking follows theory: {evaluator.verify_welfare_ranking(scenarios)}")
    print(f"- Best performing scenario: {comparison.get_best_scenario()}")
    print("\nVisualization files saved:")
    print("- counterfactual_analysis_results.png")
    print("- counterfactual_time_series.png")