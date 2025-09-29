"""
Policy Analysis and Mistake Decomposition Demo

This script demonstrates the policy analysis capabilities including:
- Policy mistake decomposition following Theorem 4
- Optimal policy calculation
- Fed reaction function estimation
- Information set reconstruction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from regional_monetary_policy.econometric.models import RegionalParameters
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.policy import (
    PolicyMistakeDecomposer, OptimalPolicyCalculator, WelfareFunction,
    FedReactionEstimator, InformationSetReconstructor
)


def create_sample_data():
    """Create sample regional economic data for demonstration."""
    print("Creating sample regional economic data...")
    
    # Time period: 2010-2020 (monthly data)
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='ME')
    n_periods = len(dates)
    n_regions = 3
    
    # Regional parameters (heterogeneous across regions)
    regional_params = RegionalParameters(
        sigma=np.array([0.5, 0.7, 0.6]),      # Interest rate sensitivities
        kappa=np.array([0.3, 0.4, 0.35]),    # Phillips curve slopes
        psi=np.array([0.1, 0.15, 0.12]),     # Demand spillovers
        phi=np.array([0.05, 0.08, 0.06]),    # Price spillovers
        beta=np.array([0.99, 0.99, 0.99]),   # Discount factors
        standard_errors={
            'sigma': np.array([0.05, 0.07, 0.06]),
            'kappa': np.array([0.03, 0.04, 0.035])
        },
        confidence_intervals={}
    )
    
    # Generate synthetic regional data
    np.random.seed(42)  # For reproducibility
    
    # Regional output gaps and inflation (regions x time)
    output_gaps = pd.DataFrame(
        np.random.normal(0, 1, (n_regions, n_periods)),
        index=[f'Region_{i+1}' for i in range(n_regions)],
        columns=dates
    )
    
    inflation_rates = pd.DataFrame(
        2.0 + np.random.normal(0, 0.5, (n_regions, n_periods)),
        index=[f'Region_{i+1}' for i in range(n_regions)],
        columns=dates
    )
    
    # Federal funds rate (time series)
    interest_rates = pd.Series(
        1.0 + np.random.normal(0, 0.3, n_periods),
        index=dates,
        name='fed_funds_rate'
    )
    
    # Create RegionalDataset
    regional_data = RegionalDataset(
        output_gaps=output_gaps,
        inflation_rates=inflation_rates,
        interest_rates=interest_rates,
        real_time_estimates={},
        metadata={'description': 'Synthetic regional data for demo'}
    )
    
    return regional_params, regional_data


def demonstrate_optimal_policy():
    """Demonstrate optimal policy calculation."""
    print("\n" + "="*60)
    print("OPTIMAL POLICY CALCULATION DEMO")
    print("="*60)
    
    regional_params, regional_data = create_sample_data()
    
    # Define welfare function
    welfare_function = WelfareFunction(
        output_gap_weight=1.0,
        inflation_weight=1.0,
        regional_weights=np.array([0.4, 0.35, 0.25]),  # Population-based weights
        loss_function='quadratic'
    )
    
    # Create optimal policy calculator
    calculator = OptimalPolicyCalculator(
        regional_params=regional_params,
        welfare_function=welfare_function
    )
    
    print(f"Optimal regional weights: {calculator.optimal_regional_weights}")
    print(f"Optimal policy coefficients: {calculator.optimal_coefficients}")
    
    # Compute optimal policy path for a subset of data
    subset_data = regional_data.output_gaps.T.join(regional_data.inflation_rates.T, rsuffix='_infl')
    subset_data.columns = [
        'output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
        'inflation_region_1', 'inflation_region_2', 'inflation_region_3'
    ]
    
    # Take first 24 months for demonstration
    demo_data = subset_data.iloc[:24]
    optimal_rates = calculator.compute_optimal_rate_path(demo_data)
    
    print(f"\nOptimal policy rates (first 6 months):")
    print(optimal_rates.head(6))
    
    # Analyze policy tradeoffs for a specific period
    period_data = demo_data.iloc[[12]]  # 13th month
    tradeoffs = calculator.analyze_policy_tradeoffs(period_data)
    
    print(f"\nPolicy tradeoffs analysis for {period_data.index[0]}:")
    print(f"  Optimal rate: {tradeoffs['optimal_rate']:.4f}")
    print(f"  Total welfare loss: {tradeoffs['total_welfare_loss']:.6f}")
    print(f"  Cross-regional variance: {tradeoffs['cross_regional_variance']:.6f}")
    
    return calculator, optimal_rates, demo_data


def demonstrate_policy_mistake_decomposition():
    """Demonstrate policy mistake decomposition."""
    print("\n" + "="*60)
    print("POLICY MISTAKE DECOMPOSITION DEMO")
    print("="*60)
    
    regional_params, regional_data = create_sample_data()
    
    # Social welfare weights
    social_weights = np.array([0.4, 0.35, 0.25])
    
    # Create policy mistake decomposer
    decomposer = PolicyMistakeDecomposer(
        regional_params=regional_params,
        social_welfare_weights=social_weights
    )
    
    print(f"Optimal regional weights: {decomposer.optimal_weights}")
    print(f"Optimal output coefficient: {decomposer.optimal_output_coeff:.4f}")
    print(f"Optimal inflation coefficient: {decomposer.optimal_inflation_coeff:.4f}")
    
    # Create sample real-time and true data for a specific period
    period_date = '2015-06-01'
    
    # Real-time data (with measurement errors)
    real_time_data = pd.DataFrame({
        'output_gap_region_1': [0.5],
        'output_gap_region_2': [-0.3],
        'output_gap_region_3': [0.1],
        'inflation_region_1': [2.1],
        'inflation_region_2': [1.8],
        'inflation_region_3': [2.0]
    }, index=[pd.Timestamp(period_date)])
    
    # True data (revised)
    true_data = pd.DataFrame({
        'output_gap_region_1': [0.6],   # Revised higher
        'output_gap_region_2': [-0.2],  # Revised higher
        'output_gap_region_3': [0.0],   # Revised lower
        'inflation_region_1': [2.0],    # Revised lower
        'inflation_region_2': [1.9],    # Revised higher
        'inflation_region_3': [2.1]     # Revised higher
    }, index=[pd.Timestamp(period_date)])
    
    # Simulate Fed decision
    actual_fed_rate = 1.25
    
    # Compute optimal rate using true data
    calculator = OptimalPolicyCalculator(
        regional_params=regional_params,
        welfare_function=WelfareFunction(regional_weights=social_weights)
    )
    optimal_rate = calculator.compute_optimal_rate(true_data)
    
    print(f"\nActual Fed rate: {actual_fed_rate:.4f}")
    print(f"Optimal rate: {optimal_rate:.4f}")
    print(f"Total policy mistake: {actual_fed_rate - optimal_rate:.4f}")
    
    # Decompose the policy mistake
    decomposition = decomposer.decompose_policy_mistake(
        actual_rate=actual_fed_rate,
        optimal_rate=optimal_rate,
        real_time_data=real_time_data,
        true_data=true_data
    )
    
    print(f"\nPolicy Mistake Decomposition:")
    print(f"  Information effect: {decomposition.information_effect:.4f}")
    print(f"  Weight misallocation: {decomposition.weight_misallocation_effect:.4f}")
    print(f"  Parameter misspecification: {decomposition.parameter_misspecification_effect:.4f}")
    print(f"  Inflation response: {decomposition.inflation_response_effect:.4f}")
    
    # Show relative contributions
    relative_contribs = decomposition.get_relative_contributions()
    print(f"\nRelative Contributions:")
    for component, pct in relative_contribs.items():
        print(f"  {component}: {pct:.1f}%")
    
    return decomposition


def demonstrate_fed_reaction_estimation():
    """Demonstrate Fed reaction function estimation."""
    print("\n" + "="*60)
    print("FED REACTION FUNCTION ESTIMATION DEMO")
    print("="*60)
    
    regional_params, regional_data = create_sample_data()
    
    # Create Fed reaction estimator
    estimator = FedReactionEstimator(estimation_method='ols')
    
    # Prepare data for estimation
    policy_rates = regional_data.interest_rates
    
    # Convert regional data to format expected by estimator (time x regions)
    regional_df = regional_data.output_gaps.T.join(regional_data.inflation_rates.T, rsuffix='_infl')
    regional_df.columns = [
        'output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3',
        'inflation_region_1', 'inflation_region_2', 'inflation_region_3'
    ]
    
    # Estimate Fed reaction function
    results = estimator.estimate_reaction_function(policy_rates, regional_df)
    
    print("Fed Reaction Function Results:")
    print(f"  R-squared: {results.model_fit['r_squared']:.4f}")
    print(f"  Estimated coefficients:")
    for coeff, value in results.estimated_coefficients.items():
        if 'output_gap' in coeff or 'inflation' in coeff:
            print(f"    {coeff}: {value:.4f}")
    
    if results.implicit_regional_weights is not None:
        print(f"  Implicit regional weights: {results.implicit_regional_weights}")
    
    # Test Taylor principle
    taylor_test = estimator.test_taylor_principle(results)
    print(f"\nTaylor Principle Test:")
    print(f"  Inflation coefficient: {taylor_test['inflation_coefficient']:.4f}")
    print(f"  Satisfies Taylor principle: {taylor_test['satisfies_taylor_principle']}")
    
    return results


def demonstrate_information_reconstruction():
    """Demonstrate information set reconstruction."""
    print("\n" + "="*60)
    print("INFORMATION SET RECONSTRUCTION DEMO")
    print("="*60)
    
    regional_params, regional_data = create_sample_data()
    
    # Create information set reconstructor
    reconstructor = InformationSetReconstructor()
    
    # Reconstruct information set for a specific FOMC meeting
    decision_date = '2015-06-17'  # June 2015 FOMC meeting
    
    info_set = reconstructor.reconstruct_information_set(
        decision_date=decision_date,
        real_time_data=regional_data
    )
    
    print(f"Information Set for {decision_date}:")
    print(f"  Available data series: {len(info_set.available_data.columns)}")
    print(f"  Data range: {info_set.available_data.index.min()} to {info_set.available_data.index.max()}")
    
    # Show latest observations for each series
    print(f"\nLatest available observations:")
    for series in info_set.available_data.columns[:6]:  # Show first 6 series
        latest_value, latest_date = info_set.get_latest_observation(series)
        if latest_value is not None:
            print(f"  {series}: {latest_value:.4f} (as of {latest_date})")
    
    # Reconstruct sequence of information sets
    decision_dates = ['2015-03-18', '2015-06-17', '2015-09-17', '2015-12-16']
    info_sets = reconstructor.reconstruct_historical_sequence(decision_dates, regional_data)
    
    print(f"\nReconstructed {len(info_sets)} information sets")
    
    # Analyze information evolution
    evolution_df = reconstructor.analyze_information_evolution(info_sets)
    print(f"\nInformation Evolution Analysis:")
    print(evolution_df[['decision_date', 'n_series', 'data_completeness']])
    
    return info_set


def create_visualization():
    """Create visualizations of policy analysis results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Get results from demonstrations
    calculator, optimal_rates, demo_data = demonstrate_optimal_policy()
    decomposition = demonstrate_policy_mistake_decomposition()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Optimal policy rates over time
    optimal_rates.plot(ax=ax1, title='Optimal Policy Rates Over Time', 
                      xlabel='Date', ylabel='Interest Rate (%)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regional output gaps
    demo_data[['output_gap_region_1', 'output_gap_region_2', 'output_gap_region_3']].plot(
        ax=ax2, title='Regional Output Gaps', xlabel='Date', ylabel='Output Gap (%)'
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Region 1', 'Region 2', 'Region 3'])
    
    # Plot 3: Regional inflation rates
    demo_data[['inflation_region_1', 'inflation_region_2', 'inflation_region_3']].plot(
        ax=ax3, title='Regional Inflation Rates', xlabel='Date', ylabel='Inflation (%)'
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(['Region 1', 'Region 2', 'Region 3'])
    
    # Plot 4: Policy mistake decomposition
    components = ['Information\nEffect', 'Weight\nMisallocation', 
                 'Parameter\nMisspec', 'Inflation\nResponse']
    values = [
        decomposition.information_effect,
        decomposition.weight_misallocation_effect,
        decomposition.parameter_misspecification_effect,
        decomposition.inflation_response_effect
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = ax4.bar(components, values, color=colors)
    ax4.set_title('Policy Mistake Decomposition')
    ax4.set_ylabel('Contribution to Policy Mistake')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + np.sign(height)*0.001,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('policy_analysis_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'policy_analysis_demo.png'")
    
    return fig


def main():
    """Run the complete policy analysis demonstration."""
    print("REGIONAL MONETARY POLICY ANALYSIS DEMO")
    print("="*60)
    print("This demo showcases the policy analysis capabilities of the")
    print("regional monetary policy framework, including:")
    print("- Optimal policy calculation")
    print("- Policy mistake decomposition")
    print("- Fed reaction function estimation")
    print("- Information set reconstruction")
    
    try:
        # Run demonstrations
        demonstrate_optimal_policy()
        demonstrate_policy_mistake_decomposition()
        demonstrate_fed_reaction_estimation()
        demonstrate_information_reconstruction()
        
        # Create visualizations
        create_visualization()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All policy analysis components have been demonstrated.")
        print("Check the generated visualization: policy_analysis_demo.png")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()