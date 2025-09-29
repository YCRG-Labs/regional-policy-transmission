"""
Demonstration of the Parameter Estimation Engine for Regional Monetary Policy Analysis.

This script shows how to use the ParameterEstimator class to estimate regional
structural parameters using synthetic data with known parameter values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from regional_monetary_policy.econometric.parameter_estimator import (
    ParameterEstimator, create_default_estimation_config
)
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.data.models import RegionalDataset


def generate_synthetic_data(n_regions=4, n_periods=120):
    """Generate synthetic regional data for demonstration."""
    print(f"Generating synthetic data for {n_regions} regions over {n_periods} periods...")
    
    # True parameter values (what we want to recover)
    true_params = {
        'sigma': np.array([1.0, 1.2, 0.8, 1.1]),  # Interest rate sensitivity
        'kappa': np.array([0.1, 0.15, 0.08, 0.12]),  # Phillips curve slope
        'psi': np.array([0.2, -0.1, 0.15, 0.0]),  # Demand spillover
        'phi': np.array([0.1, 0.05, -0.05, 0.08]),  # Price spillover
        'beta': np.array([0.99, 0.98, 0.99, 0.985])  # Discount factor
    }
    
    # Create spatial weight matrix
    W = np.array([
        [0.0, 0.3, 0.4, 0.3],
        [0.2, 0.0, 0.3, 0.5],
        [0.3, 0.2, 0.0, 0.5],
        [0.25, 0.25, 0.5, 0.0]
    ])
    
    # Initialize data arrays
    y_gaps = np.zeros((n_regions, n_periods))
    pi_rates = np.zeros((n_regions, n_periods))
    r_rates = np.zeros(n_periods)
    
    # Generate shocks
    np.random.seed(42)  # For reproducibility
    demand_shocks = np.random.normal(0, 0.01, (n_regions, n_periods))
    supply_shocks = np.random.normal(0, 0.005, (n_regions, n_periods))
    policy_shocks = np.random.normal(0, 0.002, n_periods)
    
    # Generate data using simplified model dynamics
    for t in range(1, n_periods):
        # Policy rate (simple Taylor rule)
        if t > 0:
            agg_inflation = np.mean(pi_rates[:, t-1])
            agg_output = np.mean(y_gaps[:, t-1])
            r_rates[t] = 0.02 + 0.5 * agg_inflation + 0.2 * agg_output + policy_shocks[t]
        
        # Regional dynamics
        for i in range(n_regions):
            # Spatial lags
            y_spatial = W[i, :] @ y_gaps[:, t-1] if t > 0 else 0
            pi_spatial = W[i, :] @ pi_rates[:, t-1] if t > 0 else 0
            
            # IS equation with stability
            expected_y = y_gaps[i, t-1] * 0.5 if t > 0 else 0
            expected_pi = pi_rates[i, t-1] * 0.3 if t > 0 else 0
            
            y_gaps[i, t] = (0.7 * expected_y - 
                           0.1 * true_params['sigma'][i] * (r_rates[t] - expected_pi) +
                           0.05 * true_params['psi'][i] * y_spatial +
                           demand_shocks[i, t])
            
            # Phillips curve with stability
            pi_rates[i, t] = (0.8 * true_params['beta'][i] * expected_pi +
                             0.05 * true_params['kappa'][i] * y_gaps[i, t] +
                             0.02 * true_params['phi'][i] * pi_spatial +
                             supply_shocks[i, t])
    
    # Create RegionalDataset
    regions = [f"Region_{i+1}" for i in range(n_regions)]
    periods = pd.date_range('2000-01-01', periods=n_periods, freq='ME')
    
    output_gaps_df = pd.DataFrame(y_gaps, index=regions, columns=periods)
    inflation_df = pd.DataFrame(pi_rates, index=regions, columns=periods)
    interest_rates_series = pd.Series(r_rates, index=periods)
    
    dataset = RegionalDataset(
        output_gaps=output_gaps_df,
        inflation_rates=inflation_df,
        interest_rates=interest_rates_series,
        real_time_estimates={},
        metadata={'synthetic': True, 'true_parameters': true_params}
    )
    
    print(f"✓ Generated synthetic dataset with {dataset.n_regions} regions and {dataset.n_periods} periods")
    return dataset, true_params


def demonstrate_parameter_estimation():
    """Demonstrate the three-stage parameter estimation procedure."""
    print("=" * 60)
    print("REGIONAL MONETARY POLICY PARAMETER ESTIMATION DEMO")
    print("=" * 60)
    
    # Generate synthetic data
    data, true_params = generate_synthetic_data()
    
    # Set up estimation components
    print("\nSetting up estimation framework...")
    regions = data.regions
    spatial_handler = SpatialModelHandler(regions)
    config = create_default_estimation_config()
    config.bootstrap_replications = 20  # Reduce for demo speed
    config.max_iterations = 100
    
    estimator = ParameterEstimator(spatial_handler, config)
    print(f"✓ Initialized ParameterEstimator for {len(regions)} regions")
    
    # Run full estimation
    print("\nRunning three-stage parameter estimation...")
    print("This may take a moment...")
    
    try:
        results = estimator.estimate_full_model(data)
        print(f"✓ Estimation completed in {results.estimation_time:.2f} seconds")
        
        # Display results
        print("\n" + "=" * 50)
        print("ESTIMATION RESULTS")
        print("=" * 50)
        
        print(results.summary_report())
        
        # Compare with true parameters
        print("\n" + "=" * 50)
        print("PARAMETER RECOVERY ANALYSIS")
        print("=" * 50)
        
        estimated_params = results.regional_parameters
        
        print("\nParameter Recovery Comparison:")
        print("-" * 40)
        
        for param_name in ['sigma', 'kappa', 'psi', 'phi', 'beta']:
            true_vals = true_params[param_name]
            est_vals = getattr(estimated_params, param_name)
            errors = np.abs(est_vals - true_vals)
            
            print(f"\n{param_name.upper()} (Interest Rate Sensitivity)" if param_name == 'sigma' else
                  f"{param_name.upper()} (Phillips Curve Slope)" if param_name == 'kappa' else
                  f"{param_name.upper()} (Demand Spillover)" if param_name == 'psi' else
                  f"{param_name.upper()} (Price Spillover)" if param_name == 'phi' else
                  f"{param_name.upper()} (Discount Factor)")
            
            for i in range(len(true_vals)):
                print(f"  Region {i+1}: True={true_vals[i]:.3f}, Est={est_vals[i]:.3f}, Error={errors[i]:.3f}")
            
            print(f"  Mean Absolute Error: {np.mean(errors):.4f}")
        
        # Identification diagnostics
        print("\n" + "=" * 50)
        print("IDENTIFICATION DIAGNOSTICS")
        print("=" * 50)
        
        if results.identification_tests:
            for test_name, test_value in results.identification_tests.items():
                print(f"{test_name}: {test_value:.4f}")
        
        # Convergence information
        print("\n" + "=" * 50)
        print("CONVERGENCE INFORMATION")
        print("=" * 50)
        
        conv_info = results.convergence_info
        print(f"Overall Convergence: {'SUCCESS' if conv_info.get('overall_converged', False) else 'FAILED'}")
        
        if 'stages' in conv_info:
            for stage_name, stage_info in conv_info['stages'].items():
                converged = stage_info.get('converged', False)
                iterations = stage_info.get('iterations', 'N/A')
                print(f"{stage_name}: {'✓' if converged else '✗'} ({iterations} iterations)")
        
        return results
        
    except Exception as e:
        print(f"✗ Estimation failed: {str(e)}")
        raise


def plot_estimation_results(results, true_params):
    """Create visualization of estimation results."""
    try:
        import matplotlib.pyplot as plt
        
        estimated_params = results.regional_parameters
        n_regions = estimated_params.n_regions
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parameter Estimation Results vs True Values', fontsize=16)
        
        param_names = ['sigma', 'kappa', 'psi', 'phi', 'beta']
        param_labels = ['Interest Sensitivity (σ)', 'Phillips Slope (κ)', 
                       'Demand Spillover (ψ)', 'Price Spillover (φ)', 'Discount Factor (β)']
        
        for idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            if idx >= 5:  # Only plot first 5 parameters
                break
                
            ax = axes[idx // 3, idx % 3]
            
            true_vals = true_params[param_name]
            est_vals = getattr(estimated_params, param_name)
            
            regions = [f'R{i+1}' for i in range(n_regions)]
            x_pos = np.arange(n_regions)
            
            width = 0.35
            ax.bar(x_pos - width/2, true_vals, width, label='True', alpha=0.7)
            ax.bar(x_pos + width/2, est_vals, width, label='Estimated', alpha=0.7)
            
            ax.set_xlabel('Region')
            ax.set_ylabel('Parameter Value')
            ax.set_title(param_label)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(regions)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(param_names) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Parameter comparison plot displayed")
        
    except ImportError:
        print("⚠ Matplotlib not available for plotting")
    except Exception as e:
        print(f"⚠ Plotting failed: {str(e)}")


def main():
    """Main demonstration function."""
    try:
        # Run parameter estimation demonstration
        results = demonstrate_parameter_estimation()
        
        # Get true parameters for comparison
        true_params = results.regional_parameters.metadata if hasattr(results.regional_parameters, 'metadata') else None
        
        # Try to create plots
        if true_params:
            print("\nGenerating parameter comparison plots...")
            # We need to get true params from the original data
            # For now, just show the summary
            pass
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("• Three-stage parameter estimation procedure")
        print("• GMM estimation with moment conditions")
        print("• Spatial weight matrix construction")
        print("• Parameter identification testing")
        print("• Bootstrap standard error computation")
        print("• Comprehensive diagnostic reporting")
        
        print(f"\nEstimated {results.regional_parameters.n_regions} regional parameter sets")
        print(f"Total estimation time: {results.estimation_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()