"""
Demonstration of configuration and extensibility framework.
Shows how to use the flexible configuration system, model extensions, and version control.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from regional_monetary_policy.config.config_manager import ConfigManager
from regional_monetary_policy.config.extensibility import (
    ExtensibilityConfig, SpatialWeightMethod, IdentificationStrategy,
    ModelExtension, RobustnessCheck, ParameterRestriction
)


def demonstrate_basic_configuration():
    """Demonstrate basic configuration management."""
    print("=== Basic Configuration Management ===")
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Load default configuration
    settings = config_manager.load_config()
    
    print(f"Analysis name: {settings.analysis_name}")
    print(f"Output directory: {settings.output_directory}")
    print(f"Number of regions: {len(settings.data.regions)}")
    print(f"Estimation method: {settings.estimation.estimation_method}")
    print(f"Has version control: {config_manager.has_version_control}")
    print(f"Has extensibility: {config_manager.has_extensibility}")
    
    return config_manager


def demonstrate_extensibility_framework(config_manager):
    """Demonstrate extensibility framework features."""
    print("\n=== Extensibility Framework ===")
    
    # Load extensibility configuration
    config_manager.load_extensibility_config()
    
    # Configure spatial weight methods
    ext_config = config_manager.extensibility_config
    ext_config.add_spatial_weight_method(SpatialWeightMethod.DISTANCE_ONLY)
    ext_config.add_spatial_weight_method(SpatialWeightMethod.TRADE_ONLY)
    
    print(f"Spatial weight methods: {[method.value for method in ext_config.spatial_weight_methods]}")
    
    # Add identification strategies
    ext_config.identification_strategies.append(IdentificationStrategy.ALTERNATIVE_INSTRUMENTS)
    ext_config.identification_strategies.append(IdentificationStrategy.HETEROSKEDASTICITY)
    
    print(f"Identification strategies: {[strategy.value for strategy in ext_config.identification_strategies]}")
    
    # Add parameter restrictions
    ext_config.add_parameter_restriction("sigma", "bounds", lower_bound=0.0, upper_bound=5.0)
    ext_config.add_parameter_restriction("kappa", "bounds", lower_bound=0.0, upper_bound=2.0)
    ext_config.add_parameter_restriction("beta", "equality", value=0.99)
    
    print(f"Parameter restrictions: {len(ext_config.parameter_restrictions)}")
    for restriction in ext_config.parameter_restrictions:
        print(f"  - {restriction.parameter_name}: {restriction.restriction_type}")
    
    # Enable model extensions
    ext_config.enable_model_extension(
        ModelExtension.FINANCIAL_FRICTIONS,
        {"include_credit_spreads": True, "include_bank_lending": True}
    )
    
    ext_config.enable_model_extension(
        ModelExtension.FISCAL_POLICY,
        {"include_government_spending": True, "include_taxes": False}
    )
    
    print(f"Model extensions: {[ext.value for ext in ext_config.model_extensions]}")
    
    # Add robustness checks
    ext_config.add_robustness_check(
        "subsample_stability",
        "Test parameter stability across subsamples",
        {"split_date": "2008-01-01", "test_statistic": "chow"}
    )
    
    ext_config.add_robustness_check(
        "alternative_instruments",
        "Estimate with alternative instrument set",
        {"instrument_set": "external"}
    )
    
    ext_config.add_robustness_check(
        "bootstrap_inference",
        "Bootstrap standard errors and confidence intervals",
        {"n_bootstrap": 1000, "block_size": 12}
    )
    
    print(f"Robustness checks: {len(ext_config.robustness_checks)}")
    for check in ext_config.robustness_checks:
        print(f"  - {check.name}: {check.description}")


def demonstrate_spatial_weight_builders(config_manager):
    """Demonstrate spatial weight construction."""
    print("\n=== Spatial Weight Construction ===")
    
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    
    # Trade-based weights
    print("Building trade-based spatial weights...")
    trade_builder = config_manager.get_spatial_weight_builder("trade_only")
    
    # Mock trade flows data
    trade_flows = {}
    for i, region_i in enumerate(regions):
        for j, region_j in enumerate(regions):
            if i != j:
                # Simulate trade flows (higher between adjacent regions)
                flow = np.random.exponential(100) if abs(i - j) == 1 else np.random.exponential(50)
                trade_flows[(region_i, region_j)] = flow
    
    trade_data = {"trade_flows": trade_flows}
    trade_weights = trade_builder.build_weights(regions, trade_data)
    
    print(f"Trade weights shape: {trade_weights.shape}")
    print(f"Row sums (should be ~1): {trade_weights.sum(axis=1)}")
    
    # Distance-based weights
    print("\nBuilding distance-based spatial weights...")
    distance_builder = config_manager.get_spatial_weight_builder(
        "distance_only", 
        decay_parameter=1.5, 
        cutoff_distance=1000
    )
    
    # Mock distance matrix (in miles)
    np.random.seed(42)
    distances = np.random.uniform(200, 2000, (len(regions), len(regions)))
    distances = (distances + distances.T) / 2  # Make symmetric
    np.fill_diagonal(distances, 0)
    
    distance_data = {"distances": distances}
    distance_weights = distance_builder.build_weights(regions, distance_data)
    
    print(f"Distance weights shape: {distance_weights.shape}")
    print(f"Row sums (should be ~1): {distance_weights.sum(axis=1)}")


def demonstrate_model_extensions(config_manager):
    """Demonstrate model extensions."""
    print("\n=== Model Extensions ===")
    
    # Base model specification
    base_specification = {
        "is_curve": {
            "equation": "x_t = E_t[x_{t+1}] - sigma * (r_t - E_t[pi_{t+1}]) + epsilon_x_t",
            "parameters": ["sigma"],
            "description": "Dynamic IS curve"
        },
        "phillips_curve": {
            "equation": "pi_t = beta * E_t[pi_{t+1}] + kappa * x_t + epsilon_pi_t",
            "parameters": ["beta", "kappa"],
            "description": "New Keynesian Phillips curve"
        },
        "taylor_rule": {
            "equation": "r_t = rho * r_{t-1} + (1-rho) * (phi_pi * pi_t + phi_x * x_t) + epsilon_r_t",
            "parameters": ["rho", "phi_pi", "phi_x"],
            "description": "Taylor rule for monetary policy"
        }
    }
    
    print("Base model equations:")
    for name, spec in base_specification.items():
        print(f"  {name}: {spec['description']}")
        print(f"    Parameters: {spec['parameters']}")
    
    # Get extended specification
    extended_specification = config_manager.get_model_specification(base_specification)
    
    print(f"\nExtended model equations ({len(extended_specification)} total):")
    for name, spec in extended_specification.items():
        if name not in base_specification:
            print(f"  {name}: {spec['description']}")
            print(f"    Parameters: {spec['parameters']}")
    
    # Show modified equations
    print("\nModified base equations:")
    for name in base_specification.keys():
        if name in extended_specification:
            base_params = set(base_specification[name]['parameters'])
            extended_params = set(extended_specification[name]['parameters'])
            new_params = extended_params - base_params
            if new_params:
                print(f"  {name}: Added parameters {list(new_params)}")


def demonstrate_parameter_restrictions(config_manager):
    """Demonstrate parameter restrictions."""
    print("\n=== Parameter Restrictions ===")
    
    # Simulate estimated parameters
    np.random.seed(42)
    n_regions = 5
    
    parameters = {
        "sigma": np.random.normal(1.5, 0.5, n_regions),  # Some may be negative or too large
        "kappa": np.random.normal(0.8, 0.3, n_regions),  # Some may be negative
        "beta": np.random.normal(0.99, 0.02, n_regions),  # Should be fixed at 0.99
        "phi_pi": np.random.normal(1.5, 0.2, n_regions)  # No restrictions
    }
    
    print("Original parameters:")
    for name, values in parameters.items():
        print(f"  {name}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    # Apply restrictions
    restricted_parameters = config_manager.apply_parameter_restrictions(parameters)
    
    print("\nRestricted parameters:")
    for name, values in restricted_parameters.items():
        print(f"  {name}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    # Show which parameters were affected
    print("\nRestriction effects:")
    for name in parameters.keys():
        original = parameters[name]
        restricted = restricted_parameters[name]
        if not np.array_equal(original, restricted):
            n_changed = np.sum(original != restricted)
            print(f"  {name}: {n_changed}/{len(original)} values changed")


def demonstrate_robustness_checks(config_manager):
    """Demonstrate robustness checks."""
    print("\n=== Robustness Checks ===")
    
    # Mock base results
    base_results = {
        "parameters": {
            "sigma": np.array([1.2, 1.5, 1.1, 1.8, 1.3]),
            "kappa": np.array([0.7, 0.9, 0.6, 1.1, 0.8]),
            "beta": np.array([0.99, 0.99, 0.99, 0.99, 0.99])
        },
        "standard_errors": {
            "sigma": np.array([0.1, 0.12, 0.09, 0.15, 0.11]),
            "kappa": np.array([0.08, 0.10, 0.07, 0.12, 0.09])
        },
        "welfare_loss": 0.025,
        "log_likelihood": -1250.5
    }
    
    # Mock data
    data = {
        "output_gaps": np.random.randn(120, 5),  # 10 years monthly data, 5 regions
        "inflation": np.random.randn(120, 5),
        "interest_rates": np.random.randn(120),
        "dates": ["2010-01-01", "2020-12-31"]
    }
    
    print("Running robustness checks...")
    robustness_results = config_manager.run_robustness_checks(base_results, data)
    
    print(f"Completed {len(robustness_results)} robustness checks:")
    for check_name, results in robustness_results.items():
        print(f"\n  {check_name}:")
        print(f"    Description: {results.get('description', 'N/A')}")
        if 'error' in results:
            print(f"    Status: Failed - {results['error']}")
        else:
            print(f"    Status: {results['results']['status']}")
            print(f"    Notes: {results['results']['notes']}")


def demonstrate_version_control(config_manager):
    """Demonstrate version control features."""
    print("\n=== Version Control ===")
    
    if not config_manager.has_version_control:
        print("Version control not available")
        return
    
    # Save current configuration as a version
    version_id = config_manager.save_config_version(
        description="Demo configuration with extensions",
        tags=["demo", "extensions", "baseline"]
    )
    
    print(f"Saved configuration version: {version_id}")
    
    # Start an analysis run
    run_parameters = {
        "estimation_method": "three_stage_gmm",
        "spatial_weights": "trade_migration",
        "robustness_checks": True
    }
    
    run_id = config_manager.start_analysis_run(run_parameters)
    print(f"Started analysis run: {run_id}")
    
    # Simulate analysis completion
    import time
    time.sleep(0.1)  # Simulate some work
    
    config_manager.complete_analysis_run(
        run_id, 
        results_path="demo_results/", 
        execution_time=0.1
    )
    print(f"Completed analysis run: {run_id}")
    
    # Show version history
    versions = config_manager.get_config_history(limit=5)
    print(f"\nConfiguration history ({len(versions)} versions):")
    for version in versions:
        tags_str = f" [{', '.join(version.tags)}]" if version.tags else ""
        print(f"  {version.version_id}{tags_str}: {version.description}")
    
    # Show run history
    runs = config_manager.get_run_history(limit=5)
    print(f"\nAnalysis run history ({len(runs)} runs):")
    for run in runs:
        print(f"  {run.run_id}: {run.status} ({run.timestamp.strftime('%Y-%m-%d %H:%M')})")
    
    # Create reproducibility report
    if runs:
        report = config_manager.create_reproducibility_report(runs[0].run_id)
        print(f"\nReproducibility report for {runs[0].run_id}:")
        print(f"  Configuration version: {report['configuration']['version_id']}")
        print(f"  Config hash: {report['configuration']['config_hash']}")
        print(f"  Execution time: {report['run_info']['execution_time']} seconds")


def demonstrate_configuration_export(config_manager):
    """Demonstrate configuration template export."""
    print("\n=== Configuration Export ===")
    
    # Export comprehensive configuration template
    template_path = "comprehensive_config_template.json"
    config_manager.export_configuration_template(template_path, include_extensibility=True)
    
    print(f"Exported configuration template to: {template_path}")
    
    # Show template structure
    import json
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    print("\nTemplate sections:")
    for section in template.keys():
        if section.startswith('_'):
            continue
        if isinstance(template[section], dict):
            print(f"  {section}: {len(template[section])} items")
        else:
            print(f"  {section}: {template[section]}")
    
    if '_documentation' in template:
        print(f"\nDocumentation sections: {list(template['_documentation']['sections'].keys())}")


def main():
    """Run the complete demonstration."""
    print("Regional Monetary Policy Analysis - Configuration & Extensibility Demo")
    print("=" * 70)
    
    try:
        # Basic configuration
        config_manager = demonstrate_basic_configuration()
        
        # Extensibility framework
        demonstrate_extensibility_framework(config_manager)
        
        # Spatial weight construction
        demonstrate_spatial_weight_builders(config_manager)
        
        # Model extensions
        demonstrate_model_extensions(config_manager)
        
        # Parameter restrictions
        demonstrate_parameter_restrictions(config_manager)
        
        # Robustness checks
        demonstrate_robustness_checks(config_manager)
        
        # Version control
        demonstrate_version_control(config_manager)
        
        # Configuration export
        demonstrate_configuration_export(config_manager)
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("- Flexible configuration system with validation")
        print("- Multiple spatial weight construction methods")
        print("- Model extensions (financial frictions, fiscal policy)")
        print("- Parameter restrictions and robustness checks")
        print("- Version control for reproducible research")
        print("- Configuration templates and documentation")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()