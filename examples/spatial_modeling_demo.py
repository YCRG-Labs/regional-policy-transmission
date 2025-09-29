"""
Demonstration of spatial modeling infrastructure for regional monetary policy analysis.

This script shows how to use the SpatialModelHandler to construct and validate
spatial weight matrices from various data sources.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler


def create_sample_data():
    """Create sample data for demonstration."""
    
    # Define regions
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
    
    # Sample trade flow data
    trade_data = pd.DataFrame({
        'origin': ['Northeast', 'Northeast', 'Southeast', 'Southeast', 'Midwest', 
                  'Midwest', 'Southwest', 'Southwest', 'West', 'West'],
        'destination': ['Southeast', 'Midwest', 'Northeast', 'Southwest', 'Northeast',
                       'West', 'Southeast', 'West', 'Southwest', 'Midwest'],
        'trade_flow': [150, 120, 140, 100, 110, 90, 80, 70, 85, 95]
    })
    
    # Sample migration flow data
    migration_data = pd.DataFrame({
        'origin': ['Northeast', 'Northeast', 'Southeast', 'Southeast', 'Midwest',
                  'Midwest', 'Southwest', 'Southwest', 'West', 'West'],
        'destination': ['Southeast', 'West', 'Northeast', 'Southwest', 'Northeast',
                       'West', 'Southeast', 'West', 'Southwest', 'Northeast'],
        'migration_flow': [25, 30, 20, 35, 15, 40, 18, 45, 22, 12]
    })
    
    # Sample financial linkage data
    financial_data = pd.DataFrame({
        'region1': ['Northeast', 'Northeast', 'Northeast', 'Northeast',
                   'Southeast', 'Southeast', 'Southeast',
                   'Midwest', 'Midwest', 'Southwest'],
        'region2': ['Southeast', 'Midwest', 'Southwest', 'West',
                   'Midwest', 'Southwest', 'West',
                   'Southwest', 'West', 'West'],
        'financial_linkage': [0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.4, 0.6, 0.5, 0.3]
    })
    
    # Sample geographic coordinates (approximate US regional centers)
    coordinates = pd.DataFrame({
        'latitude': [41.0, 33.0, 42.0, 32.0, 37.0],
        'longitude': [-74.0, -84.0, -87.0, -97.0, -119.0]
    }, index=regions)
    
    return regions, trade_data, migration_data, financial_data, coordinates


def demonstrate_spatial_modeling():
    """Demonstrate spatial modeling capabilities."""
    
    print("Regional Monetary Policy - Spatial Modeling Demonstration")
    print("=" * 60)
    
    # Create sample data
    regions, trade_data, migration_data, financial_data, coordinates = create_sample_data()
    
    # Initialize spatial model handler
    handler = SpatialModelHandler(regions)
    print(f"\nInitialized SpatialModelHandler for {handler.n_regions} regions:")
    print(f"Regions: {', '.join(regions)}")
    
    # Create distance matrix
    print("\n1. Creating distance matrix from coordinates...")
    distance_matrix = handler.create_distance_matrix(coordinates, distance_type="haversine")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print("Sample distances (km):")
    print(f"  Northeast to Southeast: {distance_matrix[0, 1]:.1f}")
    print(f"  Midwest to West: {distance_matrix[2, 4]:.1f}")
    
    # Construct spatial weight matrix
    print("\n2. Constructing spatial weight matrix...")
    results = handler.construct_weights(
        trade_data=trade_data,
        migration_data=migration_data,
        financial_data=financial_data,
        distance_matrix=distance_matrix,
        weights=(0.4, 0.3, 0.2, 0.1),  # Trade, migration, financial, distance
        normalize_method="row"
    )
    
    print(f"Weight matrix constructed using method: {results.construction_method}")
    print(f"Component weights: {results.component_weights}")
    
    # Validate the matrix
    print("\n3. Validation results:")
    validation = results.validation_report
    print(f"Matrix is valid: {validation.is_valid}")
    print(f"Spectral radius: {validation.properties.get('spectral_radius', 'N/A'):.4f}")
    print(f"Row sums mean: {validation.properties.get('row_sums_mean', 'N/A'):.4f}")
    
    if validation.warnings:
        print("Warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")
    
    # Display the weight matrix
    print("\n4. Spatial weight matrix:")
    W = results.weight_matrix
    weight_df = pd.DataFrame(W, index=regions, columns=regions)
    print(weight_df.round(4))
    
    # Demonstrate spatial lag computation
    print("\n5. Computing spatial lags...")
    
    # Create sample regional data (e.g., output gaps)
    np.random.seed(42)  # For reproducible results
    sample_data = pd.DataFrame(
        np.random.randn(12, len(regions)) * 2,  # 12 time periods
        columns=regions,
        index=pd.date_range('2023-01', periods=12, freq='ME')
    )
    
    # Compute spatial lags
    spatial_lags = handler.compute_spatial_lags(sample_data, W)
    
    print("Original data (first 3 periods):")
    print(sample_data.head(3).round(3))
    print("\nSpatial lags (first 3 periods):")
    print(spatial_lags.head(3).round(3))
    
    # Test for spatial autocorrelation
    print("\n6. Testing for spatial autocorrelation...")
    
    # Create residuals with some spatial pattern
    residuals = sample_data + 0.3 * spatial_lags  # Add spatial correlation
    
    autocorr_results = handler.test_spatial_autocorrelation(residuals, W)
    print(f"Moran's I statistic (mean): {autocorr_results.get('moran_i_mean', 'N/A'):.4f}")
    print(f"Expected under null: {autocorr_results.get('moran_i_expected', 'N/A'):.4f}")
    
    # Component analysis
    print("\n7. Component matrix analysis:")
    for component, matrix in results.component_matrices.items():
        if np.any(matrix > 0):
            print(f"\n{component.capitalize()} component:")
            component_df = pd.DataFrame(matrix, index=regions, columns=regions)
            print(f"  Max connection: {np.max(matrix):.4f}")
            print(f"  Mean connection: {np.mean(matrix[matrix > 0]):.4f}")
            print(f"  Non-zero connections: {np.sum(matrix > 0)}")
    
    print("\n" + "=" * 60)
    print("Spatial modeling demonstration completed successfully!")
    print("\nKey capabilities demonstrated:")
    print("✓ Multi-source spatial weight construction")
    print("✓ Matrix validation and diagnostics")
    print("✓ Distance matrix creation from coordinates")
    print("✓ Spatial lag computation")
    print("✓ Spatial autocorrelation testing")
    print("✓ Component analysis and decomposition")


if __name__ == "__main__":
    demonstrate_spatial_modeling()