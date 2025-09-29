"""
Sample synthetic data for testing and demonstration purposes.
This allows the system to work without requiring FRED API keys.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

def generate_synthetic_regional_data(
    regions: List[str] = None,
    start_date: str = "2000-01-01",
    end_date: str = "2023-12-31",
    frequency: str = "QE",  # Fixed deprecation warning
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic regional economic data for testing.
    
    Args:
        regions: List of region codes (default: US states)
        start_date: Start date for data
        end_date: End date for data
        frequency: Data frequency ('QE' for quarterly, 'ME' for monthly)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with regional datasets
    """
    np.random.seed(seed)
    
    if regions is None:
        regions = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    n_periods = len(date_range)
    n_regions = len(regions)
    
    # Generate correlated regional shocks
    correlation_matrix = generate_spatial_correlation_matrix(regions)
    
    data = {}
    
    for i, region in enumerate(regions):
        # Generate base economic indicators with realistic properties
        
        # Output gap (mean-reverting with regional persistence)
        output_gap = generate_ar_process(n_periods, phi=0.7, sigma=0.8, seed=seed+i)
        
        # Inflation (with trend and cyclical components)
        trend_inflation = 2.0 + 0.5 * np.sin(np.arange(n_periods) * 2 * np.pi / 40)  # 10-year cycle
        inflation_shock = generate_ar_process(n_periods, phi=0.5, sigma=0.6, seed=seed+i+100)
        inflation = trend_inflation + inflation_shock
        
        # Unemployment rate (counter-cyclical to output gap)
        natural_rate = 5.5 + np.random.normal(0, 0.5)  # Regional variation
        unemployment = natural_rate - 0.4 * output_gap + generate_ar_process(n_periods, phi=0.8, sigma=0.3, seed=seed+i+200)
        unemployment = np.maximum(unemployment, 1.0)  # Floor at 1%
        
        # Regional GDP growth
        trend_growth = 2.5 + np.random.normal(0, 0.3)  # Regional trend
        gdp_growth = trend_growth + 0.3 * output_gap + generate_ar_process(n_periods, phi=0.3, sigma=1.2, seed=seed+i+300)
        
        # House price index (regional variation)
        house_price_growth = 3.0 + 0.5 * output_gap + generate_ar_process(n_periods, phi=0.9, sigma=1.0, seed=seed+i+400)
        house_price_index = 100 * np.cumprod(1 + house_price_growth/100)
        
        # Create DataFrame
        regional_data = pd.DataFrame({
            'date': date_range,
            'region': region,
            'output_gap': output_gap,
            'inflation': inflation,
            'unemployment': unemployment,
            'gdp_growth': gdp_growth,
            'house_price_index': house_price_index,
            'house_price_growth': house_price_growth
        })
        
        data[region] = regional_data
    
    # Add common federal funds rate
    fed_funds_rate = generate_policy_rate(date_range, seed=seed+1000)
    
    for region in regions:
        data[region]['fed_funds_rate'] = fed_funds_rate
    
    return data

def generate_ar_process(n_periods: int, phi: float = 0.7, sigma: float = 1.0, seed: int = None) -> np.ndarray:
    """Generate AR(1) process."""
    if seed is not None:
        np.random.seed(seed)
    
    y = np.zeros(n_periods)
    innovations = np.random.normal(0, sigma, n_periods)
    
    for t in range(1, n_periods):
        y[t] = phi * y[t-1] + innovations[t]
    
    return y

def generate_policy_rate(date_range: pd.DatetimeIndex, seed: int = None) -> np.ndarray:
    """Generate realistic federal funds rate path."""
    if seed is not None:
        np.random.seed(seed)
    
    n_periods = len(date_range)
    
    # Create realistic policy rate with different regimes
    rate = np.zeros(n_periods)
    
    # Pre-crisis period (higher rates)
    pre_crisis = int(0.3 * n_periods)
    rate[:pre_crisis] = 4.0 + generate_ar_process(pre_crisis, phi=0.95, sigma=0.5, seed=seed)
    rate[:pre_crisis] = np.maximum(rate[:pre_crisis], 0.25)
    
    # Crisis period (low rates)
    crisis_length = int(0.4 * n_periods)
    crisis_end = pre_crisis + crisis_length
    rate[pre_crisis:crisis_end] = 0.25 + 0.1 * np.random.normal(0, 0.1, crisis_length)
    rate[pre_crisis:crisis_end] = np.maximum(rate[pre_crisis:crisis_end], 0.0)
    
    # Recovery period (gradual increase)
    if crisis_end < n_periods:
        recovery_periods = n_periods - crisis_end
        recovery_rate = np.linspace(0.25, 3.0, recovery_periods)
        recovery_rate += 0.2 * generate_ar_process(recovery_periods, phi=0.8, sigma=0.3, seed=seed+500)
        rate[crisis_end:] = np.maximum(recovery_rate, 0.0)
    
    return rate

def generate_spatial_correlation_matrix(regions: List[str]) -> np.ndarray:
    """Generate spatial correlation matrix based on geographic proximity."""
    n_regions = len(regions)
    
    # Simple correlation structure (can be enhanced with actual geographic data)
    correlation = np.eye(n_regions)
    
    # Add some cross-regional correlation
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            # Random correlation between 0.1 and 0.6
            corr = 0.1 + 0.5 * np.random.random()
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    return correlation

def create_sample_dataset():
    """Create a complete sample dataset for testing."""
    regions = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    
    # Generate quarterly data
    regional_data = generate_synthetic_regional_data(
        regions=regions,
        start_date="2000-01-01",
        end_date="2023-12-31",
        frequency="QE"  # Fixed deprecation warning
    )
    
    # Combine into single DataFrame
    combined_data = pd.concat(regional_data.values(), ignore_index=True)
    
    return combined_data

if __name__ == "__main__":
    # Generate and save sample data
    sample_data = create_sample_dataset()
    sample_data.to_csv("data/sample_regional_data.csv", index=False)
    print(f"Generated sample data with {len(sample_data)} observations")
    print(f"Regions: {sample_data['region'].unique()}")
    print(f"Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")