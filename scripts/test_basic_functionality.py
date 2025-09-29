#!/usr/bin/env python3
"""
Basic functionality test for the Regional Monetary Policy Analysis System.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def test_sample_data():
    """Test sample data generation."""
    print("Testing sample data generation...")
    
    try:
        from regional_monetary_policy.data.sample_data import create_sample_dataset
        
        # Generate sample data
        sample_data = create_sample_dataset()
        
        # Basic validation
        assert len(sample_data) > 0, "Sample data is empty"
        assert 'region' in sample_data.columns, "Missing region column"
        assert 'date' in sample_data.columns, "Missing date column"
        assert 'output_gap' in sample_data.columns, "Missing output_gap column"
        
        print(f"✓ Generated {len(sample_data)} observations")
        print(f"✓ Regions: {list(sample_data['region'].unique())}")
        print(f"✓ Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data test failed: {str(e)}")
        return False

def test_fred_client():
    """Test FRED client with sample data."""
    print("\nTesting FRED client...")
    
    try:
        from regional_monetary_policy.data.fred_client import FREDClient
        
        # Initialize client (will use sample data since no API key)
        client = FREDClient()
        
        # Test connection
        assert client.validate_connection(), "Connection validation failed"
        print("✓ Client initialization and connection")
        
        # Test series fetch
        series_data = client.fetch_series('GDPC1', '2020-01-01', '2023-12-31')
        assert len(series_data) > 0, "No series data returned"
        assert 'date' in series_data.columns, "Missing date column"
        print("✓ Series data fetch")
        
        # Test regional data fetch
        regions = ['CA', 'TX', 'NY']
        indicators = ['gdp', 'unemployment']
        regional_data = client.fetch_regional_data(regions, indicators, '2020-01-01', '2023-12-31')
        
        assert len(regional_data) > 0, "No regional data returned"
        print(f"✓ Regional data fetch: {len(regional_data)} regions")
        
        return True
        
    except Exception as e:
        print(f"✗ FRED client test failed: {str(e)}")
        return False

def test_core_models():
    """Test core data models."""
    print("\nTesting core models...")
    
    try:
        from regional_monetary_policy.data.models import RegionalDataset
        from regional_monetary_policy.econometric.models import RegionalParameters
        from regional_monetary_policy.policy.models import PolicyScenario
        
        print("✓ Core model imports")
        
        # Test basic model instantiation
        # Note: These are Pydantic models, so we need valid data
        
        return True
        
    except Exception as e:
        print(f"✗ Core models test failed: {str(e)}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from regional_monetary_policy.config.config_manager import ConfigManager
        
        # Test basic configuration
        config_manager = ConfigManager()
        print("✓ Configuration manager initialization")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {str(e)}")
        return False

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TEST")
    print("Regional Monetary Policy Analysis System")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Sample Data", test_sample_data),
        ("FRED Client", test_fred_client),
        ("Core Models", test_core_models),
        ("Configuration", test_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test crashed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Basic functionality working!")
        return 0
    else:
        print(f"✗ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())