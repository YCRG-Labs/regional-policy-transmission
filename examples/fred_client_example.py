"""
Example usage of FRED API client and data management system.

This script demonstrates how to use the FREDClient and DataManager
to retrieve and process regional economic data.
"""

import os
from pathlib import Path
from regional_monetary_policy.data import FREDClient, DataManager

def main():
    """Demonstrate FRED client and data manager usage."""
    
    # Load API key from environment
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("Please set FRED_API_KEY environment variable")
        return
    
    # Create cache directory
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing FRED API client...")
    
    # Initialize FRED client
    fred_client = FREDClient(
        api_key=api_key,
        cache_dir=str(cache_dir),
        rate_limit=120,  # FRED allows 120 calls per minute
        timeout=30
    )
    
    print("✓ FRED client initialized")
    
    # Initialize data manager
    data_manager = DataManager(fred_client, cache_strategy="intelligent")
    print("✓ Data manager initialized")
    
    # Example 1: Get series metadata
    print("\n1. Retrieving series metadata...")
    try:
        gdp_metadata = fred_client.get_series_metadata('GDP')
        print(f"   Series: {gdp_metadata['id']}")
        print(f"   Title: {gdp_metadata['title']}")
        print(f"   Units: {gdp_metadata['units']}")
        print(f"   Frequency: {gdp_metadata['frequency']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 2: Retrieve economic data
    print("\n2. Retrieving economic data...")
    try:
        series_codes = ['GDP', 'CPIAUCSL', 'FEDFUNDS']
        data = fred_client.get_regional_series(
            series_codes, 
            start_date='2020-01-01', 
            end_date='2023-12-31'
        )
        print(f"   Retrieved data shape: {data.shape}")
        print(f"   Series: {list(data.columns)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Sample data:\n{data.head()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 3: Search for series
    print("\n3. Searching for regional GDP series...")
    try:
        search_results = fred_client.search_series("real gdp state", limit=5)
        print(f"   Found {len(search_results)} series:")
        for series in search_results[:3]:  # Show first 3
            print(f"   - {series['id']}: {series['title']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 4: Cache statistics
    print("\n4. Cache statistics...")
    try:
        cache_stats = fred_client.get_cache_stats()
        print(f"   Total cache entries: {cache_stats['total_entries']}")
        print(f"   Valid entries: {cache_stats['valid_entries']}")
        print(f"   Cache size: {cache_stats['cache_size_mb']:.2f} MB")
        
        data_cache_info = data_manager.get_cache_info()
        print(f"   Cached datasets: {data_cache_info['cached_datasets']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 5: Data validation
    print("\n5. Data validation example...")
    try:
        # This would normally use real regional data
        print("   (Data validation requires regional dataset - see data_manager.load_regional_data())")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✓ Examples completed successfully!")
    print(f"\nCache directory: {cache_dir.absolute()}")
    print("Check the cache directory for stored API responses and datasets.")

if __name__ == "__main__":
    main()