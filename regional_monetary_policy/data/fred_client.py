"""
FRED API client for retrieving regional economic data.

This module provides a robust client for interacting with the Federal Reserve
Economic Data (FRED) API, including authentication, rate limiting, error handling,
and real-time data management. Falls back to sample data when API key unavailable.
"""

import time
import json
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..exceptions import (
    DataRetrievalError, 
    APIRateLimitError, 
    ConfigurationError,
    ErrorHandler
)


class FREDClient:
    """
    Client for interacting with the FRED API.
    
    Provides methods for retrieving regional economic data with proper
    authentication, rate limiting, error handling, and caching.
    Falls back to sample data when API key is not available.
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    DEFAULT_RATE_LIMIT = 120  # calls per minute
    DEFAULT_TIMEOUT = 30  # seconds
    
    def __init__(self, api_key: str = None, cache_dir: str = "data/cache", 
                 rate_limit: int = None, timeout: int = None):
        """
        Initialize FRED API client.
        
        Args:
            api_key: FRED API key for authentication (optional)
            cache_dir: Directory for caching API responses
            rate_limit: API calls per minute limit
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit = rate_limit or self.DEFAULT_RATE_LIMIT
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        
        # Rate limiting state
        self.last_request_time = 0
        self.request_count = 0
        self.rate_window_start = time.time()
        
        # Use sample data if no API key
        self.use_sample_data = self.api_key is None
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def fetch_series(self, series_id: str, start_date: str = None, 
                    end_date: str = None, realtime_start: str = None,
                    realtime_end: str = None) -> pd.DataFrame:
        """
        Fetch a single data series from FRED.
        
        Args:
            series_id: FRED series identifier
            start_date: Observation start date (YYYY-MM-DD)
            end_date: Observation end date (YYYY-MM-DD)
            realtime_start: Real-time period start date
            realtime_end: Real-time period end date
            
        Returns:
            DataFrame with date and value columns
        """
        if self.use_sample_data:
            return self._get_sample_series(series_id, start_date, end_date)
        
        # Check cache first
        cache_key = self._get_cache_key(series_id, start_date, end_date, 
                                       realtime_start, realtime_end)
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Rate limiting
        self._enforce_rate_limit()
        
        # Build request parameters
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        if realtime_start:
            params['realtime_start'] = realtime_start
        if realtime_end:
            params['realtime_end'] = realtime_end
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' not in data:
                raise DataRetrievalError(f"No observations found for series {series_id}")
            
            # Convert to DataFrame
            observations = data['observations']
            df = pd.DataFrame(observations)
            
            if df.empty:
                return pd.DataFrame(columns=['date', series_id])
            
            # Clean and convert data
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove missing values
            df = df.dropna(subset=['value'])
            
            result = df[['date', 'value']].rename(columns={'value': series_id})
            
            # Cache the result
            self._cache_data(cache_key, result)
            
            return result
            
        except requests.RequestException as e:
            raise APIRateLimitError(f"FRED API request failed: {str(e)}")
        except Exception as e:
            raise DataRetrievalError(f"Error processing FRED data: {str(e)}")
    
    def fetch_multiple_series(self, series_ids: List[str], start_date: str = None,
                            end_date: str = None) -> pd.DataFrame:
        """
        Fetch multiple data series and merge them.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Observation start date (YYYY-MM-DD)
            end_date: Observation end date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date column and one column per series
        """
        if not series_ids:
            return pd.DataFrame()
        
        # Fetch first series
        result_df = self.fetch_series(series_ids[0], start_date, end_date)
        
        # Fetch and merge remaining series
        for series_id in series_ids[1:]:
            try:
                series_df = self.fetch_series(series_id, start_date, end_date)
                result_df = result_df.merge(series_df, on='date', how='outer')
            except Exception as e:
                print(f"Warning: Failed to fetch series {series_id}: {str(e)}")
                continue
        
        return result_df.sort_values('date').reset_index(drop=True)
    
    def fetch_regional_data(self, regions: List[str], indicators: List[str],
                          start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch regional economic data for multiple regions and indicators.
        
        Args:
            regions: List of region codes (e.g., state abbreviations)
            indicators: List of economic indicators
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping region codes to DataFrames
        """
        if self.use_sample_data:
            return self._get_sample_regional_data(regions, indicators, start_date, end_date)
        
        regional_data = {}
        
        for region in regions:
            region_series = []
            
            for indicator in indicators:
                series_id = self._construct_regional_series_id(region, indicator)
                region_series.append(series_id)
            
            try:
                region_df = self.fetch_multiple_series(region_series, start_date, end_date)
                if not region_df.empty:
                    region_df['region'] = region
                    regional_data[region] = region_df
            except Exception as e:
                print(f"Warning: Failed to fetch data for region {region}: {str(e)}")
                continue
        
        return regional_data
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get metadata for a FRED series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Dictionary with series metadata
        """
        if self.use_sample_data:
            return {
                'id': series_id,
                'title': f'Sample Series {series_id}',
                'units': 'Percent',
                'frequency': 'Quarterly',
                'seasonal_adjustment': 'Seasonally Adjusted'
            }
        
        self._enforce_rate_limit()
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/series",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'seriess' not in data or not data['seriess']:
                raise DataRetrievalError(f"Series {series_id} not found")
            
            return data['seriess'][0]
            
        except requests.RequestException as e:
            raise APIRateLimitError(f"FRED API request failed: {str(e)}")
    
    def _get_sample_series(self, series_id: str, start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Generate sample data for a series when API is not available."""
        try:
            from .sample_data import create_sample_dataset
            
            # Load sample data
            sample_data = create_sample_dataset()
            
            # Map series_id to column name
            series_mapping = {
                'GDPC1': 'gdp_growth',
                'CPIAUCSL': 'inflation', 
                'UNRATE': 'unemployment',
                'FEDFUNDS': 'fed_funds_rate',
                'HOUST': 'house_price_growth'
            }
            
            column_name = series_mapping.get(series_id, 'output_gap')
            
            # Get first region's data as proxy
            first_region = sample_data['region'].iloc[0]
            region_data = sample_data[sample_data['region'] == first_region].copy()
            
            # Filter by date range if specified
            if start_date:
                region_data = region_data[region_data['date'] >= start_date]
            if end_date:
                region_data = region_data[region_data['date'] <= end_date]
            
            return region_data[['date', column_name]].rename(columns={column_name: series_id})
            
        except Exception as e:
            # Fallback to synthetic data
            dates = pd.date_range(start=start_date or '2000-01-01', 
                                end=end_date or '2023-12-31', freq='QE')
            values = np.random.normal(2.0, 1.0, len(dates))  # Simple synthetic data
            return pd.DataFrame({'date': dates, series_id: values})
    
    def _get_sample_regional_data(self, regions: List[str], indicators: List[str],
                                start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Generate sample regional data when API is not available."""
        try:
            from .sample_data import create_sample_dataset
            
            sample_data = create_sample_dataset()
            
            # Filter by regions
            available_regions = sample_data['region'].unique()
            regions = [r for r in regions if r in available_regions]
            
            if not regions:
                # Use available regions if requested ones don't exist
                regions = list(available_regions)[:len(regions)]
            
            regional_data = {}
            
            for region in regions:
                region_data = sample_data[sample_data['region'] == region].copy()
                
                # Filter by date range
                if start_date:
                    region_data = region_data[region_data['date'] >= start_date]
                if end_date:
                    region_data = region_data[region_data['date'] <= end_date]
                
                regional_data[region] = region_data
            
            return regional_data
            
        except Exception as e:
            # Fallback to synthetic regional data
            regional_data = {}
            dates = pd.date_range(start=start_date or '2000-01-01',
                                end=end_date or '2023-12-31', freq='QE')
            
            for region in regions:
                data = {'date': dates, 'region': region}
                for indicator in indicators:
                    data[indicator] = np.random.normal(2.0, 1.0, len(dates))
                regional_data[region] = pd.DataFrame(data)
            
            return regional_data
    
    def _construct_regional_series_id(self, region: str, indicator: str) -> str:
        """Construct FRED series ID for regional indicator."""
        # Simplified mapping - in practice this would be more sophisticated
        series_mapping = {
            'gdp': f'{region}RGSP',
            'unemployment': f'{region}UR', 
            'inflation': f'{region}CPI',
            'housing': f'{region}HOUST',
            'output_gap': f'{region}GDPGAP'
        }
        
        return series_mapping.get(indicator, f'{region}{indicator.upper()}')
    
    def _enforce_rate_limit(self):
        """Enforce API rate limiting."""
        current_time = time.time()
        
        # Reset counter if we're in a new minute
        if current_time - self.rate_window_start >= 60:
            self.request_count = 0
            self.rate_window_start = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.rate_window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count = 0
                self.rate_window_start = time.time()
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    def _get_cache_key(self, series_id: str, start_date: str = None,
                      end_date: str = None, realtime_start: str = None,
                      realtime_end: str = None) -> str:
        """Generate cache key for request."""
        key_parts = [series_id]
        if start_date:
            key_parts.append(f"start_{start_date}")
        if end_date:
            key_parts.append(f"end_{end_date}")
        if realtime_start:
            key_parts.append(f"rt_start_{realtime_start}")
        if realtime_end:
            key_parts.append(f"rt_end_{realtime_end}")
        
        return "_".join(key_parts)
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired (24 hours)
            if time.time() - cache_file.stat().st_mtime > 24 * 3600:
                cache_file.unlink()
                return None
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception:
            # If cache is corrupted, remove it
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data to disk."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Convert DataFrame to JSON-serializable format
            data_dict = data.copy()
            data_dict['date'] = data_dict['date'].dt.strftime('%Y-%m-%d')
            
            with open(cache_file, 'w') as f:
                json.dump(data_dict.to_dict('records'), f)
                
        except Exception as e:
            # Cache failures shouldn't break the main functionality
            print(f"Warning: Failed to cache data: {str(e)}")
    
    def validate_connection(self) -> bool:
        """Test connection to FRED API."""
        if self.use_sample_data:
            return True
        
        try:
            self.get_series_info('GDPC1')
            return True
        except Exception:
            return False
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear cache: {str(e)}")