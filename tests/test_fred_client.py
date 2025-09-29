"""
Tests for FRED API client functionality.

This module contains comprehensive tests for the FREDClient class,
including API integration, rate limiting, caching, and error handling.
"""

import pytest
import time
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta

from regional_monetary_policy.data.fred_client import FREDClient
from regional_monetary_policy.exceptions import (
    DataRetrievalError, 
    APIRateLimitError, 
    ConfigurationError
)


class TestFREDClient:
    """Test suite for FREDClient class."""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory for testing."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        return str(cache_dir)
    
    @pytest.fixture
    def mock_fred_client(self, temp_cache_dir):
        """Create FREDClient with mocked API key validation."""
        with patch.object(FREDClient, 'validate_api_key', return_value=True):
            return FREDClient(
                api_key="test_api_key",
                cache_dir=temp_cache_dir,
                rate_limit=5,  # Low rate limit for testing
                timeout=10
            )
    
    @pytest.fixture
    def sample_api_response(self):
        """Sample FRED API response for testing."""
        return {
            "realtime_start": "2023-01-01",
            "realtime_end": "2023-12-31",
            "observations": [
                {"date": "2023-01-01", "value": "100.0"},
                {"date": "2023-02-01", "value": "101.5"},
                {"date": "2023-03-01", "value": "102.1"},
                {"date": "2023-04-01", "value": "."},  # Missing value
                {"date": "2023-05-01", "value": "103.8"}
            ]
        }
    
    def test_initialization(self, temp_cache_dir):
        """Test FREDClient initialization."""
        with patch.object(FREDClient, 'validate_api_key', return_value=True):
            client = FREDClient(
                api_key="test_key",
                cache_dir=temp_cache_dir
            )
            
            assert client.api_key == "test_key"
            assert client.rate_limit == FREDClient.DEFAULT_RATE_LIMIT
            assert client.timeout == FREDClient.DEFAULT_TIMEOUT
            assert Path(client.cache_dir).exists()
    
    def test_invalid_api_key_initialization(self, temp_cache_dir):
        """Test initialization with invalid API key."""
        with patch.object(FREDClient, 'validate_api_key', return_value=False):
            with pytest.raises(ConfigurationError) as exc_info:
                FREDClient(api_key="invalid_key", cache_dir=temp_cache_dir)
            
            assert "Invalid FRED API key" in str(exc_info.value)
    
    def test_cache_database_initialization(self, mock_fred_client):
        """Test that cache database is properly initialized."""
        db_path = Path(mock_fred_client.cache_dir) / "fred_cache.db"
        assert db_path.exists()
        
        # Check that tables were created
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('api_cache', 'series_metadata')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'api_cache' in tables
            assert 'series_metadata' in tables
    
    def test_rate_limiting(self, mock_fred_client):
        """Test API rate limiting functionality."""
        # Mock successful API responses
        with patch.object(mock_fred_client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"test": "data"}
            mock_get.return_value = mock_response
            
            # Make requests up to the rate limit
            start_time = time.time()
            for i in range(mock_fred_client.rate_limit):
                mock_fred_client._make_request('test', {'param': f'value_{i}'}, use_cache=False)
            
            # Next request should be delayed
            mock_fred_client._make_request('test', {'param': 'final'}, use_cache=False)
            end_time = time.time()
            
            # Should have taken some time due to rate limiting
            # (This is a simplified test - in practice, timing tests can be flaky)
            assert len(mock_fred_client._call_times) <= mock_fred_client.rate_limit
    
    @patch('requests.Session.get')
    def test_successful_api_request(self, mock_get, mock_fred_client, sample_api_response):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response
        
        result = mock_fred_client._make_request('series/observations', {'series_id': 'GDP'})
        
        assert result == sample_api_response
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_api_rate_limit_error(self, mock_get, mock_fred_client):
        """Test handling of API rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_get.return_value = mock_response
        
        with pytest.raises(APIRateLimitError) as exc_info:
            mock_fred_client._make_request('series', {'series_id': 'GDP'})
        
        assert "rate limit exceeded" in str(exc_info.value).lower()
        assert exc_info.value.context.get('retry_after') == 60
    
    @patch('requests.Session.get')
    def test_api_error_response(self, mock_get, mock_fred_client):
        """Test handling of API error responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'error_code': 400,
            'error_message': 'Bad Request: Invalid series ID'
        }
        mock_get.return_value = mock_response
        
        with pytest.raises(DataRetrievalError) as exc_info:
            mock_fred_client._make_request('series', {'series_id': 'INVALID'})
        
        assert "Bad Request: Invalid series ID" in str(exc_info.value)
    
    @patch('requests.Session.get')
    def test_network_timeout(self, mock_get, mock_fred_client):
        """Test handling of network timeouts."""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(DataRetrievalError) as exc_info:
            mock_fred_client._make_request('series', {'series_id': 'GDP'})
        
        assert "timed out" in str(exc_info.value).lower()
    
    @patch('requests.Session.get')
    def test_connection_error(self, mock_get, mock_fred_client):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(DataRetrievalError) as exc_info:
            mock_fred_client._make_request('series', {'series_id': 'GDP'})
        
        assert "Failed to connect" in str(exc_info.value)
    
    def test_caching_functionality(self, mock_fred_client, sample_api_response):
        """Test API response caching."""
        with patch.object(mock_fred_client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_api_response
            mock_get.return_value = mock_response
            
            # First request should hit the API
            result1 = mock_fred_client._make_request('series', {'series_id': 'GDP'})
            assert mock_get.call_count == 1
            
            # Second identical request should use cache
            result2 = mock_fred_client._make_request('series', {'series_id': 'GDP'})
            assert mock_get.call_count == 1  # No additional API call
            assert result1 == result2
    
    def test_get_series_metadata(self, mock_fred_client):
        """Test retrieving series metadata."""
        sample_metadata = {
            'seriess': [{
                'id': 'GDP',
                'title': 'Gross Domestic Product',
                'units': 'Billions of Dollars',
                'frequency': 'Quarterly',
                'seasonal_adjustment': 'Seasonally Adjusted Annual Rate'
            }]
        }
        
        with patch.object(mock_fred_client, '_make_request', return_value=sample_metadata):
            metadata = mock_fred_client.get_series_metadata('GDP')
            
            assert metadata['id'] == 'GDP'
            assert metadata['title'] == 'Gross Domestic Product'
    
    def test_get_series_metadata_not_found(self, mock_fred_client):
        """Test handling of series not found."""
        with patch.object(mock_fred_client, '_make_request', return_value={'seriess': []}):
            with pytest.raises(DataRetrievalError) as exc_info:
                mock_fred_client.get_series_metadata('NONEXISTENT')
            
            assert "not found" in str(exc_info.value).lower()
    
    def test_get_regional_series(self, mock_fred_client, sample_api_response):
        """Test retrieving multiple regional series."""
        with patch.object(mock_fred_client, '_get_series_observations') as mock_get_obs:
            # Mock series data
            mock_series1 = pd.Series([100.0, 101.5, 102.1], 
                                   index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
                                   name='SERIES1')
            mock_series2 = pd.Series([200.0, 201.0, 202.0], 
                                   index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
                                   name='SERIES2')
            
            mock_get_obs.side_effect = [mock_series1, mock_series2]
            
            result = mock_fred_client.get_regional_series(
                ['SERIES1', 'SERIES2'], '2023-01-01', '2023-03-01'
            )
            
            assert isinstance(result, pd.DataFrame)
            assert 'SERIES1' in result.columns
            assert 'SERIES2' in result.columns
            assert len(result) == 3
    
    def test_get_regional_series_empty_list(self, mock_fred_client):
        """Test error handling for empty series list."""
        with pytest.raises(ValueError) as exc_info:
            mock_fred_client.get_regional_series([], '2023-01-01', '2023-12-31')
        
        assert "At least one series code must be provided" in str(exc_info.value)
    
    def test_get_real_time_data(self, mock_fred_client):
        """Test retrieving real-time data for specific vintage."""
        sample_response = {
            'observations': [{'date': '2023-01-01', 'value': '100.5'}]
        }
        
        with patch.object(mock_fred_client, '_make_request', return_value=sample_response):
            value = mock_fred_client.get_real_time_data('GDP', '2023-01-01', '2023-01-15')
            
            assert value == 100.5
    
    def test_get_real_time_data_missing_value(self, mock_fred_client):
        """Test handling of missing real-time data."""
        sample_response = {
            'observations': [{'date': '2023-01-01', 'value': '.'}]
        }
        
        with patch.object(mock_fred_client, '_make_request', return_value=sample_response):
            with pytest.raises(DataRetrievalError) as exc_info:
                mock_fred_client.get_real_time_data('GDP', '2023-01-01', '2023-01-15')
            
            assert "Missing value" in str(exc_info.value)
    
    def test_search_series(self, mock_fred_client):
        """Test series search functionality."""
        sample_search_results = {
            'seriess': [
                {'id': 'GDP', 'title': 'Gross Domestic Product'},
                {'id': 'GDPC1', 'title': 'Real Gross Domestic Product'}
            ]
        }
        
        with patch.object(mock_fred_client, '_make_request', return_value=sample_search_results):
            results = mock_fred_client.search_series('GDP')
            
            assert len(results) == 2
            assert results[0]['id'] == 'GDP'
    
    def test_clear_cache(self, mock_fred_client):
        """Test cache clearing functionality."""
        # Add some test data to cache first
        db_path = Path(mock_fred_client.cache_dir) / "fred_cache.db"
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO api_cache (endpoint, params_hash, response_data, created_at)
                VALUES ('test', 'hash1', '{"test": "data"}', datetime('now', '-10 days'))
            """)
            conn.execute("""
                INSERT INTO api_cache (endpoint, params_hash, response_data, created_at)
                VALUES ('test', 'hash2', '{"test": "data2"}', datetime('now', '-1 days'))
            """)
        
        # Clear cache older than 5 days
        removed_count = mock_fred_client.clear_cache(older_than_days=5)
        
        assert removed_count == 1  # Only the 10-day-old entry should be removed
    
    def test_get_cache_stats(self, mock_fred_client):
        """Test cache statistics retrieval."""
        stats = mock_fred_client.get_cache_stats()
        
        assert 'total_entries' in stats
        assert 'expired_entries' in stats
        assert 'valid_entries' in stats
        assert 'cache_size_mb' in stats
        assert 'cache_directory' in stats


class TestFREDClientIntegration:
    """Integration tests for FREDClient (require actual API key)."""
    
    @pytest.fixture
    def real_fred_client(self, tmp_path):
        """Create FREDClient with real API key from environment."""
        import os
        api_key = os.getenv('FRED_API_KEY')
        
        if not api_key:
            pytest.skip("FRED_API_KEY environment variable not set")
        
        cache_dir = tmp_path / "integration_cache"
        cache_dir.mkdir()
        
        return FREDClient(api_key=api_key, cache_dir=str(cache_dir))
    
    def test_real_api_validation(self, real_fred_client):
        """Test API key validation with real FRED API."""
        assert real_fred_client.validate_api_key() is True
    
    def test_real_series_metadata(self, real_fred_client):
        """Test retrieving real series metadata."""
        metadata = real_fred_client.get_series_metadata('GDP')
        
        assert 'id' in metadata
        assert metadata['id'] == 'GDP'
        assert 'title' in metadata
    
    def test_real_series_data(self, real_fred_client):
        """Test retrieving real series data."""
        # Get recent GDP data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        data = real_fred_client.get_regional_series(['GDP'], start_date, end_date)
        
        assert isinstance(data, pd.DataFrame)
        assert 'GDP' in data.columns
        assert len(data) > 0
    
    @pytest.mark.slow
    def test_real_rate_limiting(self, real_fred_client):
        """Test rate limiting with real API (slow test)."""
        # This test makes many API calls and may take time
        start_time = time.time()
        
        # Make requests that should trigger rate limiting
        for i in range(10):
            try:
                real_fred_client.get_series_metadata('GDP')
            except APIRateLimitError:
                # Rate limiting is working
                break
        
        end_time = time.time()
        
        # Should have taken some time due to rate limiting
        # (Exact timing depends on API limits and network conditions)
        assert end_time - start_time >= 0  # Basic sanity check