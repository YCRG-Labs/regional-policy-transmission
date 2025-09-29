"""
Tests for DataManager functionality.

This module contains comprehensive tests for the DataManager class,
including data loading, caching, validation, and preprocessing.
"""

import pytest
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from regional_monetary_policy.data.data_manager import DataManager
from regional_monetary_policy.data.fred_client import FREDClient
from regional_monetary_policy.data.models import RegionalDataset, ValidationReport
from regional_monetary_policy.exceptions import (
    DataRetrievalError, 
    DataValidationError, 
    InsufficientDataError
)


class TestDataManager:
    """Test suite for DataManager class."""
    
    @pytest.fixture
    def mock_fred_client(self, tmp_path):
        """Create mock FREDClient for testing."""
        mock_client = Mock(spec=FREDClient)
        mock_client.cache_dir = tmp_path / "test_cache"
        mock_client.cache_dir.mkdir()
        return mock_client
    
    @pytest.fixture
    def data_manager(self, mock_fred_client):
        """Create DataManager instance for testing."""
        return DataManager(mock_fred_client, cache_strategy="intelligent")
    
    @pytest.fixture
    def sample_regional_dataset(self):
        """Create sample RegionalDataset for testing."""
        dates = pd.date_range('2020-01-01', '2022-12-01', freq='M')
        regions = ['CA', 'TX', 'NY']
        
        # Create sample data with some realistic patterns
        np.random.seed(42)  # For reproducible tests
        
        output_gaps = pd.DataFrame(
            np.random.normal(0, 2, (len(regions), len(dates))),
            index=regions,
            columns=dates
        )
        
        inflation_rates = pd.DataFrame(
            np.random.normal(2.5, 1.5, (len(regions), len(dates))),
            index=regions,
            columns=dates
        )
        
        interest_rates = pd.Series(
            np.random.uniform(0.5, 5.0, len(dates)),
            index=dates,
            name='interest_rate'
        )
        
        # Add some missing values
        output_gaps.iloc[0, 5:8] = np.nan
        inflation_rates.iloc[1, 10:12] = np.nan
        
        metadata = {
            'source': 'FRED',
            'retrieved_at': datetime.now().isoformat(),
            'regions': regions,
            'indicators': ['output_gap', 'inflation', 'interest_rate'],
            'start_date': str(dates[0]),
            'end_date': str(dates[-1]),
            'n_periods': len(dates),
            'frequency': 'monthly'
        }
        
        return RegionalDataset(
            output_gaps=output_gaps,
            inflation_rates=inflation_rates,
            interest_rates=interest_rates,
            real_time_estimates={},
            metadata=metadata
        )
    
    @pytest.fixture
    def sample_fred_data(self):
        """Create sample FRED API response data."""
        dates = pd.date_range('2020-01-01', '2022-12-01', freq='M')
        
        # Mock GDP data for output gap calculation
        gdp_data = pd.Series(
            np.random.uniform(20000, 25000, len(dates)),
            index=dates,
            name='GDP'
        )
        
        # Mock CPI data for inflation calculation
        cpi_data = pd.Series(
            np.cumprod(1 + np.random.normal(0.002, 0.01, len(dates))) * 250,
            index=dates,
            name='CPIAUCSL'
        )
        
        # Mock interest rate data
        fed_funds = pd.Series(
            np.random.uniform(0.5, 5.0, len(dates)),
            index=dates,
            name='FEDFUNDS'
        )
        
        return pd.DataFrame({
            'GDP': gdp_data,
            'CPIAUCSL': cpi_data,
            'FEDFUNDS': fed_funds
        })
    
    def test_initialization(self, mock_fred_client):
        """Test DataManager initialization."""
        manager = DataManager(mock_fred_client, cache_strategy="aggressive")
        
        assert manager.fred_client == mock_fred_client
        assert manager.cache_strategy == "aggressive"
        assert manager.cache_dir == mock_fred_client.cache_dir
        
        # Check that cache database was initialized
        db_path = manager.cache_dir / "data_cache.db"
        assert db_path.exists()
    
    def test_cache_database_initialization(self, data_manager):
        """Test that cache database tables are properly created."""
        db_path = data_manager.cache_dir / "data_cache.db"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('regional_datasets', 'validation_reports')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'regional_datasets' in tables
            assert 'validation_reports' in tables
    
    def test_generate_dataset_hash(self, data_manager):
        """Test dataset hash generation for caching."""
        hash1 = data_manager._generate_dataset_hash(
            ['CA', 'TX'], ['output_gap', 'inflation'], '2020-01-01', '2022-12-31'
        )
        
        hash2 = data_manager._generate_dataset_hash(
            ['TX', 'CA'], ['inflation', 'output_gap'], '2020-01-01', '2022-12-31'
        )
        
        # Same parameters in different order should produce same hash
        assert hash1 == hash2
        
        hash3 = data_manager._generate_dataset_hash(
            ['CA', 'TX'], ['output_gap', 'inflation'], '2020-01-01', '2021-12-31'
        )
        
        # Different parameters should produce different hash
        assert hash1 != hash3
    
    def test_serialize_deserialize_dataset(self, data_manager, sample_regional_dataset):
        """Test dataset serialization and deserialization for caching."""
        # Serialize
        data_dict, metadata = data_manager._serialize_dataset(sample_regional_dataset)
        
        assert 'output_gaps' in data_dict
        assert 'inflation_rates' in data_dict
        assert 'interest_rates' in data_dict
        assert 'real_time_estimates' in data_dict
        
        # Deserialize
        restored_dataset = data_manager._deserialize_dataset(data_dict, metadata)
        
        # Check that data is preserved
        pd.testing.assert_frame_equal(
            sample_regional_dataset.output_gaps, 
            restored_dataset.output_gaps
        )
        pd.testing.assert_frame_equal(
            sample_regional_dataset.inflation_rates, 
            restored_dataset.inflation_rates
        )
        pd.testing.assert_series_equal(
            sample_regional_dataset.interest_rates, 
            restored_dataset.interest_rates
        )
        assert sample_regional_dataset.metadata == restored_dataset.metadata
    
    def test_cache_and_load_dataset(self, data_manager, sample_regional_dataset):
        """Test caching and loading of datasets."""
        dataset_hash = "test_hash_123"
        
        # Cache the dataset
        data_manager._cache_dataset(dataset_hash, sample_regional_dataset)
        
        # Load from cache
        loaded_dataset = data_manager._load_from_cache(dataset_hash)
        
        assert loaded_dataset is not None
        assert loaded_dataset.regions == sample_regional_dataset.regions
        assert loaded_dataset.n_periods == sample_regional_dataset.n_periods
    
    def test_load_from_cache_not_found(self, data_manager):
        """Test loading from cache when dataset not found."""
        result = data_manager._load_from_cache("nonexistent_hash")
        assert result is None
    
    @patch.object(DataManager, '_load_fresh_data')
    def test_load_regional_data_with_cache(self, mock_load_fresh, data_manager, sample_regional_dataset):
        """Test loading regional data with caching enabled."""
        # Setup mock
        mock_load_fresh.return_value = sample_regional_dataset
        
        # First call should load fresh data
        result1 = data_manager.load_regional_data(
            regions=['CA', 'TX'],
            indicators=['output_gap', 'inflation'],
            start_date='2020-01-01',
            end_date='2022-12-31',
            use_cache=True
        )
        
        assert mock_load_fresh.call_count == 1
        assert result1.regions == ['CA', 'TX']
        
        # Second identical call should use cache
        result2 = data_manager.load_regional_data(
            regions=['CA', 'TX'],
            indicators=['output_gap', 'inflation'],
            start_date='2020-01-01',
            end_date='2022-12-31',
            use_cache=True
        )
        
        # Should not call load_fresh_data again
        assert mock_load_fresh.call_count == 1
        assert result2.regions == result1.regions
    
    def test_load_regional_data_without_cache(self, data_manager, sample_regional_dataset):
        """Test loading regional data without caching."""
        with patch.object(data_manager, '_load_fresh_data', return_value=sample_regional_dataset):
            result = data_manager.load_regional_data(
                regions=['CA', 'TX'],
                indicators=['output_gap', 'inflation'],
                start_date='2020-01-01',
                end_date='2022-12-31',
                use_cache=False
            )
            
            assert result.regions == ['CA', 'TX']
    
    def test_load_fresh_data(self, data_manager, sample_fred_data):
        """Test loading fresh data from FRED API."""
        # Mock FRED client methods
        data_manager.fred_client.get_regional_series.return_value = sample_fred_data
        
        with patch.object(data_manager, '_get_series_mapping') as mock_mapping:
            mock_mapping.return_value = {
                'output_gap': ['GDP'],
                'inflation': ['CPIAUCSL'],
                'interest_rate': ['FEDFUNDS']
            }
            
            result = data_manager._load_fresh_data(
                regions=['US'],
                indicators=['output_gap', 'inflation', 'interest_rate'],
                start_date='2020-01-01',
                end_date='2022-12-31'
            )
            
            assert isinstance(result, RegionalDataset)
            assert result.metadata['source'] == 'FRED'
    
    def test_get_series_mapping(self, data_manager):
        """Test series mapping generation."""
        mapping = data_manager._get_series_mapping(
            regions=['CA', 'TX'],
            indicators=['output_gap', 'inflation', 'interest_rate']
        )
        
        assert 'output_gap' in mapping
        assert 'inflation' in mapping
        assert 'interest_rate' in mapping
        
        # Interest rate should be national (same for all regions)
        assert len(mapping['interest_rate']) > 0
    
    def test_calculate_output_gaps(self, data_manager, sample_fred_data):
        """Test output gap calculation from GDP data."""
        output_gaps = data_manager._calculate_output_gaps(
            sample_fred_data, ['US'], ['GDP']
        )
        
        assert 'US' in output_gaps
        assert isinstance(output_gaps['US'], pd.Series)
        assert len(output_gaps['US']) > 0
    
    def test_calculate_inflation_rates(self, data_manager, sample_fred_data):
        """Test inflation rate calculation from CPI data."""
        inflation_rates = data_manager._calculate_inflation_rates(
            sample_fred_data, ['US'], ['CPIAUCSL']
        )
        
        assert 'US' in inflation_rates
        assert isinstance(inflation_rates['US'], pd.Series)
        assert len(inflation_rates['US']) > 0
    
    def test_extract_interest_rates(self, data_manager, sample_fred_data):
        """Test interest rate extraction."""
        interest_rates = data_manager._extract_interest_rates(
            sample_fred_data, ['FEDFUNDS']
        )
        
        assert isinstance(interest_rates, pd.Series)
        assert len(interest_rates) > 0
    
    def test_infer_frequency(self, data_manager):
        """Test frequency inference from time index."""
        # Monthly data
        monthly_index = pd.date_range('2020-01-01', '2020-12-01', freq='M')
        assert data_manager._infer_frequency(monthly_index) == 'monthly'
        
        # Quarterly data
        quarterly_index = pd.date_range('2020-01-01', '2020-12-01', freq='Q')
        assert data_manager._infer_frequency(quarterly_index) == 'quarterly'
        
        # Daily data
        daily_index = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        assert data_manager._infer_frequency(daily_index) == 'daily'


class TestDataValidation:
    """Test suite for data validation functionality."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for validation testing."""
        mock_client = Mock()
        mock_client.cache_dir = tmp_path / "cache"
        mock_client.cache_dir.mkdir()
        return DataManager(mock_client)
    
    @pytest.fixture
    def good_dataset(self):
        """Create dataset with good quality data."""
        dates = pd.date_range('2020-01-01', '2022-12-01', freq='M')
        regions = ['CA', 'TX', 'NY']
        
        output_gaps = pd.DataFrame(
            np.random.normal(0, 1.5, (len(regions), len(dates))),
            index=regions, columns=dates
        )
        
        inflation_rates = pd.DataFrame(
            np.random.normal(2.0, 0.8, (len(regions), len(dates))),
            index=regions, columns=dates
        )
        
        interest_rates = pd.Series(
            np.random.uniform(1.0, 4.0, len(dates)),
            index=dates
        )
        
        return RegionalDataset(
            output_gaps=output_gaps,
            inflation_rates=inflation_rates,
            interest_rates=interest_rates,
            real_time_estimates={},
            metadata={'source': 'test'}
        )
    
    @pytest.fixture
    def bad_dataset(self):
        """Create dataset with quality issues."""
        dates = pd.date_range('2020-01-01', '2020-06-01', freq='M')  # Too short
        regions = ['CA', 'TX']
        
        # Data with many missing values and outliers
        output_gaps = pd.DataFrame(
            np.random.normal(0, 1.5, (len(regions), len(dates))),
            index=regions, columns=dates
        )
        output_gaps.iloc[:, :3] = np.nan  # Many missing values
        output_gaps.iloc[0, -1] = 50  # Extreme outlier
        
        inflation_rates = pd.DataFrame(
            np.random.normal(2.0, 0.8, (len(regions), len(dates))),
            index=regions, columns=dates
        )
        inflation_rates.iloc[1, :2] = np.nan  # Missing values
        inflation_rates.iloc[0, -1] = -15  # Unrealistic inflation
        
        interest_rates = pd.Series(
            np.random.uniform(1.0, 4.0, len(dates)),
            index=dates
        )
        
        return RegionalDataset(
            output_gaps=output_gaps,
            inflation_rates=inflation_rates,
            interest_rates=interest_rates,
            real_time_estimates={},
            metadata={'source': 'test'}
        )
    
    def test_validate_good_data(self, data_manager, good_dataset):
        """Test validation of good quality data."""
        report = data_manager.validate_data_quality(good_dataset)
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.data_quality_score > 70
        assert len(report.warnings) == 0
    
    def test_validate_bad_data(self, data_manager, bad_dataset):
        """Test validation of poor quality data."""
        report = data_manager.validate_data_quality(bad_dataset)
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert report.data_quality_score < 70
        assert len(report.warnings) > 0
        assert len(report.recommendations) > 0
    
    def test_validate_single_series(self, data_manager):
        """Test validation of individual series."""
        # Good series
        good_series = pd.Series(np.random.normal(2.0, 0.5, 100))
        issues = data_manager._validate_single_series(good_series, 'test_series', (-5, 10))
        
        assert issues['missing']['test_series'] == 0
        assert issues['outliers']['test_series'] == 0
        assert len(issues['warnings']) == 0
        
        # Bad series with missing values and outliers
        bad_series = pd.Series([1, 2, np.nan, np.nan, 100, -50, 3, 4])
        issues = data_manager._validate_single_series(bad_series, 'bad_series', (-5, 10))
        
        assert issues['missing']['bad_series'] == 2
        assert len(issues['warnings']) > 0
    
    def test_check_data_gaps(self, data_manager):
        """Test data gap detection."""
        # Create dataset with gaps
        dates = pd.date_range('2020-01-01', '2020-12-01', freq='M')
        # Remove some dates to create gaps
        dates_with_gaps = dates.delete([3, 4, 8])  # Remove April, May, September
        
        dataset = RegionalDataset(
            output_gaps=pd.DataFrame(index=['CA'], columns=dates_with_gaps),
            inflation_rates=pd.DataFrame(index=['CA'], columns=dates_with_gaps),
            interest_rates=pd.Series(index=dates_with_gaps),
            real_time_estimates={},
            metadata={}
        )
        
        issues = data_manager._check_data_gaps(dataset)
        
        # Should detect gaps
        assert len(issues['warnings']) > 0
        assert any('gap' in warning.lower() for warning in issues['warnings'])
    
    def test_check_data_sufficiency(self, data_manager):
        """Test data sufficiency checking."""
        # Insufficient data
        short_dates = pd.date_range('2020-01-01', '2020-06-01', freq='M')
        insufficient_dataset = RegionalDataset(
            output_gaps=pd.DataFrame(index=['CA'], columns=short_dates),
            inflation_rates=pd.DataFrame(index=['CA'], columns=short_dates),
            interest_rates=pd.Series(index=short_dates),
            real_time_estimates={},
            metadata={}
        )
        
        issues = data_manager._check_data_sufficiency(insufficient_dataset)
        
        assert len(issues['warnings']) > 0
        assert any('insufficient' in warning.lower() for warning in issues['warnings'])
    
    def test_calculate_quality_score(self, data_manager):
        """Test quality score calculation."""
        # Perfect data
        score1 = data_manager._calculate_quality_score({}, {}, 0)
        assert score1 == 100.0
        
        # Data with issues
        score2 = data_manager._calculate_quality_score(
            {'series1': 10}, {'series2': 5}, 3
        )
        assert score2 < 100.0
        assert score2 >= 0
    
    def test_handle_missing_data(self, data_manager):
        """Test missing data handling methods."""
        # Create data with missing values
        data = pd.DataFrame({
            'A': [1, np.nan, 3, 4, np.nan],
            'B': [np.nan, 2, 3, np.nan, 5]
        })
        
        # Test interpolation
        interpolated = data_manager.handle_missing_data(data, method='interpolation')
        assert interpolated.isna().sum().sum() < data.isna().sum().sum()
        
        # Test forward fill
        ffilled = data_manager.handle_missing_data(data, method='forward_fill')
        assert ffilled.isna().sum().sum() <= data.isna().sum().sum()
        
        # Test drop
        dropped = data_manager.handle_missing_data(data, method='drop')
        assert dropped.isna().sum().sum() == 0
        assert len(dropped.columns) <= len(data.columns)
    
    def test_handle_missing_data_invalid_method(self, data_manager):
        """Test error handling for invalid missing data method."""
        data = pd.DataFrame({'A': [1, np.nan, 3]})
        
        with pytest.raises(ValueError) as exc_info:
            data_manager.handle_missing_data(data, method='invalid_method')
        
        assert "Unknown missing data method" in str(exc_info.value)


class TestDataManagerUtilities:
    """Test suite for DataManager utility functions."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for utility testing."""
        mock_client = Mock()
        mock_client.cache_dir = tmp_path / "cache"
        mock_client.cache_dir.mkdir()
        return DataManager(mock_client)
    
    def test_get_vintage_data(self, data_manager):
        """Test vintage data retrieval."""
        # Mock FRED client response
        data_manager.fred_client.get_real_time_data.side_effect = [100.5, 2.3, 1.75]
        
        vintage_data = data_manager.get_vintage_data('2023-01-01', '2023-01-15')
        
        assert isinstance(vintage_data, pd.DataFrame)
        assert len(vintage_data) == 1
        assert vintage_data.index[0] == '2023-01-01'
    
    def test_clear_cache(self, data_manager):
        """Test cache clearing functionality."""
        # Add test data to cache
        db_path = data_manager.cache_dir / "data_cache.db"
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO regional_datasets 
                (dataset_hash, regions, indicators, start_date, end_date, 
                 data_json, metadata_json, created_at)
                VALUES ('hash1', '["CA"]', '["output_gap"]', '2020-01-01', '2020-12-31',
                        '{}', '{}', datetime('now', '-10 days'))
            """)
            conn.execute("""
                INSERT INTO regional_datasets 
                (dataset_hash, regions, indicators, start_date, end_date, 
                 data_json, metadata_json, created_at)
                VALUES ('hash2', '["TX"]', '["inflation"]', '2021-01-01', '2021-12-31',
                        '{}', '{}', datetime('now', '-1 days'))
            """)
        
        # Clear old cache entries
        removed = data_manager.clear_cache(older_than_days=5)
        
        assert removed['datasets'] == 1  # Only the 10-day-old entry
        assert removed['validations'] >= 0
    
    def test_get_cache_info(self, data_manager):
        """Test cache information retrieval."""
        info = data_manager.get_cache_info()
        
        assert 'cached_datasets' in info
        assert 'validation_reports' in info
        assert 'cache_size_mb' in info
        assert 'cache_strategy' in info
        assert 'cache_directory' in info
        assert info['cache_strategy'] == 'intelligent'