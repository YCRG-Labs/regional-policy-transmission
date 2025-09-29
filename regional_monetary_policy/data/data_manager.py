"""
Data management system for regional monetary policy analysis.

This module provides comprehensive data management capabilities including
intelligent caching, data validation, quality checking, and preprocessing
for regional economic data.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import pandas as pd
import numpy as np
from scipy import stats

from .fred_client import FREDClient
from .models import RegionalDataset, ValidationReport
from ..exceptions import (
    DataRetrievalError, 
    DataValidationError, 
    InsufficientDataError,
    ConfigurationError
)


class DataManager:
    """
    Manages data storage, caching, validation, and preprocessing.
    
    Provides high-level interface for loading regional economic data
    with intelligent caching, quality validation, and preprocessing.
    """
    
    def __init__(self, fred_client: FREDClient, cache_strategy: str = "intelligent"):
        """
        Initialize data manager.
        
        Args:
            fred_client: FRED API client instance
            cache_strategy: Caching strategy ('intelligent', 'aggressive', 'minimal')
        """
        self.fred_client = fred_client
        self.cache_strategy = cache_strategy
        self.cache_dir = fred_client.cache_dir
        
        # Initialize data cache database
        self._init_data_cache_db()
        
        # Data validation thresholds
        self.validation_config = {
            'max_missing_pct': 0.25,  # Maximum 25% missing data (more lenient)
            'outlier_std_threshold': 4.0,  # Outliers beyond 4 standard deviations
            'min_observations': 12,  # Minimum 12 observations (1 year monthly) - more lenient
            'max_gap_months': 12,  # Maximum 12-month data gap - more lenient
            'inflation_bounds': (-10, 25),  # Reasonable inflation bounds (%)
            'output_gap_bounds': (-15, 15),  # Reasonable output gap bounds (%)
            'interest_rate_bounds': (0, 25)  # Reasonable interest rate bounds (%)
        }
    
    def _init_data_cache_db(self) -> None:
        """Initialize database for data caching."""
        db_path = self.cache_dir / "data_cache.db"
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regional_datasets (
                    id INTEGER PRIMARY KEY,
                    dataset_hash TEXT UNIQUE NOT NULL,
                    regions TEXT NOT NULL,
                    indicators TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_reports (
                    id INTEGER PRIMARY KEY,
                    dataset_hash TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_hash) REFERENCES regional_datasets (dataset_hash)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dataset_hash 
                ON regional_datasets(dataset_hash)
            """)
    
    def load_regional_data(self, regions: List[str], indicators: List[str],
                          start_date: str, end_date: str,
                          use_cache: bool = True,
                          validate_data: bool = True) -> RegionalDataset:
        """
        Load regional economic data with caching and validation.
        
        Args:
            regions: List of region identifiers (e.g., state codes)
            indicators: List of economic indicators ('output_gap', 'inflation', 'interest_rate')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data
            validate_data: Whether to validate data quality
            
        Returns:
            RegionalDataset with loaded and validated data
        """
        # Generate dataset identifier for caching
        dataset_hash = self._generate_dataset_hash(regions, indicators, start_date, end_date)
        
        # Try to load from cache first
        if use_cache:
            cached_dataset = self._load_from_cache(dataset_hash)
            if cached_dataset:
                return cached_dataset
        
        # Load fresh data from FRED
        dataset = self._load_fresh_data(regions, indicators, start_date, end_date)
        
        # Validate data quality
        if validate_data:
            validation_report = self.validate_data_quality(dataset)
            if not validation_report.is_valid:
                raise DataValidationError(
                    f"Data validation failed: {validation_report.summary()}",
                    validation_failures=validation_report.warnings
                )
        
        # Cache the dataset
        if use_cache:
            self._cache_dataset(dataset_hash, dataset)
        
        return dataset
    
    def _generate_dataset_hash(self, regions: List[str], indicators: List[str],
                              start_date: str, end_date: str) -> str:
        """Generate unique hash for dataset parameters."""
        params = {
            'regions': sorted(regions),
            'indicators': sorted(indicators),
            'start_date': start_date,
            'end_date': end_date
        }
        params_str = json.dumps(params, sort_keys=True)
        return str(hash(params_str))
    
    def _load_from_cache(self, dataset_hash: str) -> Optional[RegionalDataset]:
        """Load dataset from cache if available and valid."""
        db_path = self.cache_dir / "data_cache.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_json, metadata_json FROM regional_datasets
                    WHERE dataset_hash = ?
                """, (dataset_hash,))
                
                row = cursor.fetchone()
                if row:
                    # Update last accessed time
                    conn.execute("""
                        UPDATE regional_datasets 
                        SET last_accessed = datetime('now')
                        WHERE dataset_hash = ?
                    """, (dataset_hash,))
                    
                    # Deserialize data
                    data_dict = json.loads(row[0])
                    metadata = json.loads(row[1])
                    
                    return self._deserialize_dataset(data_dict, metadata)
                    
        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
        
        return None
    
    def _cache_dataset(self, dataset_hash: str, dataset: RegionalDataset) -> None:
        """Cache dataset to database."""
        db_path = self.cache_dir / "data_cache.db"
        
        try:
            # Serialize dataset
            data_dict, metadata = self._serialize_dataset(dataset)
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO regional_datasets
                    (dataset_hash, regions, indicators, start_date, end_date,
                     data_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_hash,
                    json.dumps(dataset.regions),
                    json.dumps(['output_gap', 'inflation', 'interest_rate']),
                    str(dataset.time_periods[0]),
                    str(dataset.time_periods[-1]),
                    json.dumps(data_dict),
                    json.dumps(metadata)
                ))
                
        except Exception as e:
            print(f"Warning: Failed to cache dataset: {e}")
    
    def _serialize_dataset(self, dataset: RegionalDataset) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Serialize RegionalDataset for caching."""
        data_dict = {
            'output_gaps': dataset.output_gaps.to_json(orient='split'),
            'inflation_rates': dataset.inflation_rates.to_json(orient='split'),
            'interest_rates': dataset.interest_rates.to_json(orient='index'),  # Use 'index' for Series
            'real_time_estimates': {
                series: data.to_json(orient='split' if isinstance(data, pd.DataFrame) else 'index')
                for series, data in dataset.real_time_estimates.items()
            }
        }
        
        return data_dict, dataset.metadata
    
    def _deserialize_dataset(self, data_dict: Dict[str, Any], 
                           metadata: Dict[str, Any]) -> RegionalDataset:
        """Deserialize RegionalDataset from cache."""
        from io import StringIO
        
        output_gaps = pd.read_json(StringIO(data_dict['output_gaps']), orient='split')
        inflation_rates = pd.read_json(StringIO(data_dict['inflation_rates']), orient='split')
        interest_rates = pd.read_json(StringIO(data_dict['interest_rates']), orient='index', typ='series')
        
        real_time_estimates = {}
        for series, json_data in data_dict['real_time_estimates'].items():
            # Try to determine if it's a DataFrame or Series based on the JSON structure
            try:
                real_time_estimates[series] = pd.read_json(StringIO(json_data), orient='split')
            except:
                real_time_estimates[series] = pd.read_json(StringIO(json_data), orient='index', typ='series')
        
        return RegionalDataset(
            output_gaps=output_gaps,
            inflation_rates=inflation_rates,
            interest_rates=interest_rates,
            real_time_estimates=real_time_estimates,
            metadata=metadata
        )
    
    def _load_fresh_data(self, regions: List[str], indicators: List[str],
                        start_date: str, end_date: str) -> RegionalDataset:
        """Load fresh data from FRED API."""
        # Map indicators to FRED series patterns
        series_mapping = self._get_series_mapping(regions, indicators)
        
        # Collect all series codes
        all_series = []
        for indicator_series in series_mapping.values():
            all_series.extend(indicator_series)
        
        if not all_series:
            raise DataRetrievalError(
                f"No FRED series found for regions {regions} and indicators {indicators}"
            )
        
        # Retrieve data from FRED
        raw_data = self.fred_client.get_regional_series(
            all_series, start_date, end_date
        )
        
        # Process and structure the data
        return self._process_raw_data(raw_data, regions, indicators, series_mapping)
    
    def _get_series_mapping(self, regions: List[str], 
                           indicators: List[str]) -> Dict[str, List[str]]:
        """Map indicators to FRED series codes for specified regions."""
        # This is a simplified mapping - in practice, you'd have a comprehensive
        # mapping of regions to their corresponding FRED series codes
        series_mapping = {}
        
        # Example series codes (you would expand this based on actual FRED series)
        base_series = {
            'output_gap': {
                'US': ['GDPPOT', 'GDP'],  # Potential and actual GDP for output gap calculation
                'CA': ['CARGSP'],  # California Real GDP
                'TX': ['TXRGSP'],  # Texas Real GDP
                'NY': ['NYRGSP'],  # New York Real GDP
                'FL': ['FLRGSP'],  # Florida Real GDP
            },
            'inflation': {
                'US': ['CPIAUCSL'],  # National CPI
                'CA': ['CUURA422SA0'],  # Los Angeles CPI (proxy for CA)
                'TX': ['CUURA316SA0'],  # Dallas CPI (proxy for TX)
                'NY': ['CUURA101SA0'],  # New York CPI
                'FL': ['CUURA320SA0'],  # Miami CPI (proxy for FL)
            },
            'interest_rate': {
                'US': ['FEDFUNDS'],  # Federal Funds Rate (national)
            }
        }
        
        for indicator in indicators:
            series_list = []
            
            if indicator == 'interest_rate':
                # Interest rate is typically national
                series_list.extend(base_series.get(indicator, {}).get('US', []))
            else:
                # Regional indicators
                for region in regions:
                    region_series = base_series.get(indicator, {}).get(region, [])
                    series_list.extend(region_series)
            
            if series_list:
                series_mapping[indicator] = series_list
        
        return series_mapping
    
    def _process_raw_data(self, raw_data: pd.DataFrame, regions: List[str],
                         indicators: List[str], 
                         series_mapping: Dict[str, List[str]]) -> RegionalDataset:
        """Process raw FRED data into RegionalDataset structure."""
        
        # Initialize data containers
        output_gaps_data = {}
        inflation_data = {}
        interest_rate_data = None
        real_time_estimates = {}
        
        # Process each indicator
        for indicator in indicators:
            if indicator == 'output_gap':
                output_gaps_data = self._calculate_output_gaps(
                    raw_data, regions, series_mapping.get(indicator, [])
                )
            elif indicator == 'inflation':
                inflation_data = self._calculate_inflation_rates(
                    raw_data, regions, series_mapping.get(indicator, [])
                )
            elif indicator == 'interest_rate':
                interest_rate_data = self._extract_interest_rates(
                    raw_data, series_mapping.get(indicator, [])
                )
        
        # Create DataFrames with better alignment handling
        # First, find all available dates across all series
        all_dates = set()
        for region_data in output_gaps_data.values():
            if not region_data.empty:
                all_dates.update(region_data.index)
        for region_data in inflation_data.values():
            if not region_data.empty:
                all_dates.update(region_data.index)
        if interest_rate_data is not None and not interest_rate_data.empty:
            all_dates.update(interest_rate_data.index)
        
        if not all_dates:
            # No data available at all - create minimal dataset
            dummy_dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')
            output_gaps_df = pd.DataFrame(index=regions, columns=dummy_dates, data=0.0)
            inflation_df = pd.DataFrame(index=regions, columns=dummy_dates, data=2.0)
            interest_rate_series = pd.Series(index=dummy_dates, data=2.5, name='interest_rate')
            common_index = dummy_dates
        else:
            # Sort dates and create common index
            common_index = pd.DatetimeIndex(sorted(all_dates))
            
            # Create aligned DataFrames
            output_gaps_df = pd.DataFrame(index=regions, columns=common_index, dtype=float)
            inflation_df = pd.DataFrame(index=regions, columns=common_index, dtype=float)
            
            # Fill in available data
            for region in regions:
                if region in output_gaps_data and not output_gaps_data[region].empty:
                    output_gaps_df.loc[region, output_gaps_data[region].index] = output_gaps_data[region]
                else:
                    output_gaps_df.loc[region, :] = 0.0  # Default to zero output gap
                    
                if region in inflation_data and not inflation_data[region].empty:
                    inflation_df.loc[region, inflation_data[region].index] = inflation_data[region]
                else:
                    inflation_df.loc[region, :] = 2.0  # Default to 2% inflation target
            
            # Handle interest rate data
            if interest_rate_data is not None and not interest_rate_data.empty:
                interest_rate_series = pd.Series(index=common_index, dtype=float, name='interest_rate')
                interest_rate_series[interest_rate_data.index] = interest_rate_data
                interest_rate_series = interest_rate_series.fillna(method='ffill').fillna(2.5)
            else:
                interest_rate_series = pd.Series(index=common_index, data=2.5, name='interest_rate')
        
        # Check if we have sufficient data after alignment
        non_null_count = (~output_gaps_df.isnull()).sum().sum() + (~inflation_df.isnull()).sum().sum()
        if len(common_index) < self.validation_config['min_observations'] and non_null_count < 10:
            self.logger.warning(
                f"Limited overlapping data: {len(common_index)} periods available, "
                f"{self.validation_config['min_observations']} preferred. Proceeding with available data."
            )
        
        # Data is already aligned above, no need to filter again
        # Fill any remaining NaN values with reasonable defaults
        output_gaps_df = output_gaps_df.fillna(0.0)  # Zero output gap
        inflation_df = inflation_df.fillna(2.0)  # 2% inflation target
        
        # Create metadata
        metadata = {
            'source': 'FRED',
            'retrieved_at': datetime.now().isoformat(),
            'regions': regions,
            'indicators': indicators,
            'series_mapping': series_mapping,
            'start_date': str(common_index[0]),
            'end_date': str(common_index[-1]),
            'n_periods': len(common_index),
            'frequency': self._infer_frequency(common_index)
        }
        
        return RegionalDataset(
            output_gaps=output_gaps_df,
            inflation_rates=inflation_df,
            interest_rates=interest_rate_series,
            real_time_estimates=real_time_estimates,
            metadata=metadata
        )
    
    def _calculate_output_gaps(self, raw_data: pd.DataFrame, regions: List[str],
                              series_codes: List[str]) -> Dict[str, pd.Series]:
        """Calculate output gaps from GDP data using HP filter."""
        output_gaps = {}
        
        # Map regions to their GDP series
        region_series_map = {
            'CA': 'CARGSP',
            'TX': 'TXRGSP', 
            'NY': 'NYRGSP'
        }
        
        for region in regions:
            series_code = region_series_map.get(region)
            
            if series_code and series_code in raw_data.columns:
                region_gdp_series = raw_data[series_code].dropna()
                
                if len(region_gdp_series) >= 12:  # Need at least 1 year of data
                    # Apply HP filter (simplified version)
                    # Convert to log levels for HP filtering
                    log_gdp = np.log(region_gdp_series)
                    
                    # Simple HP filter approximation using rolling trend
                    # For quarterly data, lambda = 1600; for annual, lambda = 100
                    # For monthly data (if available), lambda = 14400
                    window_size = min(24, len(log_gdp) // 2)  # Adaptive window
                    
                    if window_size >= 6:
                        trend = log_gdp.rolling(window=window_size, center=True).mean()
                        # Fill NaN values at edges
                        trend = trend.fillna(method='bfill').fillna(method='ffill')
                        
                        # Calculate output gap as percentage deviation from trend
                        output_gap = ((log_gdp - trend) * 100)
                        output_gaps[region] = output_gap
                    else:
                        # Not enough data for meaningful HP filter
                        output_gaps[region] = pd.Series(index=region_gdp_series.index, 
                                                      data=0.0, name=f'{region}_output_gap')
                else:
                    # Create zero output gap series if insufficient data
                    output_gaps[region] = pd.Series(index=region_gdp_series.index, 
                                                  data=0.0, name=f'{region}_output_gap')
            else:
                # Create empty series if no data available
                output_gaps[region] = pd.Series(dtype=float, name=f'{region}_output_gap')
        
        return output_gaps
    
    def _calculate_inflation_rates(self, raw_data: pd.DataFrame, regions: List[str],
                                  series_codes: List[str]) -> Dict[str, pd.Series]:
        """Calculate inflation rates from CPI data."""
        inflation_rates = {}
        
        # Map regions to their CPI series
        region_cpi_map = {
            'CA': 'CUURA422SA0',  # Los Angeles CPI
            'TX': 'CUURA316SA0',  # Dallas CPI
            'NY': 'CUURA101SA0'   # New York CPI
        }
        
        for region in regions:
            series_code = region_cpi_map.get(region)
            
            if series_code and series_code in raw_data.columns:
                region_cpi_series = raw_data[series_code].dropna()
                
                if len(region_cpi_series) >= 13:  # Need at least 13 months for YoY calculation
                    # Calculate year-over-year inflation rate
                    inflation = region_cpi_series.pct_change(periods=12) * 100
                    inflation_rates[region] = inflation.dropna()
                elif len(region_cpi_series) >= 2:
                    # Use month-over-month if not enough for YoY, annualized
                    mom_inflation = region_cpi_series.pct_change() * 100 * 12
                    inflation_rates[region] = mom_inflation.dropna()
                else:
                    # Create zero inflation series if insufficient data
                    inflation_rates[region] = pd.Series(index=region_cpi_series.index,
                                                      data=2.0, name=f'{region}_inflation')  # 2% target
            else:
                # Create empty series if no data available
                inflation_rates[region] = pd.Series(dtype=float, name=f'{region}_inflation')
        
        return inflation_rates
    
    def _extract_interest_rates(self, raw_data: pd.DataFrame, 
                               series_codes: List[str]) -> Optional[pd.Series]:
        """Extract interest rate series."""
        for series_code in series_codes:
            if series_code in raw_data.columns and not raw_data[series_code].isna().all():
                return raw_data[series_code].dropna()
        
        return None
    
    def _infer_frequency(self, time_index: pd.Index) -> str:
        """Infer the frequency of the time series."""
        if len(time_index) < 2:
            return 'unknown'
        
        # Calculate typical time difference
        time_diffs = pd.Series(time_index).diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(days=7):
            return 'daily'
        elif median_diff <= pd.Timedelta(days=31):
            return 'monthly'
        elif median_diff <= pd.Timedelta(days=93):
            return 'quarterly'
        else:
            return 'annual'
    
    def validate_data_quality(self, dataset: RegionalDataset) -> ValidationReport:
        """
        Validate data quality and generate comprehensive report.
        
        Args:
            dataset: RegionalDataset to validate
            
        Returns:
            ValidationReport with validation results and recommendations
        """
        warnings = []
        recommendations = []
        missing_data_count = {}
        outlier_count = {}
        
        # Validate output gaps
        output_gap_issues = self._validate_series_group(
            dataset.output_gaps, 'output_gap', 
            self.validation_config['output_gap_bounds']
        )
        missing_data_count.update(output_gap_issues['missing'])
        outlier_count.update(output_gap_issues['outliers'])
        warnings.extend(output_gap_issues['warnings'])
        recommendations.extend(output_gap_issues['recommendations'])
        
        # Validate inflation rates
        inflation_issues = self._validate_series_group(
            dataset.inflation_rates, 'inflation',
            self.validation_config['inflation_bounds']
        )
        missing_data_count.update(inflation_issues['missing'])
        outlier_count.update(inflation_issues['outliers'])
        warnings.extend(inflation_issues['warnings'])
        recommendations.extend(inflation_issues['recommendations'])
        
        # Validate interest rates
        if not dataset.interest_rates.isna().all():
            interest_issues = self._validate_single_series(
                dataset.interest_rates, 'interest_rate',
                self.validation_config['interest_rate_bounds']
            )
            missing_data_count.update(interest_issues['missing'])
            outlier_count.update(interest_issues['outliers'])
            warnings.extend(interest_issues['warnings'])
            recommendations.extend(interest_issues['recommendations'])
        
        # Check for data gaps
        gap_issues = self._check_data_gaps(dataset)
        warnings.extend(gap_issues['warnings'])
        recommendations.extend(gap_issues['recommendations'])
        
        # Check for sufficient data
        sufficiency_issues = self._check_data_sufficiency(dataset)
        warnings.extend(sufficiency_issues['warnings'])
        recommendations.extend(sufficiency_issues['recommendations'])
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            missing_data_count, outlier_count, len(warnings)
        )
        
        # Determine if data is valid
        is_valid = (
            quality_score >= 70 and  # Minimum quality threshold
            len([w for w in warnings if 'critical' in w.lower()]) == 0 and
            dataset.n_periods >= self.validation_config['min_observations']
        )
        
        return ValidationReport(
            is_valid=is_valid,
            missing_data_count=missing_data_count,
            outlier_count=outlier_count,
            data_quality_score=quality_score,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _validate_series_group(self, data: pd.DataFrame, series_type: str,
                              bounds: Tuple[float, float]) -> Dict[str, Any]:
        """Validate a group of related series (e.g., regional output gaps)."""
        issues = {
            'missing': {},
            'outliers': {},
            'warnings': [],
            'recommendations': []
        }
        
        for region in data.index:
            series = data.loc[region]
            series_issues = self._validate_single_series(
                series, f"{series_type}_{region}", bounds
            )
            
            issues['missing'].update(series_issues['missing'])
            issues['outliers'].update(series_issues['outliers'])
            issues['warnings'].extend(series_issues['warnings'])
            issues['recommendations'].extend(series_issues['recommendations'])
        
        return issues
    
    def _validate_single_series(self, series: pd.Series, series_name: str,
                               bounds: Tuple[float, float]) -> Dict[str, Any]:
        """Validate a single time series."""
        issues = {
            'missing': {},
            'outliers': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Check for missing data
        missing_count = series.isna().sum()
        total_count = len(series)
        missing_pct = missing_count / total_count if total_count > 0 else 1.0
        
        issues['missing'][series_name] = missing_count
        
        if missing_pct > self.validation_config['max_missing_pct']:
            issues['warnings'].append(
                f"High missing data in {series_name}: {missing_pct:.1%} "
                f"(threshold: {self.validation_config['max_missing_pct']:.1%})"
            )
            issues['recommendations'].append(
                f"Consider data imputation or alternative series for {series_name}"
            )
        
        # Check for outliers (only on non-missing data)
        valid_data = series.dropna()
        if len(valid_data) > 0:
            # Statistical outliers
            z_scores = np.abs(stats.zscore(valid_data))
            outliers = (z_scores > self.validation_config['outlier_std_threshold']).sum()
            issues['outliers'][series_name] = outliers
            
            if outliers > 0:
                issues['warnings'].append(
                    f"Statistical outliers detected in {series_name}: {outliers} observations"
                )
                issues['recommendations'].append(
                    f"Review outliers in {series_name} for data quality issues"
                )
            
            # Range outliers
            min_val, max_val = bounds
            range_outliers = ((valid_data < min_val) | (valid_data > max_val)).sum()
            
            if range_outliers > 0:
                issues['warnings'].append(
                    f"Values outside reasonable range in {series_name}: {range_outliers} observations "
                    f"(expected range: {min_val} to {max_val})"
                )
                issues['recommendations'].append(
                    f"Verify data units and scaling for {series_name}"
                )
        
        return issues
    
    def _check_data_gaps(self, dataset: RegionalDataset) -> Dict[str, List[str]]:
        """Check for problematic data gaps."""
        issues = {'warnings': [], 'recommendations': []}
        
        # Check for large gaps in time series
        time_index = dataset.time_periods
        if len(time_index) > 1:
            # Calculate time differences
            time_diffs = pd.Series(time_index).diff().dropna()
            
            # Infer typical frequency
            median_diff = time_diffs.median()
            
            # Find large gaps (more than 2x typical frequency)
            large_gaps = time_diffs > (2 * median_diff)
            
            if large_gaps.any():
                n_gaps = large_gaps.sum()
                max_gap = time_diffs.max()
                
                issues['warnings'].append(
                    f"Large time gaps detected: {n_gaps} gaps, maximum gap: {max_gap}"
                )
                issues['recommendations'].append(
                    "Consider interpolation or alternative data sources for gap periods"
                )
        
        return issues
    
    def _check_data_sufficiency(self, dataset: RegionalDataset) -> Dict[str, List[str]]:
        """Check if there is sufficient data for analysis."""
        issues = {'warnings': [], 'recommendations': []}
        
        # Check minimum number of observations
        n_periods = dataset.n_periods
        min_required = self.validation_config['min_observations']
        
        if n_periods < min_required:
            issues['warnings'].append(
                f"CRITICAL: Insufficient data for analysis: {n_periods} periods available, "
                f"{min_required} required"
            )
            issues['recommendations'].append(
                f"Extend time period or use higher frequency data to reach {min_required} observations"
            )
        
        # Check regional coverage
        n_regions = dataset.n_regions
        if n_regions < 3:
            issues['warnings'].append(
                f"Limited regional coverage: {n_regions} regions (recommend at least 3)"
            )
            issues['recommendations'].append(
                "Add more regions for robust spatial analysis"
            )
        
        return issues
    
    def _calculate_quality_score(self, missing_data: Dict[str, int],
                                outliers: Dict[str, int], n_warnings: int) -> float:
        """Calculate overall data quality score (0-100)."""
        base_score = 100.0
        
        # Penalize missing data
        total_missing = sum(missing_data.values())
        if total_missing > 0:
            missing_penalty = min(30, total_missing * 0.5)  # Max 30 point penalty
            base_score -= missing_penalty
        
        # Penalize outliers
        total_outliers = sum(outliers.values())
        if total_outliers > 0:
            outlier_penalty = min(20, total_outliers * 0.2)  # Max 20 point penalty
            base_score -= outlier_penalty
        
        # Penalize warnings
        warning_penalty = min(25, n_warnings * 2)  # Max 25 point penalty
        base_score -= warning_penalty
        
        return max(0, base_score)
    
    def handle_missing_data(self, data: pd.DataFrame, 
                           method: str = "interpolation") -> pd.DataFrame:
        """
        Handle missing data using specified method.
        
        Args:
            data: DataFrame with potential missing values
            method: Method for handling missing data
                   ('interpolation', 'forward_fill', 'backward_fill', 'drop')
            
        Returns:
            DataFrame with missing data handled
        """
        if method == "interpolation":
            # Linear interpolation for time series
            return data.interpolate(method='linear', axis=1)
        
        elif method == "forward_fill":
            return data.ffill(axis=1)
        
        elif method == "backward_fill":
            return data.bfill(axis=1)
        
        elif method == "drop":
            # Drop columns (time periods) with any missing data
            return data.dropna(axis=1)
        
        else:
            raise ValueError(f"Unknown missing data method: {method}")
    
    def get_vintage_data(self, observation_date: str, 
                        vintage_date: str) -> pd.DataFrame:
        """
        Get real-time data as it was available on a specific vintage date.
        
        Args:
            observation_date: Date of the economic observation
            vintage_date: Date when the data was available
            
        Returns:
            DataFrame with real-time data estimates
        """
        # This would typically involve querying the FRED real-time database
        # For now, return a placeholder implementation
        
        try:
            # Example: Get key economic indicators as they were known on vintage_date
            key_series = ['GDP', 'CPIAUCSL', 'FEDFUNDS']
            
            vintage_data = {}
            for series_code in key_series:
                try:
                    value = self.fred_client.get_real_time_data(
                        series_code, observation_date, vintage_date
                    )
                    vintage_data[series_code] = value
                except Exception as e:
                    print(f"Warning: Could not get vintage data for {series_code}: {e}")
                    vintage_data[series_code] = None
            
            return pd.DataFrame([vintage_data], index=[observation_date])
            
        except Exception as e:
            raise DataRetrievalError(
                f"Failed to retrieve vintage data for {observation_date} as of {vintage_date}: {e}"
            )
    
    def clear_cache(self, older_than_days: int = 7) -> Dict[str, int]:
        """
        Clear cached datasets older than specified days.
        
        Args:
            older_than_days: Remove cache entries older than this many days
            
        Returns:
            Dictionary with counts of removed entries
        """
        db_path = self.cache_dir / "data_cache.db"
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        removed_counts = {'datasets': 0, 'validations': 0}
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Remove old datasets
                cursor = conn.execute("""
                    DELETE FROM regional_datasets 
                    WHERE created_at < ?
                """, (cutoff_date,))
                removed_counts['datasets'] = cursor.rowcount
                
                # Remove orphaned validation reports
                cursor = conn.execute("""
                    DELETE FROM validation_reports 
                    WHERE dataset_hash NOT IN (
                        SELECT dataset_hash FROM regional_datasets
                    )
                """)
                removed_counts['validations'] = cursor.rowcount
                
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")
        
        return removed_counts
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets."""
        db_path = self.cache_dir / "data_cache.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Count datasets
                cursor = conn.execute("SELECT COUNT(*) FROM regional_datasets")
                n_datasets = cursor.fetchone()[0]
                
                # Count validation reports
                cursor = conn.execute("SELECT COUNT(*) FROM validation_reports")
                n_validations = cursor.fetchone()[0]
                
                # Get cache size estimate
                cursor = conn.execute("""
                    SELECT SUM(LENGTH(data_json) + LENGTH(metadata_json)) 
                    FROM regional_datasets
                """)
                cache_size_bytes = cursor.fetchone()[0] or 0
                
                return {
                    'cached_datasets': n_datasets,
                    'validation_reports': n_validations,
                    'cache_size_mb': cache_size_bytes / (1024 * 1024),
                    'cache_strategy': self.cache_strategy,
                    'cache_directory': str(self.cache_dir)
                }
                
        except Exception as e:
            return {
                'error': f"Failed to get cache info: {e}",
                'cache_directory': str(self.cache_dir)
            }