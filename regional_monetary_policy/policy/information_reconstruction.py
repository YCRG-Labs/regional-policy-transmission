"""
Information set reconstruction for historical policy analysis.

This module reconstructs the information available to the Federal Reserve
at the time of policy decisions, accounting for real-time data limitations,
revisions, and information lags.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from ..data.models import RegionalDataset


@dataclass
class InformationSet:
    """
    Container for Fed's information set at a specific point in time.
    """
    
    decision_date: str                    # Date of policy decision
    available_data: pd.DataFrame          # Data available at decision time
    data_vintages: Dict[str, str]        # Vintage dates for each series
    information_lags: Dict[str, int]     # Publication lags for each series
    forecast_data: Optional[pd.DataFrame] = None  # Fed forecasts if available
    
    def __post_init__(self):
        """Validate information set consistency."""
        self._validate_information_set()
    
    def _validate_information_set(self):
        """Ensure information set is internally consistent."""
        decision_dt = pd.to_datetime(self.decision_date)
        
        # Check that all data is available before decision date
        if not self.available_data.index.max() <= decision_dt:
            warnings.warn("Some data appears to be from after the decision date")
        
        # Validate vintage dates
        for series, vintage in self.data_vintages.items():
            vintage_dt = pd.to_datetime(vintage)
            if vintage_dt > decision_dt:
                raise ValueError(f"Vintage date {vintage} is after decision date {self.decision_date}")
    
    def get_latest_observation(self, series_name: str) -> Tuple[Any, str]:
        """
        Get the latest available observation for a series.
        
        Args:
            series_name: Name of the data series
            
        Returns:
            Tuple of (value, observation_date)
        """
        if series_name not in self.available_data.columns:
            raise ValueError(f"Series {series_name} not found in information set")
        
        series_data = self.available_data[series_name].dropna()
        if len(series_data) == 0:
            return None, None
        
        latest_date = series_data.index[-1]
        latest_value = series_data.iloc[-1]
        
        return latest_value, str(latest_date)
    
    def get_data_as_of_lag(self, series_name: str, lag_months: int) -> pd.Series:
        """
        Get data series as it would have appeared with a specific lag.
        
        Args:
            series_name: Name of the data series
            lag_months: Number of months of lag to apply
            
        Returns:
            Data series with lag applied
        """
        if series_name not in self.available_data.columns:
            raise ValueError(f"Series {series_name} not found in information set")
        
        decision_dt = pd.to_datetime(self.decision_date)
        cutoff_date = decision_dt - pd.DateOffset(months=lag_months)
        
        series_data = self.available_data[series_name]
        lagged_data = series_data[series_data.index <= cutoff_date]
        
        return lagged_data
    
    def summary_report(self) -> str:
        """Generate summary of information set."""
        n_series = len(self.available_data.columns)
        date_range = f"{self.available_data.index.min()} to {self.available_data.index.max()}"
        
        return f"""
Information Set Summary
======================
Decision Date: {self.decision_date}
Number of Series: {n_series}
Data Range: {date_range}

Series Coverage:
{self._format_series_coverage()}

Information Lags:
{self._format_information_lags()}
        """.strip()
    
    def _format_series_coverage(self) -> str:
        """Format series coverage information."""
        lines = []
        for col in self.available_data.columns:
            series_data = self.available_data[col].dropna()
            if len(series_data) > 0:
                latest_date = series_data.index[-1]
                vintage = self.data_vintages.get(col, 'Unknown')
                lines.append(f"  {col}: Latest = {latest_date}, Vintage = {vintage}")
            else:
                lines.append(f"  {col}: No data available")
        
        return "\n".join(lines)
    
    def _format_information_lags(self) -> str:
        """Format information lag information."""
        if not self.information_lags:
            return "  No lag information available"
        
        lines = []
        for series, lag in self.information_lags.items():
            lines.append(f"  {series}: {lag} months")
        
        return "\n".join(lines)


class InformationSetReconstructor:
    """
    Reconstructs historical Fed information sets for policy analysis.
    
    This class builds the information that was available to the Federal Reserve
    at each policy decision date, accounting for publication lags, data revisions,
    and real-time constraints.
    """
    
    def __init__(self,
                 publication_lags: Optional[Dict[str, int]] = None,
                 include_forecasts: bool = False):
        """
        Initialize information set reconstructor.
        
        Args:
            publication_lags: Dictionary mapping series names to publication lags (months)
            include_forecasts: Whether to include Fed forecasts in information sets
        """
        self.publication_lags = publication_lags or self._default_publication_lags()
        self.include_forecasts = include_forecasts
    
    def _default_publication_lags(self) -> Dict[str, int]:
        """Default publication lags for common economic series."""
        return {
            'gdp': 1,           # GDP published with 1 month lag
            'employment': 0,     # Employment data available quickly
            'inflation': 0,      # CPI available quickly
            'industrial_production': 0,  # IP available quickly
            'retail_sales': 0,   # Retail sales available quickly
            'housing_starts': 0, # Housing data available quickly
            'output_gap': 1,     # Output gap estimates have lag
            'core_inflation': 0  # Core inflation available quickly
        }
    
    def reconstruct_information_set(self,
                                  decision_date: str,
                                  real_time_data: RegionalDataset,
                                  vintage_data: Optional[Dict[str, pd.DataFrame]] = None) -> InformationSet:
        """
        Reconstruct Fed's information set for a specific decision date.
        
        Args:
            decision_date: Date of Fed policy decision
            real_time_data: Real-time regional dataset
            vintage_data: Historical data vintages (optional)
            
        Returns:
            InformationSet available at decision date
        """
        decision_dt = pd.to_datetime(decision_date)
        
        # Build available data considering publication lags
        available_data = self._build_available_data(decision_dt, real_time_data)
        
        # Determine data vintages
        data_vintages = self._determine_data_vintages(decision_dt, available_data, vintage_data)
        
        # Add forecasts if requested
        forecast_data = None
        if self.include_forecasts:
            forecast_data = self._reconstruct_forecasts(decision_dt, available_data)
        
        return InformationSet(
            decision_date=decision_date,
            available_data=available_data,
            data_vintages=data_vintages,
            information_lags=self.publication_lags,
            forecast_data=forecast_data
        )
    
    def _build_available_data(self,
                            decision_date: pd.Timestamp,
                            real_time_data: RegionalDataset) -> pd.DataFrame:
        """
        Build dataset of information available at decision date.
        
        Args:
            decision_date: Date of policy decision
            real_time_data: Real-time regional dataset
            
        Returns:
            DataFrame with available data
        """
        available_data = pd.DataFrame()
        
        # Process output gaps (transpose to have time as index)
        if real_time_data.output_gaps is not None:
            output_gap_transposed = real_time_data.output_gaps.T  # Transpose to time x regions
            output_gap_data = self._apply_publication_lags(
                output_gap_transposed, decision_date, 'output_gap'
            )
            for col in output_gap_data.columns:
                available_data[f'output_gap_{col}'] = output_gap_data[col]
        
        # Process inflation rates (transpose to have time as index)
        if real_time_data.inflation_rates is not None:
            inflation_transposed = real_time_data.inflation_rates.T  # Transpose to time x regions
            inflation_data = self._apply_publication_lags(
                inflation_transposed, decision_date, 'inflation'
            )
            for col in inflation_data.columns:
                available_data[f'inflation_{col}'] = inflation_data[col]
        
        # Process interest rates (no regional dimension)
        if real_time_data.interest_rates is not None:
            interest_rate_data = self._apply_publication_lags(
                real_time_data.interest_rates.to_frame('interest_rate'), 
                decision_date, 'interest_rate'
            )
            available_data['interest_rate'] = interest_rate_data['interest_rate']
        
        return available_data
    
    def _apply_publication_lags(self,
                              data: pd.DataFrame,
                              decision_date: pd.Timestamp,
                              series_type: str) -> pd.DataFrame:
        """
        Apply publication lags to data series.
        
        Args:
            data: Original data
            decision_date: Policy decision date
            series_type: Type of series for lag lookup
            
        Returns:
            Data with publication lags applied
        """
        # Get publication lag for this series type
        lag_months = self.publication_lags.get(series_type, 0)
        
        # Calculate cutoff date
        cutoff_date = decision_date - pd.DateOffset(months=lag_months)
        
        # Filter data to only include observations available by cutoff date
        available_data = data[data.index <= cutoff_date].copy()
        
        return available_data
    
    def _determine_data_vintages(self,
                               decision_date: pd.Timestamp,
                               available_data: pd.DataFrame,
                               vintage_data: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, str]:
        """
        Determine vintage dates for each data series.
        
        Args:
            decision_date: Policy decision date
            available_data: Available data at decision date
            vintage_data: Historical vintage data (optional)
            
        Returns:
            Dictionary mapping series names to vintage dates
        """
        data_vintages = {}
        
        for series_name in available_data.columns:
            if vintage_data and series_name in vintage_data:
                # Use actual vintage information if available
                vintage_df = vintage_data[series_name]
                
                # Find latest vintage before decision date
                vintage_dates = pd.to_datetime(vintage_df.columns)
                valid_vintages = vintage_dates[vintage_dates <= decision_date]
                
                if len(valid_vintages) > 0:
                    latest_vintage = valid_vintages.max()
                    data_vintages[series_name] = str(latest_vintage.date())
                else:
                    data_vintages[series_name] = str(decision_date.date())
            else:
                # Use decision date as vintage if no vintage data available
                data_vintages[series_name] = str(decision_date.date())
        
        return data_vintages
    
    def _reconstruct_forecasts(self,
                             decision_date: pd.Timestamp,
                             available_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct Fed forecasts available at decision date.
        
        This is a placeholder - actual implementation would require
        historical Fed forecast data.
        """
        # Placeholder implementation
        # In practice, this would use historical FOMC forecasts
        return None
    
    def reconstruct_historical_sequence(self,
                                      decision_dates: List[str],
                                      real_time_data: RegionalDataset,
                                      vintage_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[InformationSet]:
        """
        Reconstruct information sets for a sequence of decision dates.
        
        Args:
            decision_dates: List of Fed policy decision dates
            real_time_data: Real-time regional dataset
            vintage_data: Historical data vintages (optional)
            
        Returns:
            List of InformationSet objects
        """
        information_sets = []
        
        for date in decision_dates:
            try:
                info_set = self.reconstruct_information_set(date, real_time_data, vintage_data)
                information_sets.append(info_set)
            except Exception as e:
                warnings.warn(f"Could not reconstruct information set for {date}: {e}")
                continue
        
        return information_sets
    
    def analyze_information_evolution(self,
                                    information_sets: List[InformationSet]) -> pd.DataFrame:
        """
        Analyze how information availability evolved over time.
        
        Args:
            information_sets: List of historical information sets
            
        Returns:
            DataFrame with information evolution analysis
        """
        analysis_data = []
        
        for info_set in information_sets:
            decision_date = info_set.decision_date
            
            # Count available series
            n_series = len(info_set.available_data.columns)
            
            # Calculate data freshness (days between latest data and decision)
            latest_data_dates = []
            for col in info_set.available_data.columns:
                series_data = info_set.available_data[col].dropna()
                if len(series_data) > 0:
                    latest_data_dates.append(series_data.index[-1])
            
            if latest_data_dates:
                avg_data_lag = (pd.to_datetime(decision_date) - 
                              pd.to_datetime(latest_data_dates).mean()).days
            else:
                avg_data_lag = np.nan
            
            # Calculate data completeness
            total_possible_obs = len(info_set.available_data.index) * len(info_set.available_data.columns)
            actual_obs = info_set.available_data.count().sum()
            completeness = actual_obs / total_possible_obs if total_possible_obs > 0 else 0
            
            analysis_data.append({
                'decision_date': decision_date,
                'n_series': n_series,
                'avg_data_lag_days': avg_data_lag,
                'data_completeness': completeness
            })
        
        return pd.DataFrame(analysis_data)
    
    def compare_real_time_vs_revised(self,
                                   information_set: InformationSet,
                                   revised_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare real-time information set with subsequently revised data.
        
        Args:
            information_set: Historical information set
            revised_data: Final revised data
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison_results = {}
        
        for series_name in information_set.available_data.columns:
            if series_name in revised_data.columns:
                # Get overlapping time periods
                rt_series = information_set.available_data[series_name].dropna()
                revised_series = revised_data[series_name].dropna()
                
                common_dates = rt_series.index.intersection(revised_series.index)
                
                if len(common_dates) > 0:
                    rt_values = rt_series.loc[common_dates]
                    revised_values = revised_series.loc[common_dates]
                    
                    # Calculate revision statistics
                    revisions = revised_values - rt_values
                    
                    comparison_results[series_name] = {
                        'mean_revision': revisions.mean(),
                        'std_revision': revisions.std(),
                        'max_revision': revisions.abs().max(),
                        'correlation': rt_values.corr(revised_values),
                        'n_observations': len(common_dates)
                    }
        
        return comparison_results
    
    def estimate_information_value(self,
                                 information_sets: List[InformationSet],
                                 policy_outcomes: pd.Series) -> Dict[str, float]:
        """
        Estimate the value of different types of information for policy decisions.
        
        Args:
            information_sets: Historical information sets
            policy_outcomes: Observed policy outcomes
            
        Returns:
            Dictionary with information value estimates
        """
        # This would implement a more sophisticated analysis
        # For now, return placeholder results
        
        return {
            'output_gap_information_value': 0.5,
            'inflation_information_value': 0.7,
            'forecast_information_value': 0.3,
            'timeliness_value': 0.4
        }