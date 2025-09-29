"""
Core data models for regional monetary policy analysis.
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
import pandas as pd
import numpy as np


@dataclass
class RegionalDataset:
    """
    Container for regional economic data used in monetary policy analysis.
    
    This class stores regional output gaps, inflation rates, interest rates,
    and real-time data estimates with vintage tracking capabilities.
    """
    
    output_gaps: pd.DataFrame  # Regions x Time matrix
    inflation_rates: pd.DataFrame  # Regions x Time matrix  
    interest_rates: pd.DataFrame  # Time series of policy rates
    real_time_estimates: Dict[str, pd.DataFrame]  # Vintage tracking by series
    metadata: Dict[str, Any]  # Data source and processing metadata
    
    def __post_init__(self):
        """Validate data consistency after initialization."""
        self._validate_data_structure()
    
    def _validate_data_structure(self):
        """Ensure data matrices have consistent dimensions and indices."""
        if not self.output_gaps.index.equals(self.inflation_rates.index):
            raise ValueError("Output gaps and inflation rates must have matching region indices")
        
        if not self.output_gaps.columns.equals(self.inflation_rates.columns):
            raise ValueError("Output gaps and inflation rates must have matching time indices")
        
        if not self.interest_rates.index.equals(self.output_gaps.columns):
            raise ValueError("Interest rates index must match regional data time columns")
    
    def get_region_data(self, region: str) -> pd.DataFrame:
        """
        Extract all economic indicators for a specific region.
        
        Args:
            region: Region identifier
            
        Returns:
            DataFrame with output gap, inflation, and interest rate data for the region
        """
        if region not in self.output_gaps.index:
            raise ValueError(f"Region '{region}' not found in dataset")
        
        region_data = pd.DataFrame({
            'output_gap': self.output_gaps.loc[region],
            'inflation': self.inflation_rates.loc[region],
            'interest_rate': self.interest_rates
        })
        
        return region_data
    
    def get_time_period(self, start: str, end: str) -> 'RegionalDataset':
        """
        Extract data for a specific time period.
        
        Args:
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            
        Returns:
            New RegionalDataset with filtered time period
        """
        time_mask = (self.output_gaps.columns >= start) & (self.output_gaps.columns <= end)
        
        filtered_real_time = {}
        for series, data in self.real_time_estimates.items():
            filtered_real_time[series] = data.loc[
                (data.index >= start) & (data.index <= end)
            ]
        
        return RegionalDataset(
            output_gaps=self.output_gaps.loc[:, time_mask],
            inflation_rates=self.inflation_rates.loc[:, time_mask],
            interest_rates=self.interest_rates.loc[
                (self.interest_rates.index >= start) & (self.interest_rates.index <= end)
            ],
            real_time_estimates=filtered_real_time,
            metadata=self.metadata.copy()
        )
    
    def apply_transformation(self, transform_func: Callable) -> 'RegionalDataset':
        """
        Apply a transformation function to the economic data.
        
        Args:
            transform_func: Function that takes and returns a DataFrame
            
        Returns:
            New RegionalDataset with transformed data
        """
        return RegionalDataset(
            output_gaps=transform_func(self.output_gaps),
            inflation_rates=transform_func(self.inflation_rates),
            interest_rates=transform_func(self.interest_rates),
            real_time_estimates={
                series: transform_func(data) 
                for series, data in self.real_time_estimates.items()
            },
            metadata=self.metadata.copy()
        )
    
    @property
    def regions(self) -> list:
        """Get list of regions in the dataset."""
        return self.output_gaps.index.tolist()
    
    @property
    def time_periods(self) -> pd.Index:
        """Get time periods covered by the dataset."""
        return self.output_gaps.columns
    
    @property
    def n_regions(self) -> int:
        """Get number of regions in the dataset."""
        return len(self.output_gaps.index)
    
    @property
    def n_periods(self) -> int:
        """Get number of time periods in the dataset."""
        return len(self.output_gaps.columns)


@dataclass
class ValidationReport:
    """
    Container for data validation results.
    """
    
    is_valid: bool
    missing_data_count: Dict[str, int]
    outlier_count: Dict[str, int]
    data_quality_score: float
    warnings: list
    recommendations: list
    
    def summary(self) -> str:
        """Generate a summary report of validation results."""
        status = "PASSED" if self.is_valid else "FAILED"
        return f"""
Data Validation Report
=====================
Status: {status}
Quality Score: {self.data_quality_score:.2f}/100

Missing Data:
{self._format_dict(self.missing_data_count)}

Outliers Detected:
{self._format_dict(self.outlier_count)}

Warnings: {len(self.warnings)}
Recommendations: {len(self.recommendations)}
        """.strip()
    
    def _format_dict(self, d: Dict[str, int]) -> str:
        """Format dictionary for display."""
        if not d:
            return "  None"
        return "\n".join(f"  {k}: {v}" for k, v in d.items())