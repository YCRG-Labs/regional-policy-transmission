"""
Data quality diagnostics and cleaning system for regional monetary policy analysis.

This module provides comprehensive data quality assessment, validation,
and cleaning capabilities for economic time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

from .exceptions import DataValidationError, InsufficientDataError
from .logging_config import get_logger, get_performance_logger


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_columns: List[str]
    affected_rows: Optional[List[int]] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""
    
    dataset_name: str
    assessment_date: datetime
    total_observations: int
    total_variables: int
    issues: List[DataQualityIssue] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: DataQualityIssue):
        """Add a data quality issue to the report."""
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: str) -> List[DataQualityIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.severity == 'critical' for issue in self.issues)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of data quality assessment."""
        severity_counts = {}
        for severity in ['low', 'medium', 'high', 'critical']:
            severity_counts[severity] = len(self.get_issues_by_severity(severity))
        
        return {
            'total_issues': len(self.issues),
            'severity_breakdown': severity_counts,
            'has_critical_issues': self.has_critical_issues(),
            'total_observations': self.total_observations,
            'total_variables': self.total_variables
        }


class DataQualityAssessor:
    """
    Comprehensive data quality assessment system.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.1,
                 outlier_threshold: float = 3.0,
                 min_observations: int = 24):
        """
        Initialize data quality assessor.
        
        Args:
            missing_threshold: Threshold for missing data warnings (0-1)
            outlier_threshold: Z-score threshold for outlier detection
            min_observations: Minimum required observations
        """
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.min_observations = min_observations
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.perf_logger = get_performance_logger(f"{__name__}.{self.__class__.__name__}")
    
    def assess_data_quality(self, data: pd.DataFrame, 
                          dataset_name: str = "Unknown") -> DataQualityReport:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            data: DataFrame to assess
            dataset_name: Name of the dataset
            
        Returns:
            DataQualityReport with findings and recommendations
        """
        with self.perf_logger.timer("data_quality_assessment", dataset=dataset_name):
            report = DataQualityReport(
                dataset_name=dataset_name,
                assessment_date=datetime.now(),
                total_observations=len(data),
                total_variables=len(data.columns)
            )
            
            # Check for sufficient data
            self._check_data_sufficiency(data, report)
            
            # Check for missing values
            self._check_missing_values(data, report)
            
            # Check for outliers
            self._check_outliers(data, report)
            
            # Check for duplicates
            self._check_duplicates(data, report)
            
            # Check data types and consistency
            self._check_data_types(data, report)
            
            # Check for temporal consistency (if time series)
            if isinstance(data.index, pd.DatetimeIndex):
                self._check_temporal_consistency(data, report)
            
            # Check for multicollinearity
            self._check_multicollinearity(data, report)
            
            # Generate summary statistics
            report.summary_statistics = self._generate_summary_statistics(data)
            
            # Generate recommendations
            self._generate_recommendations(report)
            
            self.logger.info(f"Data quality assessment completed for {dataset_name}: "
                           f"{len(report.issues)} issues found")
            
            return report
    
    def _check_data_sufficiency(self, data: pd.DataFrame, report: DataQualityReport):
        """Check if there is sufficient data for analysis."""
        if len(data) < self.min_observations:
            issue = DataQualityIssue(
                issue_type="insufficient_data",
                severity="critical",
                description=f"Dataset has only {len(data)} observations, minimum {self.min_observations} required",
                affected_columns=list(data.columns),
                recommendation="Collect more data or reduce minimum observation requirement"
            )
            report.add_issue(issue)
    
    def _check_missing_values(self, data: pd.DataFrame, report: DataQualityReport):
        """Check for missing values in the dataset."""
        missing_stats = data.isnull().sum()
        missing_pct = missing_stats / len(data)
        
        for col in data.columns:
            if missing_stats[col] > 0:
                severity = self._get_missing_severity(missing_pct[col])
                
                issue = DataQualityIssue(
                    issue_type="missing_values",
                    severity=severity,
                    description=f"Column '{col}' has {missing_stats[col]} missing values ({missing_pct[col]:.1%})",
                    affected_columns=[col],
                    affected_rows=data[data[col].isnull()].index.tolist(),
                    recommendation=self._get_missing_recommendation(missing_pct[col]),
                    metadata={'missing_count': missing_stats[col], 'missing_percentage': missing_pct[col]}
                )
                report.add_issue(issue)
    
    def _get_missing_severity(self, missing_pct: float) -> str:
        """Determine severity of missing data."""
        if missing_pct >= 0.5:
            return "critical"
        elif missing_pct >= 0.2:
            return "high"
        elif missing_pct >= self.missing_threshold:
            return "medium"
        else:
            return "low"
    
    def _get_missing_recommendation(self, missing_pct: float) -> str:
        """Get recommendation for handling missing data."""
        if missing_pct >= 0.5:
            return "Consider excluding this variable or collecting more data"
        elif missing_pct >= 0.2:
            return "Use advanced imputation methods or consider variable transformation"
        elif missing_pct >= 0.05:
            return "Use interpolation or forward/backward fill"
        else:
            return "Simple imputation methods should suffice"
    
    def _check_outliers(self, data: pd.DataFrame, report: DataQualityReport):
        """Check for outliers in numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].notna().sum() < 3:  # Need at least 3 observations
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outlier_indices = data[col].dropna().index[z_scores > self.outlier_threshold]
            
            if len(outlier_indices) > 0:
                outlier_pct = len(outlier_indices) / len(data[col].dropna())
                severity = "high" if outlier_pct > 0.05 else "medium" if outlier_pct > 0.02 else "low"
                
                issue = DataQualityIssue(
                    issue_type="outliers",
                    severity=severity,
                    description=f"Column '{col}' has {len(outlier_indices)} outliers ({outlier_pct:.1%})",
                    affected_columns=[col],
                    affected_rows=outlier_indices.tolist(),
                    recommendation="Consider winsorizing or investigating data collection issues",
                    metadata={'outlier_count': len(outlier_indices), 'outlier_percentage': outlier_pct}
                )
                report.add_issue(issue)
    
    def _check_duplicates(self, data: pd.DataFrame, report: DataQualityReport):
        """Check for duplicate rows."""
        duplicates = data.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            duplicate_pct = duplicate_count / len(data)
            severity = "high" if duplicate_pct > 0.1 else "medium" if duplicate_pct > 0.05 else "low"
            
            issue = DataQualityIssue(
                issue_type="duplicates",
                severity=severity,
                description=f"Dataset has {duplicate_count} duplicate rows ({duplicate_pct:.1%})",
                affected_columns=list(data.columns),
                affected_rows=data[duplicates].index.tolist(),
                recommendation="Remove duplicate rows or investigate data collection process",
                metadata={'duplicate_count': duplicate_count, 'duplicate_percentage': duplicate_pct}
            )
            report.add_issue(issue)
    
    def _check_data_types(self, data: pd.DataFrame, report: DataQualityReport):
        """Check data types and consistency."""
        for col in data.columns:
            # Check for mixed types
            if data[col].dtype == 'object':
                # Try to identify if it should be numeric
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    try:
                        pd.to_numeric(non_null_values)
                        issue = DataQualityIssue(
                            issue_type="data_type",
                            severity="medium",
                            description=f"Column '{col}' appears to be numeric but stored as object",
                            affected_columns=[col],
                            recommendation="Convert to numeric type and handle any conversion errors"
                        )
                        report.add_issue(issue)
                    except (ValueError, TypeError):
                        pass
    
    def _check_temporal_consistency(self, data: pd.DataFrame, report: DataQualityReport):
        """Check temporal consistency for time series data."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return
        
        # Check for gaps in time series
        expected_freq = pd.infer_freq(data.index)
        if expected_freq:
            expected_index = pd.date_range(start=data.index.min(), 
                                         end=data.index.max(), 
                                         freq=expected_freq)
            missing_dates = expected_index.difference(data.index)
            
            if len(missing_dates) > 0:
                missing_pct = len(missing_dates) / len(expected_index)
                severity = "high" if missing_pct > 0.1 else "medium" if missing_pct > 0.05 else "low"
                
                issue = DataQualityIssue(
                    issue_type="temporal_gaps",
                    severity=severity,
                    description=f"Time series has {len(missing_dates)} missing periods ({missing_pct:.1%})",
                    affected_columns=list(data.columns),
                    recommendation="Fill missing periods or adjust analysis to handle irregular frequency",
                    metadata={'missing_periods': len(missing_dates), 'expected_frequency': expected_freq}
                )
                report.add_issue(issue)
    
    def _check_multicollinearity(self, data: pd.DataFrame, report: DataQualityReport):
        """Check for multicollinearity among numeric variables."""
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) < 2:
            return
        
        try:
            corr_matrix = numeric_data.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.9:  # High correlation threshold
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_val
                        ))
            
            if high_corr_pairs:
                for col1, col2, corr_val in high_corr_pairs:
                    issue = DataQualityIssue(
                        issue_type="multicollinearity",
                        severity="medium",
                        description=f"High correlation ({corr_val:.3f}) between '{col1}' and '{col2}'",
                        affected_columns=[col1, col2],
                        recommendation="Consider removing one variable or using dimensionality reduction",
                        metadata={'correlation': corr_val}
                    )
                    report.add_issue(issue)
        
        except Exception as e:
            self.logger.warning(f"Could not compute correlation matrix: {e}")
    
    def _generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        summary = {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values_total': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / data.size) * 100
        }
        
        if len(numeric_data.columns) > 0:
            summary.update({
                'numeric_columns': len(numeric_data.columns),
                'mean_values': numeric_data.mean().to_dict(),
                'std_values': numeric_data.std().to_dict(),
                'min_values': numeric_data.min().to_dict(),
                'max_values': numeric_data.max().to_dict()
            })
        
        return summary
    
    def _generate_recommendations(self, report: DataQualityReport):
        """Generate overall recommendations based on issues found."""
        critical_issues = report.get_issues_by_severity('critical')
        high_issues = report.get_issues_by_severity('high')
        
        if critical_issues:
            report.recommendations.append("Address critical data quality issues before proceeding with analysis")
        
        if high_issues:
            report.recommendations.append("High-severity issues may significantly impact analysis results")
        
        # Specific recommendations based on issue types
        issue_types = set(issue.issue_type for issue in report.issues)
        
        if 'missing_values' in issue_types:
            report.recommendations.append("Implement appropriate missing value imputation strategy")
        
        if 'outliers' in issue_types:
            report.recommendations.append("Investigate and handle outliers appropriately")
        
        if 'temporal_gaps' in issue_types:
            report.recommendations.append("Address temporal gaps in time series data")
        
        if 'multicollinearity' in issue_types:
            report.recommendations.append("Consider variable selection or dimensionality reduction")


class DataCleaner:
    """
    Automated data cleaning system with multiple strategies.
    """
    
    def __init__(self):
        """Initialize data cleaner."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.perf_logger = get_performance_logger(f"{__name__}.{self.__class__.__name__}")
    
    def clean_data(self, data: pd.DataFrame, 
                  quality_report: Optional[DataQualityReport] = None,
                  cleaning_strategy: str = "conservative") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data based on quality assessment.
        
        Args:
            data: DataFrame to clean
            quality_report: Data quality report (will generate if not provided)
            cleaning_strategy: Cleaning strategy ('conservative', 'aggressive', 'custom')
            
        Returns:
            Tuple of (cleaned_data, cleaning_log)
        """
        with self.perf_logger.timer("data_cleaning", strategy=cleaning_strategy):
            cleaning_log = {
                'strategy': cleaning_strategy,
                'actions_taken': [],
                'original_shape': data.shape,
                'issues_addressed': []
            }
            
            # Generate quality report if not provided
            if quality_report is None:
                assessor = DataQualityAssessor()
                quality_report = assessor.assess_data_quality(data)
            
            cleaned_data = data.copy()
            
            # Handle critical issues first
            critical_issues = quality_report.get_issues_by_severity('critical')
            for issue in critical_issues:
                if issue.issue_type == 'insufficient_data':
                    raise InsufficientDataError(
                        issue.description,
                        required_periods=24,  # Default minimum
                        available_periods=len(data)
                    )
            
            # Handle missing values
            cleaned_data, missing_log = self._handle_missing_values(
                cleaned_data, quality_report, cleaning_strategy
            )
            cleaning_log['actions_taken'].extend(missing_log)
            
            # Handle outliers
            cleaned_data, outlier_log = self._handle_outliers(
                cleaned_data, quality_report, cleaning_strategy
            )
            cleaning_log['actions_taken'].extend(outlier_log)
            
            # Handle duplicates
            cleaned_data, duplicate_log = self._handle_duplicates(
                cleaned_data, quality_report, cleaning_strategy
            )
            cleaning_log['actions_taken'].extend(duplicate_log)
            
            # Handle data type issues
            cleaned_data, type_log = self._handle_data_types(
                cleaned_data, quality_report, cleaning_strategy
            )
            cleaning_log['actions_taken'].extend(type_log)
            
            cleaning_log['final_shape'] = cleaned_data.shape
            cleaning_log['rows_removed'] = data.shape[0] - cleaned_data.shape[0]
            cleaning_log['columns_removed'] = data.shape[1] - cleaned_data.shape[1]
            
            self.logger.info(f"Data cleaning completed: {len(cleaning_log['actions_taken'])} actions taken")
            
            return cleaned_data, cleaning_log
    
    def _handle_missing_values(self, data: pd.DataFrame, 
                             quality_report: DataQualityReport,
                             strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values based on strategy."""
        actions = []
        cleaned_data = data.copy()
        
        missing_issues = [issue for issue in quality_report.issues 
                         if issue.issue_type == 'missing_values']
        
        for issue in missing_issues:
            col = issue.affected_columns[0]
            missing_pct = issue.metadata.get('missing_percentage', 0)
            
            if strategy == 'aggressive' and missing_pct > 0.3:
                # Drop columns with too much missing data
                cleaned_data = cleaned_data.drop(columns=[col])
                actions.append(f"Dropped column '{col}' (missing: {missing_pct:.1%})")
            
            elif missing_pct > 0.5:
                # Always drop if more than 50% missing
                cleaned_data = cleaned_data.drop(columns=[col])
                actions.append(f"Dropped column '{col}' (missing: {missing_pct:.1%})")
            
            else:
                # Impute missing values
                if cleaned_data[col].dtype in ['int64', 'float64']:
                    # Numeric imputation
                    if isinstance(cleaned_data.index, pd.DatetimeIndex):
                        # Time series: use interpolation
                        cleaned_data[col] = cleaned_data[col].interpolate(method='time')
                        actions.append(f"Interpolated missing values in '{col}'")
                    else:
                        # Cross-sectional: use median
                        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                        actions.append(f"Filled missing values in '{col}' with median")
                else:
                    # Categorical: use mode
                    mode_val = cleaned_data[col].mode()
                    if len(mode_val) > 0:
                        cleaned_data[col] = cleaned_data[col].fillna(mode_val[0])
                        actions.append(f"Filled missing values in '{col}' with mode")
        
        return cleaned_data, actions
    
    def _handle_outliers(self, data: pd.DataFrame,
                        quality_report: DataQualityReport,
                        strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle outliers based on strategy."""
        actions = []
        cleaned_data = data.copy()
        
        outlier_issues = [issue for issue in quality_report.issues 
                         if issue.issue_type == 'outliers']
        
        for issue in outlier_issues:
            col = issue.affected_columns[0]
            outlier_pct = issue.metadata.get('outlier_percentage', 0)
            
            if strategy == 'conservative' and outlier_pct < 0.05:
                # Keep outliers in conservative mode unless too many
                continue
            
            # Winsorize outliers (cap at 1st and 99th percentiles)
            q01 = cleaned_data[col].quantile(0.01)
            q99 = cleaned_data[col].quantile(0.99)
            
            outliers_low = cleaned_data[col] < q01
            outliers_high = cleaned_data[col] > q99
            
            cleaned_data.loc[outliers_low, col] = q01
            cleaned_data.loc[outliers_high, col] = q99
            
            total_winsorized = outliers_low.sum() + outliers_high.sum()
            actions.append(f"Winsorized {total_winsorized} outliers in '{col}'")
        
        return cleaned_data, actions
    
    def _handle_duplicates(self, data: pd.DataFrame,
                          quality_report: DataQualityReport,
                          strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle duplicate rows."""
        actions = []
        cleaned_data = data.copy()
        
        duplicate_issues = [issue for issue in quality_report.issues 
                           if issue.issue_type == 'duplicates']
        
        if duplicate_issues:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            rows_removed = initial_rows - len(cleaned_data)
            
            if rows_removed > 0:
                actions.append(f"Removed {rows_removed} duplicate rows")
        
        return cleaned_data, actions
    
    def _handle_data_types(self, data: pd.DataFrame,
                          quality_report: DataQualityReport,
                          strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle data type issues."""
        actions = []
        cleaned_data = data.copy()
        
        type_issues = [issue for issue in quality_report.issues 
                      if issue.issue_type == 'data_type']
        
        for issue in type_issues:
            col = issue.affected_columns[0]
            
            try:
                # Attempt to convert to numeric
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                actions.append(f"Converted column '{col}' to numeric type")
            except Exception as e:
                self.logger.warning(f"Could not convert column '{col}' to numeric: {e}")
        
        return cleaned_data, actions


def validate_data_for_analysis(data: pd.DataFrame, 
                             min_observations: int = 24,
                             required_columns: Optional[List[str]] = None) -> DataQualityReport:
    """
    Validate data for monetary policy analysis.
    
    Args:
        data: DataFrame to validate
        min_observations: Minimum required observations
        required_columns: List of required column names
        
    Returns:
        DataQualityReport with validation results
        
    Raises:
        DataValidationError: If critical validation issues are found
    """
    assessor = DataQualityAssessor(min_observations=min_observations)
    report = assessor.assess_data_quality(data, "Analysis Dataset")
    
    # Check for required columns
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            issue = DataQualityIssue(
                issue_type="missing_required_columns",
                severity="critical",
                description=f"Missing required columns: {list(missing_cols)}",
                affected_columns=list(missing_cols),
                recommendation="Ensure all required data is available"
            )
            report.add_issue(issue)
    
    # Raise exception if critical issues found
    if report.has_critical_issues():
        critical_issues = report.get_issues_by_severity('critical')
        issue_descriptions = [issue.description for issue in critical_issues]
        raise DataValidationError(
            f"Critical data validation issues found: {'; '.join(issue_descriptions)}",
            validation_failures=issue_descriptions
        )
    
    return report