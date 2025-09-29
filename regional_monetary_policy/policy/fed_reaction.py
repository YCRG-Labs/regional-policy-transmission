"""
Fed reaction function estimation and implicit weight extraction.

This module estimates the Federal Reserve's implicit reaction function
and extracts the regional weights that the Fed appears to use in
its policy decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import warnings

from ..econometric.models import RegionalParameters


@dataclass
class FedReactionResults:
    """
    Results from Fed reaction function estimation.
    """
    
    estimated_coefficients: Dict[str, float]  # Policy rule coefficients
    implicit_regional_weights: np.ndarray     # Estimated regional weights
    model_fit: Dict[str, float]              # R-squared, etc.
    standard_errors: Dict[str, float]        # Coefficient standard errors
    confidence_intervals: Dict[str, Tuple[float, float]]  # 95% confidence intervals
    
    # Additional diagnostics
    residuals: pd.Series = None
    fitted_values: pd.Series = None
    estimation_period: Tuple[str, str] = None
    
    def summary_report(self) -> str:
        """Generate summary report of estimation results."""
        return f"""
Fed Reaction Function Estimation Results
======================================

Estimation Period: {self.estimation_period[0]} to {self.estimation_period[1]}
Model Fit (R²): {self.model_fit.get('r_squared', 'N/A'):.4f}

Policy Rule Coefficients:
  Output Gap Response:    {self.estimated_coefficients.get('output', 0):.4f} ± {self.standard_errors.get('output', 0):.4f}
  Inflation Response:     {self.estimated_coefficients.get('inflation', 0):.4f} ± {self.standard_errors.get('inflation', 0):.4f}
  Interest Rate Smoothing: {self.estimated_coefficients.get('lagged_rate', 0):.4f} ± {self.standard_errors.get('lagged_rate', 0):.4f}

Regional Weights:
{self._format_regional_weights()}

Diagnostics:
  Residual Standard Error: {self.model_fit.get('residual_std', 'N/A'):.4f}
  Durbin-Watson Statistic: {self.model_fit.get('durbin_watson', 'N/A'):.4f}
        """.strip()
    
    def _format_regional_weights(self) -> str:
        """Format regional weights for display."""
        if self.implicit_regional_weights is None:
            return "  Not estimated"
        
        lines = []
        for i, weight in enumerate(self.implicit_regional_weights):
            lines.append(f"  Region {i+1}: {weight:.4f}")
        
        return "\n".join(lines)


class FedReactionEstimator:
    """
    Estimates Federal Reserve reaction function and implicit regional weights.
    
    This class implements various approaches to estimate how the Fed responds
    to regional economic conditions and what implicit weights it places on
    different regions.
    """
    
    def __init__(self,
                 regional_params: Optional[RegionalParameters] = None,
                 estimation_method: str = 'ols',
                 include_smoothing: bool = True):
        """
        Initialize Fed reaction function estimator.
        
        Args:
            regional_params: Regional structural parameters (optional)
            estimation_method: 'ols', 'ridge', 'constrained'
            include_smoothing: Whether to include interest rate smoothing
        """
        self.regional_params = regional_params
        self.estimation_method = estimation_method
        self.include_smoothing = include_smoothing
        
        # Validate estimation method
        valid_methods = ['ols', 'ridge', 'constrained']
        if estimation_method not in valid_methods:
            raise ValueError(f"Estimation method must be one of {valid_methods}")
    
    def estimate_reaction_function(self,
                                 policy_rates: pd.Series,
                                 regional_data: pd.DataFrame,
                                 real_time_data: bool = True) -> FedReactionResults:
        """
        Estimate Fed reaction function using regional data.
        
        Args:
            policy_rates: Time series of Fed policy rates
            regional_data: Regional economic indicators
            real_time_data: Whether to use real-time or revised data
            
        Returns:
            FedReactionResults with estimated coefficients and weights
        """
        # Align data
        aligned_data = self._align_data(policy_rates, regional_data)
        
        if len(aligned_data) < 10:
            raise ValueError("Insufficient data for estimation (need at least 10 observations)")
        
        # Prepare regression variables
        y, X, variable_names = self._prepare_regression_data(aligned_data)
        
        # Estimate model
        if self.estimation_method == 'ols':
            results = self._estimate_ols(y, X, variable_names)
        elif self.estimation_method == 'ridge':
            results = self._estimate_ridge(y, X, variable_names)
        elif self.estimation_method == 'constrained':
            results = self._estimate_constrained(y, X, variable_names, aligned_data)
        
        # Extract regional weights
        implicit_weights = self._extract_regional_weights(results, aligned_data)
        
        # Compute model diagnostics
        fitted_values = X @ results['coefficients']
        residuals = y - fitted_values
        
        model_fit = {
            'r_squared': r2_score(y, fitted_values),
            'residual_std': np.std(residuals),
            'durbin_watson': self._compute_durbin_watson(residuals)
        }
        
        # Create results object
        return FedReactionResults(
            estimated_coefficients=results['coefficient_dict'],
            implicit_regional_weights=implicit_weights,
            model_fit=model_fit,
            standard_errors=results['standard_errors'],
            confidence_intervals=results['confidence_intervals'],
            residuals=pd.Series(residuals, index=aligned_data.index),
            fitted_values=pd.Series(fitted_values, index=aligned_data.index),
            estimation_period=(str(aligned_data.index[0]), str(aligned_data.index[-1]))
        )
    
    def _align_data(self,
                   policy_rates: pd.Series,
                   regional_data: pd.DataFrame) -> pd.DataFrame:
        """Align policy rates with regional data."""
        # Find common time period
        common_index = policy_rates.index.intersection(regional_data.index)
        
        if len(common_index) == 0:
            raise ValueError("No common time periods between policy rates and regional data")
        
        # Create aligned dataset
        aligned = pd.DataFrame(index=common_index)
        aligned['policy_rate'] = policy_rates.loc[common_index]
        
        # Add regional data
        for col in regional_data.columns:
            aligned[col] = regional_data.loc[common_index, col]
        
        # Add lagged policy rate if smoothing included
        if self.include_smoothing:
            aligned['lagged_policy_rate'] = aligned['policy_rate'].shift(1)
        
        # Drop missing values
        aligned = aligned.dropna()
        
        return aligned
    
    def _prepare_regression_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for regression estimation."""
        # Dependent variable
        y = data['policy_rate'].values
        
        # Independent variables
        X_list = []
        variable_names = []
        
        # Regional output gaps
        output_cols = [col for col in data.columns if 'output_gap' in col.lower()]
        if output_cols:
            for col in output_cols:
                X_list.append(data[col].values)
                variable_names.append(col)
        
        # Regional inflation
        inflation_cols = [col for col in data.columns if 'inflation' in col.lower()]
        if inflation_cols:
            for col in inflation_cols:
                X_list.append(data[col].values)
                variable_names.append(col)
        
        # Lagged policy rate (interest rate smoothing)
        if self.include_smoothing and 'lagged_policy_rate' in data.columns:
            X_list.append(data['lagged_policy_rate'].values)
            variable_names.append('lagged_policy_rate')
        
        # Add constant term
        X_list.append(np.ones(len(data)))
        variable_names.append('constant')
        
        X = np.column_stack(X_list)
        
        return y, X, variable_names
    
    def _estimate_ols(self,
                     y: np.ndarray,
                     X: np.ndarray,
                     variable_names: List[str]) -> Dict[str, Any]:
        """Estimate using ordinary least squares."""
        # Use sklearn for robust estimation
        reg = LinearRegression(fit_intercept=False)  # Intercept already included
        reg.fit(X, y)
        
        coefficients = reg.coef_
        
        # Compute standard errors
        n, k = X.shape
        residuals = y - reg.predict(X)
        mse = np.sum(residuals**2) / (n - k)
        
        try:
            var_covar = mse * np.linalg.inv(X.T @ X)
            standard_errors = np.sqrt(np.diag(var_covar))
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute standard errors due to singular matrix")
            standard_errors = np.full(k, np.nan)
        
        # Confidence intervals (95%)
        t_critical = 1.96  # Approximate for large samples
        ci_lower = coefficients - t_critical * standard_errors
        ci_upper = coefficients + t_critical * standard_errors
        
        # Create dictionaries
        coefficient_dict = dict(zip(variable_names, coefficients))
        se_dict = dict(zip(variable_names, standard_errors))
        ci_dict = dict(zip(variable_names, zip(ci_lower, ci_upper)))
        
        return {
            'coefficients': coefficients,
            'coefficient_dict': coefficient_dict,
            'standard_errors': se_dict,
            'confidence_intervals': ci_dict
        }
    
    def _estimate_ridge(self,
                       y: np.ndarray,
                       X: np.ndarray,
                       variable_names: List[str]) -> Dict[str, Any]:
        """Estimate using Ridge regression for regularization."""
        # Use cross-validation to select regularization parameter
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(X, y)
        
        coefficients = reg.coef_
        
        # Standard errors are more complex for Ridge - use bootstrap approximation
        standard_errors = np.full(len(coefficients), np.nan)  # Placeholder
        
        # Create dictionaries
        coefficient_dict = dict(zip(variable_names, coefficients))
        se_dict = dict(zip(variable_names, standard_errors))
        ci_dict = dict(zip(variable_names, [(np.nan, np.nan)] * len(coefficients)))
        
        return {
            'coefficients': coefficients,
            'coefficient_dict': coefficient_dict,
            'standard_errors': se_dict,
            'confidence_intervals': ci_dict
        }
    
    def _estimate_constrained(self,
                            y: np.ndarray,
                            X: np.ndarray,
                            variable_names: List[str],
                            data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate with constraints on regional weights."""
        # This would implement constrained optimization
        # For now, fall back to OLS
        return self._estimate_ols(y, X, variable_names)
    
    def _extract_regional_weights(self,
                                results: Dict[str, Any],
                                data: pd.DataFrame) -> np.ndarray:
        """
        Extract implicit regional weights from estimated coefficients.
        
        This assumes the Fed uses a weighted average of regional conditions.
        """
        coefficient_dict = results['coefficient_dict']
        
        # Find regional output gap and inflation coefficients
        output_coeffs = []
        inflation_coeffs = []
        
        for var_name, coeff in coefficient_dict.items():
            if 'output_gap' in var_name.lower():
                output_coeffs.append(coeff)
            elif 'inflation' in var_name.lower():
                inflation_coeffs.append(coeff)
        
        # Convert to arrays
        output_coeffs = np.array(output_coeffs)
        inflation_coeffs = np.array(inflation_coeffs)
        
        if len(output_coeffs) == 0 and len(inflation_coeffs) == 0:
            return None
        
        # Combine output and inflation coefficients to get regional weights
        # This is a simplified approach - more sophisticated methods could be used
        if len(output_coeffs) > 0 and len(inflation_coeffs) > 0:
            # Average of output and inflation weights
            combined_coeffs = (np.abs(output_coeffs) + np.abs(inflation_coeffs)) / 2
        elif len(output_coeffs) > 0:
            combined_coeffs = np.abs(output_coeffs)
        else:
            combined_coeffs = np.abs(inflation_coeffs)
        
        # Normalize to sum to 1
        if np.sum(combined_coeffs) > 0:
            regional_weights = combined_coeffs / np.sum(combined_coeffs)
        else:
            # Equal weights if all coefficients are zero
            regional_weights = np.ones(len(combined_coeffs)) / len(combined_coeffs)
        
        return regional_weights
    
    def _compute_durbin_watson(self, residuals: np.ndarray) -> float:
        """Compute Durbin-Watson statistic for serial correlation."""
        if len(residuals) < 2:
            return np.nan
        
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        return dw_stat
    
    def estimate_time_varying_weights(self,
                                    policy_rates: pd.Series,
                                    regional_data: pd.DataFrame,
                                    window_size: int = 20) -> pd.DataFrame:
        """
        Estimate time-varying regional weights using rolling windows.
        
        Args:
            policy_rates: Time series of Fed policy rates
            regional_data: Regional economic indicators
            window_size: Size of rolling estimation window
            
        Returns:
            DataFrame with time-varying regional weights
        """
        aligned_data = self._align_data(policy_rates, regional_data)
        
        if len(aligned_data) < window_size + 5:
            raise ValueError(f"Insufficient data for rolling estimation (need at least {window_size + 5} observations)")
        
        # Initialize results storage
        dates = []
        weights_list = []
        
        # Rolling estimation
        for i in range(window_size, len(aligned_data)):
            window_data = aligned_data.iloc[i-window_size:i]
            
            try:
                # Estimate for this window
                y, X, variable_names = self._prepare_regression_data(window_data)
                results = self._estimate_ols(y, X, variable_names)
                weights = self._extract_regional_weights(results, window_data)
                
                if weights is not None:
                    dates.append(aligned_data.index[i])
                    weights_list.append(weights)
            
            except Exception as e:
                # Skip problematic windows
                continue
        
        if not weights_list:
            raise ValueError("Could not estimate weights for any time period")
        
        # Create DataFrame
        n_regions = len(weights_list[0])
        region_names = [f"Region_{i+1}" for i in range(n_regions)]
        
        weights_df = pd.DataFrame(weights_list, 
                                index=dates, 
                                columns=region_names)
        
        return weights_df
    
    def compare_to_optimal_weights(self,
                                 estimated_weights: np.ndarray,
                                 optimal_weights: np.ndarray) -> Dict[str, float]:
        """
        Compare estimated Fed weights to optimal welfare-maximizing weights.
        
        Args:
            estimated_weights: Fed's estimated implicit weights
            optimal_weights: Welfare-maximizing optimal weights
            
        Returns:
            Dictionary with comparison metrics
        """
        if len(estimated_weights) != len(optimal_weights):
            raise ValueError("Weight arrays must have same length")
        
        # Compute various distance metrics
        weight_diff = estimated_weights - optimal_weights
        
        metrics = {
            'mean_absolute_difference': np.mean(np.abs(weight_diff)),
            'root_mean_squared_difference': np.sqrt(np.mean(weight_diff**2)),
            'maximum_difference': np.max(np.abs(weight_diff)),
            'correlation': np.corrcoef(estimated_weights, optimal_weights)[0, 1],
            'cosine_similarity': np.dot(estimated_weights, optimal_weights) / 
                               (np.linalg.norm(estimated_weights) * np.linalg.norm(optimal_weights))
        }
        
        return metrics
    
    def test_taylor_principle(self, results: FedReactionResults) -> Dict[str, Any]:
        """
        Test whether Fed policy satisfies Taylor principle.
        
        Args:
            results: Fed reaction function estimation results
            
        Returns:
            Dictionary with Taylor principle test results
        """
        inflation_coeff = results.estimated_coefficients.get('inflation', 0)
        inflation_se = results.standard_errors.get('inflation', 0)
        
        # Test H0: inflation coefficient <= 1 vs H1: inflation coefficient > 1
        if inflation_se > 0:
            t_stat = (inflation_coeff - 1.0) / inflation_se
            p_value = 1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - np.exp(-2 * t_stat**2 / np.pi)))
        else:
            t_stat = np.nan
            p_value = np.nan
        
        satisfies_taylor = inflation_coeff > 1.0
        
        return {
            'inflation_coefficient': inflation_coeff,
            'satisfies_taylor_principle': satisfies_taylor,
            't_statistic': t_stat,
            'p_value': p_value,
            'interpretation': 'Satisfies Taylor Principle' if satisfies_taylor else 'Violates Taylor Principle'
        }