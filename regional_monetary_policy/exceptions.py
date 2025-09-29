"""
Exception classes for regional monetary policy analysis system.

This module defines the exception hierarchy and error handling framework
for the regional monetary policy analysis system.
"""

from typing import Optional, Dict, Any, List


class RegionalMonetaryPolicyError(Exception):
    """
    Base exception class for the regional monetary policy system.
    
    All custom exceptions in the system inherit from this base class.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        """String representation of the exception."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        
        return base_msg


class DataRetrievalError(RegionalMonetaryPolicyError):
    """
    Raised when FRED API data retrieval fails.
    
    This exception is raised for various data acquisition issues including
    API failures, network problems, authentication errors, and data format issues.
    """
    
    def __init__(self, message: str, series_code: Optional[str] = None,
                 api_response_code: Optional[int] = None, **kwargs):
        """
        Initialize data retrieval error.
        
        Args:
            message: Error message
            series_code: FRED series code that failed
            api_response_code: HTTP response code from FRED API
        """
        context = kwargs.get('context', {})
        if series_code:
            context['series_code'] = series_code
        if api_response_code:
            context['api_response_code'] = api_response_code
        
        super().__init__(message, error_code="DATA_RETRIEVAL", context=context)


class DataValidationError(RegionalMonetaryPolicyError):
    """
    Raised when data validation fails.
    
    This exception is raised when data quality checks fail, including
    missing values, outliers, inconsistent dimensions, or invalid data ranges.
    """
    
    def __init__(self, message: str, validation_failures: Optional[List[str]] = None, **kwargs):
        """
        Initialize data validation error.
        
        Args:
            message: Error message
            validation_failures: List of specific validation failures
        """
        context = kwargs.get('context', {})
        if validation_failures:
            context['validation_failures'] = validation_failures
        
        super().__init__(message, error_code="DATA_VALIDATION", context=context)


class EstimationError(RegionalMonetaryPolicyError):
    """
    Raised when parameter estimation fails.
    
    This exception is raised for various estimation issues including
    convergence failures, numerical instability, and invalid parameter values.
    """
    
    def __init__(self, message: str, estimation_stage: Optional[str] = None,
                 convergence_info: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize estimation error.
        
        Args:
            message: Error message
            estimation_stage: Which estimation stage failed (stage_1, stage_2, stage_3)
            convergence_info: Information about convergence failure
        """
        context = kwargs.get('context', {})
        if estimation_stage:
            context['estimation_stage'] = estimation_stage
        if convergence_info:
            context.update(convergence_info)
        
        super().__init__(message, error_code="ESTIMATION_FAILURE", context=context)


class IdentificationError(EstimationError):
    """
    Raised when parameters are not identified.
    
    This exception is raised when identification tests fail or when
    the model parameters cannot be uniquely determined from the data.
    """
    
    def __init__(self, message: str, identification_tests: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialize identification error.
        
        Args:
            message: Error message
            identification_tests: Results of identification tests
        """
        context = kwargs.get('context', {})
        if identification_tests:
            context['identification_tests'] = identification_tests
        
        super().__init__(message, error_code="IDENTIFICATION_FAILURE", context=context)


class SpatialModelError(RegionalMonetaryPolicyError):
    """
    Raised when spatial model specification is invalid.
    
    This exception is raised for spatial weight matrix issues, including
    invalid dimensions, non-positive definiteness, or specification errors.
    """
    
    def __init__(self, message: str, matrix_properties: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize spatial model error.
        
        Args:
            message: Error message
            matrix_properties: Properties of the problematic spatial matrix
        """
        context = kwargs.get('context', {})
        if matrix_properties:
            context['matrix_properties'] = matrix_properties
        
        super().__init__(message, error_code="SPATIAL_MODEL", context=context)


class WelfareCalculationError(RegionalMonetaryPolicyError):
    """
    Raised when welfare calculations fail.
    
    This exception is raised for issues in welfare computation, policy
    mistake decomposition, or counterfactual analysis.
    """
    
    def __init__(self, message: str, calculation_type: Optional[str] = None, **kwargs):
        """
        Initialize welfare calculation error.
        
        Args:
            message: Error message
            calculation_type: Type of welfare calculation that failed
        """
        context = kwargs.get('context', {})
        if calculation_type:
            context['calculation_type'] = calculation_type
        
        super().__init__(message, error_code="WELFARE_CALCULATION", context=context)


class ConfigurationError(RegionalMonetaryPolicyError):
    """
    Raised when configuration is invalid or incomplete.
    
    This exception is raised for configuration validation failures,
    missing required settings, or incompatible parameter combinations.
    """
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            validation_errors: List of configuration validation errors
        """
        context = kwargs.get('context', {})
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(message, error_code="CONFIGURATION", context=context)


class NumericalError(RegionalMonetaryPolicyError):
    """
    Raised when numerical computations fail.
    
    This exception is raised for numerical issues including matrix
    singularity, overflow/underflow, or convergence problems.
    """
    
    def __init__(self, message: str, numerical_details: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize numerical error.
        
        Args:
            message: Error message
            numerical_details: Details about the numerical problem
        """
        context = kwargs.get('context', {})
        if numerical_details:
            context.update(numerical_details)
        
        super().__init__(message, error_code="NUMERICAL", context=context)


class APIRateLimitError(DataRetrievalError):
    """
    Raised when FRED API rate limits are exceeded.
    
    This exception is raised specifically for API rate limiting issues
    and includes information about retry timing.
    """
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        """
        Initialize API rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
        """
        context = kwargs.get('context', {})
        if retry_after:
            context['retry_after'] = retry_after
        
        super().__init__(message, error_code="API_RATE_LIMIT", context=context)


class InsufficientDataError(DataValidationError):
    """
    Raised when there is insufficient data for analysis.
    
    This exception is raised when the available data is insufficient
    for reliable parameter estimation or analysis.
    """
    
    def __init__(self, message: str, required_periods: Optional[int] = None,
                 available_periods: Optional[int] = None, **kwargs):
        """
        Initialize insufficient data error.
        
        Args:
            message: Error message
            required_periods: Minimum required time periods
            available_periods: Actually available time periods
        """
        context = kwargs.get('context', {})
        if required_periods:
            context['required_periods'] = required_periods
        if available_periods:
            context['available_periods'] = available_periods
        
        super().__init__(message, error_code="INSUFFICIENT_DATA", context=context)


# Error handling utilities

class ErrorHandler:
    """
    Utility class for consistent error handling and logging.
    """
    
    def __init__(self, logger=None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error with appropriate logging and context.
        
        Args:
            error: Exception to handle
            context: Additional context information
        """
        if self.logger:
            error_msg = str(error)
            if context:
                context_str = ", ".join(f"{k}={v}" for k, v in context.items())
                error_msg += f" (Additional context: {context_str})"
            
            if isinstance(error, RegionalMonetaryPolicyError):
                self.logger.error(f"System error: {error_msg}")
            else:
                self.logger.error(f"Unexpected error: {error_msg}")
    
    def wrap_api_call(self, func, *args, **kwargs):
        """
        Wrap an API call with error handling.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            DataRetrievalError: For API-related failures
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"FRED API rate limit exceeded: {e}")
            elif "authentication" in str(e).lower():
                raise DataRetrievalError(f"FRED API authentication failed: {e}")
            else:
                raise DataRetrievalError(f"FRED API call failed: {e}")
    
    def wrap_estimation(self, func, *args, **kwargs):
        """
        Wrap an estimation procedure with error handling.
        
        Args:
            func: Estimation function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Estimation result
            
        Raises:
            EstimationError: For estimation failures
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "convergence" in str(e).lower():
                raise EstimationError(f"Estimation failed to converge: {e}")
            elif "singular" in str(e).lower() or "invertible" in str(e).lower():
                raise NumericalError(f"Numerical singularity in estimation: {e}")
            else:
                raise EstimationError(f"Estimation procedure failed: {e}")


def create_error_context(**kwargs) -> Dict[str, Any]:
    """
    Create error context dictionary with non-None values.
    
    Args:
        **kwargs: Context key-value pairs
        
    Returns:
        Dictionary with non-None values
    """
    return {k: v for k, v in kwargs.items() if v is not None}