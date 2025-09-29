"""
Error recovery and resilience system for regional monetary policy analysis.

This module provides comprehensive error recovery mechanisms, retry logic,
and graceful degradation strategies for various system components.
"""

import time
import random
from typing import Callable, Any, Optional, Dict, List, Union, Type
from functools import wraps
import numpy as np
import pandas as pd
from contextlib import contextmanager

from .exceptions import (
    RegionalMonetaryPolicyError, DataRetrievalError, APIRateLimitError,
    EstimationError, NumericalError, DataValidationError, InsufficientDataError
)
from .logging_config import get_logger, ErrorLogger


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 25% jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay


class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str):
        """
        Initialize recovery strategy.
        
        Args:
            name: Name of the recovery strategy
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_logger = ErrorLogger(self.logger)
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Check if this strategy can recover from the given error.
        
        Args:
            error: Exception to check
            context: Error context
            
        Returns:
            True if recovery is possible
        """
        raise NotImplementedError
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """
        Attempt to recover from the error.
        
        Args:
            error: Exception to recover from
            context: Error context
            
        Returns:
            Recovery result or raises exception if recovery fails
        """
        raise NotImplementedError


class APIRetryStrategy(RecoveryStrategy):
    """Recovery strategy for API-related errors."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """
        Initialize API retry strategy.
        
        Args:
            retry_config: Retry configuration
        """
        super().__init__("API_RETRY")
        self.retry_config = retry_config or RetryConfig()
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if API error can be recovered."""
        return isinstance(error, (DataRetrievalError, APIRateLimitError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from API error."""
        if not self.can_recover(error, context):
            raise error
        
        original_func = context.get('original_function')
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})
        
        if not original_func:
            raise error
        
        for attempt in range(self.retry_config.max_attempts):
            if attempt > 0:
                delay = self.retry_config.get_delay(attempt - 1)
                
                # Special handling for rate limit errors
                if isinstance(error, APIRateLimitError) and hasattr(error, 'retry_after'):
                    delay = max(delay, error.retry_after)
                
                self.error_logger.log_recovery_attempt(
                    error, attempt + 1, self.retry_config.max_attempts,
                    f"retry_after_{delay:.1f}s", delay=delay
                )
                
                time.sleep(delay)
            
            try:
                result = original_func(*args, **kwargs)
                if attempt > 0:
                    self.error_logger.log_recovery_success(
                        error, attempt + 1, self.name
                    )
                return result
            except Exception as retry_error:
                error = retry_error
                if attempt == self.retry_config.max_attempts - 1:
                    self.error_logger.log_recovery_failure(
                        error, self.retry_config.max_attempts, [self.name]
                    )
                    raise
        
        raise error


class CacheRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy using cached data."""
    
    def __init__(self, cache_manager=None):
        """
        Initialize cache recovery strategy.
        
        Args:
            cache_manager: Cache manager instance
        """
        super().__init__("CACHE_FALLBACK")
        self.cache_manager = cache_manager
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if cached data can be used for recovery."""
        return (isinstance(error, DataRetrievalError) and 
                self.cache_manager is not None and
                'series_code' in context)
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover using cached data."""
        if not self.can_recover(error, context):
            raise error
        
        series_code = context['series_code']
        
        try:
            cached_data = self.cache_manager.get_cached_data(series_code)
            if cached_data is not None:
                self.logger.warning(f"Using cached data for {series_code} due to API failure")
                return cached_data
        except Exception as cache_error:
            self.logger.error(f"Cache recovery failed: {cache_error}")
        
        raise error


class EstimationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for estimation failures."""
    
    def __init__(self):
        """Initialize estimation recovery strategy."""
        super().__init__("ESTIMATION_RECOVERY")
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if estimation error can be recovered."""
        return isinstance(error, (EstimationError, NumericalError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from estimation error."""
        if not self.can_recover(error, context):
            raise error
        
        # Try different recovery strategies based on error type
        if isinstance(error, NumericalError):
            return self._recover_numerical_error(error, context)
        elif isinstance(error, EstimationError):
            return self._recover_estimation_error(error, context)
        
        raise error
    
    def _recover_numerical_error(self, error: NumericalError, context: Dict[str, Any]) -> Any:
        """Recover from numerical errors."""
        original_func = context.get('original_function')
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})
        
        if not original_func:
            raise error
        
        # Try with regularization
        try:
            self.logger.info("Attempting recovery with regularization")
            kwargs_reg = kwargs.copy()
            kwargs_reg['regularization'] = kwargs_reg.get('regularization', 1e-6) * 10
            result = original_func(*args, **kwargs_reg)
            self.error_logger.log_recovery_success(error, 1, "regularization")
            return result
        except Exception:
            pass
        
        # Try with different solver
        try:
            self.logger.info("Attempting recovery with alternative solver")
            kwargs_solver = kwargs.copy()
            kwargs_solver['solver'] = 'robust'
            result = original_func(*args, **kwargs_solver)
            self.error_logger.log_recovery_success(error, 1, "alternative_solver")
            return result
        except Exception:
            pass
        
        raise error
    
    def _recover_estimation_error(self, error: EstimationError, context: Dict[str, Any]) -> Any:
        """Recover from estimation errors."""
        original_func = context.get('original_function')
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})
        
        if not original_func:
            raise error
        
        # Try with relaxed convergence criteria
        try:
            self.logger.info("Attempting recovery with relaxed convergence criteria")
            kwargs_relaxed = kwargs.copy()
            kwargs_relaxed['tolerance'] = kwargs_relaxed.get('tolerance', 1e-6) * 10
            kwargs_relaxed['max_iterations'] = kwargs_relaxed.get('max_iterations', 1000) * 2
            result = original_func(*args, **kwargs_relaxed)
            self.error_logger.log_recovery_success(error, 1, "relaxed_convergence")
            return result
        except Exception:
            pass
        
        raise error


class DataQualityRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for data quality issues."""
    
    def __init__(self):
        """Initialize data quality recovery strategy."""
        super().__init__("DATA_QUALITY_RECOVERY")
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if data quality error can be recovered."""
        return isinstance(error, (DataValidationError, InsufficientDataError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from data quality error."""
        if not self.can_recover(error, context):
            raise error
        
        data = context.get('data')
        if data is None:
            raise error
        
        if isinstance(error, DataValidationError):
            return self._recover_validation_error(error, data, context)
        elif isinstance(error, InsufficientDataError):
            return self._recover_insufficient_data(error, data, context)
        
        raise error
    
    def _recover_validation_error(self, error: DataValidationError, 
                                data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Recover from data validation errors."""
        self.logger.info("Attempting data quality recovery")
        
        # Handle missing values
        if data.isnull().any().any():
            # Forward fill then backward fill
            data_cleaned = data.fillna(method='ffill').fillna(method='bfill')
            
            # If still missing, interpolate
            if data_cleaned.isnull().any().any():
                data_cleaned = data_cleaned.interpolate(method='linear')
            
            # If still missing, use median
            if data_cleaned.isnull().any().any():
                data_cleaned = data_cleaned.fillna(data_cleaned.median())
        else:
            data_cleaned = data.copy()
        
        # Handle outliers (replace with 99th percentile values)
        for col in data_cleaned.select_dtypes(include=[np.number]).columns:
            q99 = data_cleaned[col].quantile(0.99)
            q01 = data_cleaned[col].quantile(0.01)
            data_cleaned[col] = data_cleaned[col].clip(lower=q01, upper=q99)
        
        self.error_logger.log_recovery_success(error, 1, "data_cleaning")
        return data_cleaned
    
    def _recover_insufficient_data(self, error: InsufficientDataError,
                                 data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Recover from insufficient data errors."""
        # Try to extend data range or use alternative frequency
        self.logger.info("Attempting to recover from insufficient data")
        
        # This would typically involve requesting more data or using different parameters
        # For now, we'll raise the original error as this requires external data
        raise error


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and coordinates recovery attempts.
    """
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.strategies: List[RecoveryStrategy] = []
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_logger = ErrorLogger(self.logger)
        
        # Register default strategies
        self.register_strategy(APIRetryStrategy())
        self.register_strategy(EstimationRecoveryStrategy())
        self.register_strategy(DataQualityRecoveryStrategy())
    
    def register_strategy(self, strategy: RecoveryStrategy):
        """
        Register a recovery strategy.
        
        Args:
            strategy: Recovery strategy to register
        """
        self.strategies.append(strategy)
        self.logger.debug(f"Registered recovery strategy: {strategy.name}")
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """
        Attempt to recover from an error using available strategies.
        
        Args:
            error: Exception to recover from
            context: Error context
            
        Returns:
            Recovery result
            
        Raises:
            Original exception if recovery fails
        """
        applicable_strategies = [s for s in self.strategies if s.can_recover(error, context)]
        
        if not applicable_strategies:
            self.logger.debug(f"No recovery strategies available for {type(error).__name__}")
            raise error
        
        self.logger.info(f"Attempting recovery for {type(error).__name__} using {len(applicable_strategies)} strategies")
        
        for strategy in applicable_strategies:
            try:
                result = strategy.recover(error, context)
                self.logger.info(f"Successfully recovered using strategy: {strategy.name}")
                return result
            except Exception as recovery_error:
                self.logger.debug(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                continue
        
        # All strategies failed
        strategy_names = [s.name for s in applicable_strategies]
        self.error_logger.log_recovery_failure(error, len(applicable_strategies), strategy_names)
        raise error


# Global recovery manager instance
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """
    Get the global error recovery manager.
    
    Returns:
        ErrorRecoveryManager instance
    """
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


def with_recovery(func: Callable = None, *, 
                 recovery_context: Optional[Dict[str, Any]] = None,
                 enable_recovery: bool = True):
    """
    Decorator to add error recovery to functions.
    
    Args:
        func: Function to decorate
        recovery_context: Additional context for recovery
        enable_recovery: Whether to enable recovery
        
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not enable_recovery:
                return f(*args, **kwargs)
            
            try:
                return f(*args, **kwargs)
            except Exception as error:
                context = {
                    'original_function': f,
                    'args': args,
                    'kwargs': kwargs,
                    'function_name': f.__name__,
                    'module': f.__module__
                }
                
                if recovery_context:
                    context.update(recovery_context)
                
                recovery_manager = get_recovery_manager()
                return recovery_manager.recover(error, context)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def error_recovery_context(**context):
    """
    Context manager for error recovery.
    
    Args:
        **context: Context information for recovery
    """
    try:
        yield
    except Exception as error:
        recovery_manager = get_recovery_manager()
        recovery_manager.recover(error, context)