"""
Comprehensive logging configuration for regional monetary policy analysis system.

This module provides centralized logging configuration with different levels,
formatters, and handlers for various components of the system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import traceback
from contextlib import contextmanager

from .exceptions import RegionalMonetaryPolicyError


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """
    Filter to add contextual information to log records.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize context filter.
        
        Args:
            context: Dictionary of context information to add to all records
        """
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context information to log record.
        
        Args:
            record: Log record to modify
            
        Returns:
            True to allow record through
        """
        if not hasattr(record, 'extra_fields'):
            record.extra_fields = {}
        
        record.extra_fields.update(self.context)
        return True


class PerformanceLogger:
    """
    Logger for performance monitoring and timing information.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self._timers: Dict[str, float] = {}
    
    @contextmanager
    def timer(self, operation_name: str, **context):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            **context: Additional context information
        """
        import time
        
        start_time = time.time()
        self.logger.info(f"Starting {operation_name}", extra={'extra_fields': {
            'operation': operation_name,
            'event': 'start',
            **context
        }})
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.info(f"Completed {operation_name} in {duration:.3f}s", 
                           extra={'extra_fields': {
                               'operation': operation_name,
                               'event': 'complete',
                               'duration_seconds': duration,
                               **context
                           }})
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}", 
                            extra={'extra_fields': {
                                'operation': operation_name,
                                'event': 'error',
                                'duration_seconds': duration,
                                'error': str(e),
                                **context
                            }})
            raise
    
    def log_progress(self, operation: str, current: int, total: int, **context):
        """
        Log progress information for long-running operations.
        
        Args:
            operation: Name of the operation
            current: Current progress count
            total: Total expected count
            **context: Additional context information
        """
        percentage = (current / total) * 100 if total > 0 else 0
        
        self.logger.info(f"Progress {operation}: {current}/{total} ({percentage:.1f}%)",
                        extra={'extra_fields': {
                            'operation': operation,
                            'event': 'progress',
                            'current': current,
                            'total': total,
                            'percentage': percentage,
                            **context
                        }})


class LoggingConfig:
    """
    Centralized logging configuration manager.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: Optional[Union[str, Path]] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (defaults to logs/)
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            enable_json: Whether to use JSON formatting
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        if self.enable_file:
            self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self._configure_root_logger()
    
    def _configure_root_logger(self):
        """Configure the root logger with appropriate handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.enable_json:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
            
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file:
            log_file = self.log_dir / "regional_monetary_policy.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            
            if self.enable_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
            
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """
        Get a logger with optional context.
        
        Args:
            name: Logger name
            context: Context information to add to all log records
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        if context:
            context_filter = ContextFilter(context)
            logger.addFilter(context_filter)
        
        return logger
    
    def get_performance_logger(self, name: str, context: Optional[Dict[str, Any]] = None) -> PerformanceLogger:
        """
        Get a performance logger for timing operations.
        
        Args:
            name: Logger name
            context: Context information
            
        Returns:
            Performance logger instance
        """
        base_logger = self.get_logger(name, context)
        return PerformanceLogger(base_logger)
    
    def configure_component_loggers(self):
        """Configure loggers for specific system components."""
        # Data layer loggers
        data_logger = self.get_logger('regional_monetary_policy.data')
        data_logger.setLevel(self.log_level)
        
        # API logger with more detailed logging
        api_logger = self.get_logger('regional_monetary_policy.data.fred_client')
        api_logger.setLevel(logging.DEBUG if self.log_level <= logging.INFO else self.log_level)
        
        # Econometric layer loggers
        econometric_logger = self.get_logger('regional_monetary_policy.econometric')
        econometric_logger.setLevel(self.log_level)
        
        # Policy analysis loggers
        policy_logger = self.get_logger('regional_monetary_policy.policy')
        policy_logger.setLevel(self.log_level)
        
        # Presentation layer loggers
        presentation_logger = self.get_logger('regional_monetary_policy.presentation')
        presentation_logger.setLevel(self.log_level)


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(log_level: str = "INFO",
                 log_dir: Optional[Union[str, Path]] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = False,
                 **kwargs) -> LoggingConfig:
    """
    Setup global logging configuration.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        enable_json: Whether to use JSON formatting
        **kwargs: Additional configuration options
        
    Returns:
        Configured LoggingConfig instance
    """
    global _logging_config
    
    _logging_config = LoggingConfig(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_json=enable_json,
        **kwargs
    )
    
    _logging_config.configure_component_loggers()
    return _logging_config


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name
        context: Context information
        
    Returns:
        Logger instance
    """
    if _logging_config is None:
        setup_logging()
    
    return _logging_config.get_logger(name, context)


def get_performance_logger(name: str, context: Optional[Dict[str, Any]] = None) -> PerformanceLogger:
    """
    Get a performance logger instance.
    
    Args:
        name: Logger name
        context: Context information
        
    Returns:
        Performance logger instance
    """
    if _logging_config is None:
        setup_logging()
    
    return _logging_config.get_performance_logger(name, context)


class ErrorLogger:
    """
    Specialized logger for error handling and recovery.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize error logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                  recovery_action: Optional[str] = None):
        """
        Log an error with detailed information.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            recovery_action: Description of recovery action taken
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
        }
        
        if isinstance(error, RegionalMonetaryPolicyError):
            error_info['error_code'] = error.error_code
            if error.context:
                error_info['error_context'] = error.context
        
        if context:
            error_info['additional_context'] = context
        
        if recovery_action:
            error_info['recovery_action'] = recovery_action
        
        self.logger.error(f"Error occurred: {error}", 
                         extra={'extra_fields': error_info},
                         exc_info=True)
    
    def log_recovery_attempt(self, error: Exception, attempt: int, max_attempts: int,
                           strategy: str, **context):
        """
        Log a recovery attempt.
        
        Args:
            error: Original error
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            strategy: Recovery strategy being used
            **context: Additional context
        """
        self.logger.warning(f"Recovery attempt {attempt}/{max_attempts} for {type(error).__name__}: {strategy}",
                          extra={'extra_fields': {
                              'error_type': type(error).__name__,
                              'attempt': attempt,
                              'max_attempts': max_attempts,
                              'recovery_strategy': strategy,
                              **context
                          }})
    
    def log_recovery_success(self, error: Exception, attempts: int, strategy: str, **context):
        """
        Log successful error recovery.
        
        Args:
            error: Original error that was recovered from
            attempts: Number of attempts required
            strategy: Recovery strategy that succeeded
            **context: Additional context
        """
        self.logger.info(f"Successfully recovered from {type(error).__name__} after {attempts} attempts using {strategy}",
                        extra={'extra_fields': {
                            'error_type': type(error).__name__,
                            'recovery_attempts': attempts,
                            'recovery_strategy': strategy,
                            'recovery_success': True,
                            **context
                        }})
    
    def log_recovery_failure(self, error: Exception, attempts: int, strategies: list, **context):
        """
        Log failed error recovery.
        
        Args:
            error: Original error
            attempts: Total number of attempts made
            strategies: List of strategies attempted
            **context: Additional context
        """
        self.logger.error(f"Failed to recover from {type(error).__name__} after {attempts} attempts",
                         extra={'extra_fields': {
                             'error_type': type(error).__name__,
                             'recovery_attempts': attempts,
                             'recovery_strategies': strategies,
                             'recovery_success': False,
                             **context
                         }})