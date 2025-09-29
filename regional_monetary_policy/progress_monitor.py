"""
Progress monitoring and performance tracking system for regional monetary policy analysis.

This module provides comprehensive progress tracking, performance monitoring,
and user feedback for long-running operations.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import psutil
import os
from collections import deque
import json

from .logging_config import get_logger, get_performance_logger


@dataclass
class ProgressMetrics:
    """Metrics for tracking operation progress."""
    
    operation_name: str
    start_time: datetime
    current_step: int = 0
    total_steps: int = 0
    current_phase: str = ""
    status: str = "running"  # running, completed, failed, cancelled
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)
    
    @property
    def estimated_remaining_time(self) -> Optional[timedelta]:
        """Estimate remaining time based on current progress."""
        if self.current_step == 0 or self.total_steps == 0:
            return None
        
        elapsed = self.elapsed_time.total_seconds()
        progress_ratio = self.current_step / self.total_steps
        
        if progress_ratio > 0:
            total_estimated = elapsed / progress_ratio
            remaining = total_estimated - elapsed
            return timedelta(seconds=max(0, remaining))
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_phase': self.current_phase,
            'status': self.status,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'elapsed_seconds': self.elapsed_time.total_seconds(),
            'progress_percentage': self.progress_percentage,
            'estimated_remaining_seconds': self.estimated_remaining_time.total_seconds() if self.estimated_remaining_time else None,
            'metadata': self.metadata
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent
        }


class ProgressTracker:
    """
    Tracks progress of individual operations.
    """
    
    def __init__(self, operation_name: str, total_steps: int = 0):
        """
        Initialize progress tracker.
        
        Args:
            operation_name: Name of the operation being tracked
            total_steps: Total number of steps (0 for indeterminate progress)
        """
        self.metrics = ProgressMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            total_steps=total_steps
        )
        self.logger = get_logger(f"{__name__}.{operation_name}")
        self._callbacks: List[Callable[[ProgressMetrics], None]] = []
        self._cancelled = False
    
    def add_callback(self, callback: Callable[[ProgressMetrics], None]):
        """
        Add a callback function to be called on progress updates.
        
        Args:
            callback: Function that takes ProgressMetrics as argument
        """
        self._callbacks.append(callback)
    
    def update(self, step: Optional[int] = None, phase: Optional[str] = None, 
              increment: bool = False, **metadata):
        """
        Update progress information.
        
        Args:
            step: Current step number
            phase: Current phase description
            increment: Whether to increment current step by 1
            **metadata: Additional metadata to store
        """
        if self._cancelled:
            raise InterruptedError("Operation was cancelled")
        
        if increment:
            self.metrics.current_step += 1
        elif step is not None:
            self.metrics.current_step = step
        
        if phase is not None:
            self.metrics.current_phase = phase
        
        if metadata:
            self.metrics.metadata.update(metadata)
        
        # Log progress
        if self.metrics.total_steps > 0:
            self.logger.info(f"Progress: {self.metrics.current_step}/{self.metrics.total_steps} "
                           f"({self.metrics.progress_percentage:.1f}%) - {self.metrics.current_phase}")
        else:
            self.logger.info(f"Progress: Step {self.metrics.current_step} - {self.metrics.current_phase}")
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
    
    def set_total_steps(self, total_steps: int):
        """
        Set or update the total number of steps.
        
        Args:
            total_steps: Total number of steps
        """
        self.metrics.total_steps = total_steps
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """
        Mark operation as completed.
        
        Args:
            success: Whether operation completed successfully
            error_message: Error message if operation failed
        """
        self.metrics.end_time = datetime.now()
        self.metrics.status = "completed" if success else "failed"
        self.metrics.error_message = error_message
        
        if success:
            self.logger.info(f"Operation '{self.metrics.operation_name}' completed successfully "
                           f"in {self.metrics.elapsed_time}")
        else:
            self.logger.error(f"Operation '{self.metrics.operation_name}' failed "
                            f"after {self.metrics.elapsed_time}: {error_message}")
        
        # Final callback
        for callback in self._callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
    
    def cancel(self):
        """Cancel the operation."""
        self._cancelled = True
        self.metrics.status = "cancelled"
        self.metrics.end_time = datetime.now()
        self.logger.warning(f"Operation '{self.metrics.operation_name}' was cancelled")
    
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self._cancelled


class SystemMonitor:
    """
    Monitors system performance metrics.
    """
    
    def __init__(self, update_interval: float = 5.0, history_size: int = 100):
        """
        Initialize system monitor.
        
        Args:
            update_interval: Seconds between metric updates
            history_size: Number of historical metrics to keep
        """
        self.update_interval = update_interval
        self.history: deque = deque(maxlen=history_size)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SystemMetrics], None]] = []
    
    def add_callback(self, callback: Callable[[SystemMetrics], None]):
        """
        Add callback for system metric updates.
        
        Args:
            callback: Function that takes SystemMetrics as argument
        """
        self._callbacks.append(callback)
    
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.update_interval + 1)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.history.append(metrics)
                
                # Call callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"System monitor callback failed: {e}")
                
                time.sleep(self.update_interval)
            
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage for current directory
        try:
            disk = psutil.disk_usage(os.getcwd())
            disk_percent = (disk.used / disk.total) * 100
        except Exception:
            disk_percent = 0.0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024**2,
            memory_available_mb=memory.available / 1024**2,
            disk_usage_percent=disk_percent
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.history[-1] if self.history else None
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[Dict[str, float]]:
        """
        Get average metrics over specified time period.
        
        Args:
            minutes: Number of minutes to average over
            
        Returns:
            Dictionary with average metrics
        """
        if not self.history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        return {
            'cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'memory_used_mb': sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
            'disk_usage_percent': sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
        }


class ProgressMonitor:
    """
    Central progress monitoring system that coordinates multiple operations.
    """
    
    def __init__(self, enable_system_monitoring: bool = True):
        """
        Initialize progress monitor.
        
        Args:
            enable_system_monitoring: Whether to enable system performance monitoring
        """
        self.active_operations: Dict[str, ProgressTracker] = {}
        self.completed_operations: List[ProgressMetrics] = []
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.perf_logger = get_performance_logger(f"{__name__}.{self.__class__.__name__}")
        
        # System monitoring
        self.system_monitor = SystemMonitor() if enable_system_monitoring else None
        if self.system_monitor:
            self.system_monitor.start_monitoring()
    
    def create_tracker(self, operation_name: str, total_steps: int = 0) -> ProgressTracker:
        """
        Create a new progress tracker for an operation.
        
        Args:
            operation_name: Name of the operation
            total_steps: Total number of steps
            
        Returns:
            ProgressTracker instance
        """
        # Ensure unique operation name
        base_name = operation_name
        counter = 1
        while operation_name in self.active_operations:
            operation_name = f"{base_name}_{counter}"
            counter += 1
        
        tracker = ProgressTracker(operation_name, total_steps)
        
        # Add callback to handle completion
        def completion_callback(metrics: ProgressMetrics):
            if metrics.status in ['completed', 'failed', 'cancelled']:
                self._handle_operation_completion(operation_name, metrics)
        
        tracker.add_callback(completion_callback)
        
        self.active_operations[operation_name] = tracker
        self.logger.info(f"Created progress tracker for '{operation_name}'")
        
        return tracker
    
    def _handle_operation_completion(self, operation_name: str, metrics: ProgressMetrics):
        """Handle completion of an operation."""
        if operation_name in self.active_operations:
            del self.active_operations[operation_name]
            self.completed_operations.append(metrics)
            
            # Keep only last 50 completed operations
            if len(self.completed_operations) > 50:
                self.completed_operations = self.completed_operations[-50:]
    
    def get_operation_status(self, operation_name: str) -> Optional[ProgressMetrics]:
        """
        Get status of a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            ProgressMetrics if operation exists, None otherwise
        """
        tracker = self.active_operations.get(operation_name)
        return tracker.metrics if tracker else None
    
    def get_all_active_operations(self) -> Dict[str, ProgressMetrics]:
        """Get status of all active operations."""
        return {name: tracker.metrics for name, tracker in self.active_operations.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            'active_operations': len(self.active_operations),
            'completed_operations': len(self.completed_operations),
            'system_metrics': None
        }
        
        if self.system_monitor:
            current_metrics = self.system_monitor.get_current_metrics()
            if current_metrics:
                status['system_metrics'] = current_metrics.to_dict()
            
            avg_metrics = self.system_monitor.get_average_metrics(5)
            if avg_metrics:
                status['average_system_metrics'] = avg_metrics
        
        return status
    
    def cancel_operation(self, operation_name: str) -> bool:
        """
        Cancel a specific operation.
        
        Args:
            operation_name: Name of the operation to cancel
            
        Returns:
            True if operation was cancelled, False if not found
        """
        tracker = self.active_operations.get(operation_name)
        if tracker:
            tracker.cancel()
            return True
        return False
    
    def cancel_all_operations(self):
        """Cancel all active operations."""
        for tracker in self.active_operations.values():
            tracker.cancel()
    
    def shutdown(self):
        """Shutdown the progress monitor."""
        self.cancel_all_operations()
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
        self.logger.info("Progress monitor shutdown complete")
    
    @contextmanager
    def track_operation(self, operation_name: str, total_steps: int = 0):
        """
        Context manager for tracking an operation.
        
        Args:
            operation_name: Name of the operation
            total_steps: Total number of steps
        """
        tracker = self.create_tracker(operation_name, total_steps)
        try:
            yield tracker
            tracker.complete(success=True)
        except Exception as e:
            tracker.complete(success=False, error_message=str(e))
            raise


# Global progress monitor instance
_progress_monitor: Optional[ProgressMonitor] = None


def get_progress_monitor() -> ProgressMonitor:
    """
    Get the global progress monitor instance.
    
    Returns:
        ProgressMonitor instance
    """
    global _progress_monitor
    if _progress_monitor is None:
        _progress_monitor = ProgressMonitor()
    return _progress_monitor


def track_progress(operation_name: str, total_steps: int = 0):
    """
    Decorator for tracking function progress.
    
    Args:
        operation_name: Name of the operation
        total_steps: Total number of steps
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_progress_monitor()
            with monitor.track_operation(operation_name, total_steps) as tracker:
                # Add tracker to function kwargs if it accepts it
                import inspect
                sig = inspect.signature(func)
                if 'progress_tracker' in sig.parameters:
                    kwargs['progress_tracker'] = tracker
                
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ProgressReporter:
    """
    Formats and reports progress information for different output formats.
    """
    
    def __init__(self, monitor: Optional[ProgressMonitor] = None):
        """
        Initialize progress reporter.
        
        Args:
            monitor: ProgressMonitor instance (uses global if None)
        """
        self.monitor = monitor or get_progress_monitor()
    
    def get_text_report(self) -> str:
        """Get a text-based progress report."""
        lines = ["=== Progress Report ==="]
        
        # Active operations
        active_ops = self.monitor.get_all_active_operations()
        if active_ops:
            lines.append("\nActive Operations:")
            for name, metrics in active_ops.items():
                progress_str = f"{metrics.progress_percentage:.1f}%" if metrics.total_steps > 0 else "Running"
                elapsed = str(metrics.elapsed_time).split('.')[0]  # Remove microseconds
                lines.append(f"  • {name}: {progress_str} ({elapsed}) - {metrics.current_phase}")
        else:
            lines.append("\nNo active operations")
        
        # System status
        system_status = self.monitor.get_system_status()
        if system_status.get('system_metrics'):
            metrics = system_status['system_metrics']
            lines.append(f"\nSystem Status:")
            lines.append(f"  • CPU: {metrics['cpu_percent']:.1f}%")
            lines.append(f"  • Memory: {metrics['memory_percent']:.1f}% ({metrics['memory_used_mb']:.0f} MB)")
            lines.append(f"  • Disk: {metrics['disk_usage_percent']:.1f}%")
        
        return "\n".join(lines)
    
    def get_json_report(self) -> str:
        """Get a JSON-formatted progress report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'active_operations': {name: metrics.to_dict() 
                                for name, metrics in self.monitor.get_all_active_operations().items()},
            'system_status': self.monitor.get_system_status()
        }
        return json.dumps(report, indent=2)
    
    def print_status(self):
        """Print current status to console."""
        print(self.get_text_report())