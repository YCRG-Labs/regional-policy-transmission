"""
Production configuration for performance monitoring and optimization.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .monitoring import SystemMonitor, EmailAlertHandler, LogAlertHandler
from .cache_manager import IntelligentCacheManager
from .memory_manager import MemoryManager
from .optimizer import ComputationOptimizer
from .profiler import PerformanceProfiler

logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Configuration for production performance monitoring."""
    
    # System monitoring
    monitoring_interval: int = 60  # seconds
    enable_alerts: bool = True
    
    # Memory management
    memory_limit_gb: float = 16.0
    enable_disk_cache: bool = True
    
    # Computation optimization
    max_workers: Optional[int] = None
    enable_gpu: bool = False
    
    # Caching
    cache_size_gb: float = 5.0
    default_ttl_hours: int = 24
    enable_compression: bool = True
    
    # Profiling
    enable_profiling: bool = True
    profile_sample_rate: float = 0.1  # Profile 10% of operations
    
    # Alert thresholds
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    disk_warning_threshold: float = 90.0
    disk_critical_threshold: float = 98.0
    
    # Email alerts (optional)
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_emails: List[str] = None
    
    def __post_init__(self):
        if self.alert_emails is None:
            self.alert_emails = []

class ProductionPerformanceManager:
    """Manages performance optimization in production environment."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all performance components."""
        
        logger.info("Initializing production performance components")
        
        # System monitor
        self.components['monitor'] = SystemMonitor(
            monitoring_interval=self.config.monitoring_interval,
            enable_alerts=self.config.enable_alerts
        )
        
        # Set custom thresholds
        monitor = self.components['monitor']
        monitor.set_threshold('cpu_percent', 
                            warning=self.config.cpu_warning_threshold,
                            critical=self.config.cpu_critical_threshold)
        monitor.set_threshold('memory_percent',
                            warning=self.config.memory_warning_threshold, 
                            critical=self.config.memory_critical_threshold)
        monitor.set_threshold('disk_percent',
                            warning=self.config.disk_warning_threshold,
                            critical=self.config.disk_critical_threshold)
        
        # Add alert handlers
        if self.config.enable_alerts:
            # Always add log handler
            monitor.add_alert_handler(LogAlertHandler())
            
            # Add email handler if configured
            if (self.config.smtp_server and 
                self.config.smtp_username and 
                self.config.smtp_password and 
                self.config.alert_emails):
                
                email_handler = EmailAlertHandler(
                    smtp_server=self.config.smtp_server,
                    smtp_port=self.config.smtp_port,
                    username=self.config.smtp_username,
                    password=self.config.smtp_password,
                    from_email=self.config.smtp_username,
                    to_emails=self.config.alert_emails
                )
                monitor.add_alert_handler(email_handler)
                logger.info("Email alerts configured")
        
        # Memory manager
        self.components['memory'] = MemoryManager(
            memory_limit_gb=self.config.memory_limit_gb,
            enable_disk_cache=self.config.enable_disk_cache
        )
        
        # Cache manager
        self.components['cache'] = IntelligentCacheManager(
            cache_dir="data/cache/production",
            max_cache_size_gb=self.config.cache_size_gb,
            default_ttl_hours=self.config.default_ttl_hours,
            enable_compression=self.config.enable_compression
        )
        
        # Computation optimizer
        self.components['optimizer'] = ComputationOptimizer(
            max_workers=self.config.max_workers,
            use_gpu=self.config.enable_gpu
        )
        
        # Performance profiler
        if self.config.enable_profiling:
            self.components['profiler'] = PerformanceProfiler()
        
        logger.info("Production performance components initialized")
    
    def start_monitoring(self):
        """Start production monitoring."""
        if 'monitor' in self.components:
            self.components['monitor'].start_monitoring()
            logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop production monitoring."""
        if 'monitor' in self.components:
            self.components['monitor'].stop_monitoring_service()
            logger.info("Production monitoring stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        status = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "monitoring_active": False,
            "system_health": {},
            "cache_stats": {},
            "memory_stats": {},
            "active_alerts": [],
            "performance_summary": {}
        }
        
        try:
            # System health
            if 'monitor' in self.components:
                monitor = self.components['monitor']
                status["monitoring_active"] = True
                status["system_health"] = monitor.get_metrics_summary()
                status["active_alerts"] = [
                    {
                        "level": alert.level.value,
                        "message": alert.message,
                        "metric": alert.metric_name,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in monitor.get_active_alerts()
                ]
            
            # Cache statistics
            if 'cache' in self.components:
                status["cache_stats"] = self.components['cache'].get_cache_stats()
            
            # Memory statistics
            if 'memory' in self.components:
                memory_stats = self.components['memory'].get_memory_stats()
                status["memory_stats"] = {
                    "total_gb": memory_stats.total_memory,
                    "available_gb": memory_stats.available_memory,
                    "used_percent": memory_stats.memory_percent,
                    "process_gb": memory_stats.process_memory
                }
            
            # Performance summary
            if 'profiler' in self.components:
                status["performance_summary"] = self.components['profiler'].get_performance_summary()
        
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            status["error"] = str(e)
        
        return status
    
    def cleanup_resources(self):
        """Clean up all performance resources."""
        
        logger.info("Cleaning up production performance resources")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clean up cache
        if 'cache' in self.components:
            self.components['cache'].clear_expired()
        
        # Clean up memory
        if 'memory' in self.components:
            self.components['memory'].cleanup_cache()
        
        logger.info("Production performance cleanup completed")
    
    def get_component(self, component_name: str):
        """Get specific performance component."""
        return self.components.get(component_name)
    
    def record_application_metric(self, 
                                name: str, 
                                value: float, 
                                unit: str = "",
                                warning_threshold: Optional[float] = None,
                                critical_threshold: Optional[float] = None):
        """Record application-specific metric."""
        
        if 'monitor' in self.components:
            self.components['monitor'].record_custom_metric(
                name, value, unit, warning_threshold, critical_threshold
            )


def create_production_config_from_env() -> ProductionConfig:
    """Create production configuration from environment variables."""
    
    return ProductionConfig(
        # System monitoring
        monitoring_interval=int(os.getenv('PERF_MONITORING_INTERVAL', '60')),
        enable_alerts=os.getenv('PERF_ENABLE_ALERTS', 'true').lower() == 'true',
        
        # Memory management
        memory_limit_gb=float(os.getenv('PERF_MEMORY_LIMIT_GB', '16.0')),
        enable_disk_cache=os.getenv('PERF_ENABLE_DISK_CACHE', 'true').lower() == 'true',
        
        # Computation optimization
        max_workers=int(os.getenv('PERF_MAX_WORKERS')) if os.getenv('PERF_MAX_WORKERS') else None,
        enable_gpu=os.getenv('PERF_ENABLE_GPU', 'false').lower() == 'true',
        
        # Caching
        cache_size_gb=float(os.getenv('PERF_CACHE_SIZE_GB', '5.0')),
        default_ttl_hours=int(os.getenv('PERF_DEFAULT_TTL_HOURS', '24')),
        enable_compression=os.getenv('PERF_ENABLE_COMPRESSION', 'true').lower() == 'true',
        
        # Profiling
        enable_profiling=os.getenv('PERF_ENABLE_PROFILING', 'true').lower() == 'true',
        profile_sample_rate=float(os.getenv('PERF_PROFILE_SAMPLE_RATE', '0.1')),
        
        # Alert thresholds
        cpu_warning_threshold=float(os.getenv('PERF_CPU_WARNING', '80.0')),
        cpu_critical_threshold=float(os.getenv('PERF_CPU_CRITICAL', '95.0')),
        memory_warning_threshold=float(os.getenv('PERF_MEMORY_WARNING', '85.0')),
        memory_critical_threshold=float(os.getenv('PERF_MEMORY_CRITICAL', '95.0')),
        disk_warning_threshold=float(os.getenv('PERF_DISK_WARNING', '90.0')),
        disk_critical_threshold=float(os.getenv('PERF_DISK_CRITICAL', '98.0')),
        
        # Email alerts
        smtp_server=os.getenv('PERF_SMTP_SERVER'),
        smtp_port=int(os.getenv('PERF_SMTP_PORT', '587')),
        smtp_username=os.getenv('PERF_SMTP_USERNAME'),
        smtp_password=os.getenv('PERF_SMTP_PASSWORD'),
        alert_emails=os.getenv('PERF_ALERT_EMAILS', '').split(',') if os.getenv('PERF_ALERT_EMAILS') else []
    )


# Global production manager instance
_production_manager: Optional[ProductionPerformanceManager] = None

def get_production_manager() -> ProductionPerformanceManager:
    """Get or create global production performance manager."""
    global _production_manager
    
    if _production_manager is None:
        config = create_production_config_from_env()
        _production_manager = ProductionPerformanceManager(config)
        _production_manager.start_monitoring()
    
    return _production_manager

def initialize_production_performance():
    """Initialize production performance monitoring."""
    manager = get_production_manager()
    logger.info("Production performance monitoring initialized")
    return manager

def shutdown_production_performance():
    """Shutdown production performance monitoring."""
    global _production_manager
    
    if _production_manager is not None:
        _production_manager.cleanup_resources()
        _production_manager = None
        logger.info("Production performance monitoring shutdown")


# Context manager for production performance
class ProductionPerformanceContext:
    """Context manager for production performance monitoring."""
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or create_production_config_from_env()
        self.manager = None
    
    def __enter__(self) -> ProductionPerformanceManager:
        self.manager = ProductionPerformanceManager(self.config)
        self.manager.start_monitoring()
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.manager:
            self.manager.cleanup_resources()