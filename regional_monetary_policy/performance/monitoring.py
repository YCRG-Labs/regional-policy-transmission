"""
System monitoring and alerting for performance and health.
"""

import time
import psutil
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Container for performance metrics."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def get_alert_level(self) -> Optional[AlertLevel]:
        """Determine alert level based on thresholds."""
        if self.threshold_critical and self.value >= self.threshold_critical:
            return AlertLevel.CRITICAL
        elif self.threshold_warning and self.value >= self.threshold_warning:
            return AlertLevel.WARNING
        return None

@dataclass
class PerformanceAlert:
    """Container for performance alerts."""
    level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SystemHealth:
    """Container for system health status."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    process_count: int
    uptime_seconds: float
    timestamp: datetime
    alerts: List[PerformanceAlert] = field(default_factory=list)

class SystemMonitor:
    """Monitors system performance and health metrics."""
    
    def __init__(self, 
                 monitoring_interval: int = 30,
                 enable_alerts: bool = True,
                 alert_handlers: Optional[List[Callable]] = None):
        
        self.monitoring_interval = monitoring_interval
        self.enable_alerts = enable_alerts
        self.alert_handlers = alert_handlers or []
        
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_queue = queue.Queue()
        
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Default thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 80.0, 'critical': 95.0},
            'memory_percent': {'warning': 85.0, 'critical': 95.0},
            'disk_percent': {'warning': 90.0, 'critical': 98.0},
            'response_time': {'warning': 5.0, 'critical': 10.0},
            'error_rate': {'warning': 0.05, 'critical': 0.10}
        }
        
        logger.info(f"System monitor initialized with {monitoring_interval}s interval")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        self.stop_monitoring.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start alert processing thread
        if self.enable_alerts:
            self.alert_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
            self.alert_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring_service(self):
        """Stop background monitoring."""
        self.stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Collect system metrics
                health = self.collect_system_health()
                
                # Store metrics
                self._store_metrics(health)
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts(health)
                
                # Wait for next interval
                self.stop_monitoring.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _alert_processing_loop(self):
        """Process alerts from queue."""
        while not self.stop_monitoring.is_set():
            try:
                # Get alert from queue (with timeout)
                alert = self.alert_queue.get(timeout=1)
                
                # Process alert
                self._process_alert(alert)
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {str(e)}")
    
    def collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics."""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # System uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        return SystemHealth(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io=network_io,
            process_count=process_count,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now()
        )
    
    def _store_metrics(self, health: SystemHealth):
        """Store metrics in history."""
        
        metrics = [
            PerformanceMetric(
                name='cpu_percent',
                value=health.cpu_percent,
                unit='%',
                timestamp=health.timestamp,
                threshold_warning=self.thresholds['cpu_percent']['warning'],
                threshold_critical=self.thresholds['cpu_percent']['critical']
            ),
            PerformanceMetric(
                name='memory_percent',
                value=health.memory_percent,
                unit='%',
                timestamp=health.timestamp,
                threshold_warning=self.thresholds['memory_percent']['warning'],
                threshold_critical=self.thresholds['memory_percent']['critical']
            ),
            PerformanceMetric(
                name='disk_percent',
                value=health.disk_percent,
                unit='%',
                timestamp=health.timestamp,
                threshold_warning=self.thresholds['disk_percent']['warning'],
                threshold_critical=self.thresholds['disk_percent']['critical']
            )
        ]
        
        for metric in metrics:
            if metric.name not in self.metrics_history:
                self.metrics_history[metric.name] = []
            
            self.metrics_history[metric.name].append(metric)
            
            # Keep only last 1000 entries per metric
            if len(self.metrics_history[metric.name]) > 1000:
                self.metrics_history[metric.name] = self.metrics_history[metric.name][-1000:]
    
    def _check_alerts(self, health: SystemHealth):
        """Check metrics against thresholds and generate alerts."""
        
        current_time = datetime.now()
        
        # Check CPU usage
        self._check_metric_alert(
            'cpu_percent', 
            health.cpu_percent, 
            self.thresholds['cpu_percent'],
            current_time
        )
        
        # Check memory usage
        self._check_metric_alert(
            'memory_percent', 
            health.memory_percent, 
            self.thresholds['memory_percent'],
            current_time
        )
        
        # Check disk usage
        self._check_metric_alert(
            'disk_percent', 
            health.disk_percent, 
            self.thresholds['disk_percent'],
            current_time
        )
    
    def _check_metric_alert(self, 
                          metric_name: str, 
                          value: float, 
                          thresholds: Dict[str, float],
                          timestamp: datetime):
        """Check individual metric against thresholds."""
        
        alert_level = None
        threshold_value = None
        
        if value >= thresholds['critical']:
            alert_level = AlertLevel.CRITICAL
            threshold_value = thresholds['critical']
        elif value >= thresholds['warning']:
            alert_level = AlertLevel.WARNING
            threshold_value = thresholds['warning']
        
        if alert_level:
            # Create or update alert
            alert_key = f"{metric_name}_{alert_level.value}"
            
            if alert_key not in self.active_alerts:
                alert = PerformanceAlert(
                    level=alert_level,
                    message=f"{metric_name.replace('_', ' ').title()} is {value:.1f}% (threshold: {threshold_value:.1f}%)",
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=threshold_value,
                    timestamp=timestamp
                )
                
                self.active_alerts[alert_key] = alert
                self.alert_queue.put(alert)
                
                logger.warning(f"Alert triggered: {alert.message}")
        else:
            # Check if we need to resolve existing alerts
            for level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                alert_key = f"{metric_name}_{level.value}"
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.resolved = True
                    alert.resolution_time = timestamp
                    
                    # Create resolution alert
                    resolution_alert = PerformanceAlert(
                        level=AlertLevel.INFO,
                        message=f"{metric_name.replace('_', ' ').title()} returned to normal: {value:.1f}%",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=alert.threshold,
                        timestamp=timestamp,
                        resolved=True
                    )
                    
                    self.alert_queue.put(resolution_alert)
                    del self.active_alerts[alert_key]
                    
                    logger.info(f"Alert resolved: {resolution_alert.message}")
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process individual alert through handlers."""
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def set_threshold(self, 
                     metric_name: str, 
                     warning: Optional[float] = None,
                     critical: Optional[float] = None):
        """Set custom thresholds for metrics."""
        
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        if warning is not None:
            self.thresholds[metric_name]['warning'] = warning
        
        if critical is not None:
            self.thresholds[metric_name]['critical'] = critical
        
        logger.info(f"Updated thresholds for {metric_name}: {self.thresholds[metric_name]}")
    
    def get_metrics_summary(self, 
                          hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics for specified time period."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        summary = {}
        
        for metric_name, metrics in self.metrics_history.items():
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[metric_name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'current': values[-1] if values else None,
                    'unit': recent_metrics[0].unit
                }
        
        return summary
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def record_custom_metric(self, 
                           name: str, 
                           value: float, 
                           unit: str = "",
                           warning_threshold: Optional[float] = None,
                           critical_threshold: Optional[float] = None):
        """Record custom application metric."""
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold
        )
        
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        self.metrics_history[name].append(metric)
        
        # Check for alerts if thresholds are set
        if self.enable_alerts and (warning_threshold or critical_threshold):
            thresholds = {}
            if warning_threshold:
                thresholds['warning'] = warning_threshold
            if critical_threshold:
                thresholds['critical'] = critical_threshold
            
            self._check_metric_alert(name, value, thresholds, metric.timestamp)
        
        logger.debug(f"Recorded custom metric {name}: {value} {unit}")

class EmailAlertHandler:
    """Email alert handler for sending notifications."""
    
    def __init__(self, 
                 smtp_server: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_email: str,
                 to_emails: List[str]):
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def __call__(self, alert: PerformanceAlert):
        """Send email alert."""
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] Regional Monetary Policy System Alert"
            
            # Email body
            body = f"""
            Alert Level: {alert.level.value.upper()}
            Metric: {alert.metric_name}
            Message: {alert.message}
            Value: {alert.metric_value}
            Threshold: {alert.threshold}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            {'Alert Resolved' if alert.resolved else 'Alert Active'}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

class LogAlertHandler:
    """Log-based alert handler."""
    
    def __init__(self, log_level: str = "WARNING"):
        self.log_level = getattr(logging, log_level.upper())
    
    def __call__(self, alert: PerformanceAlert):
        """Log alert message."""
        
        log_message = f"ALERT [{alert.level.value.upper()}] {alert.message}"
        
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)