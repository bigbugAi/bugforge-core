"""
System monitoring implementation for BugForge intelligence systems.

SystemMonitor provides health monitoring, metrics collection,
and system status tracking for intelligent systems.
"""

import psutil
import threading
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json


@dataclass
class SystemMetric:
    """System metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status."""
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)


class HealthCheck:
    """
    Base class for health checks.
    
    Health checks monitor specific aspects of system health.
    """
    
    def __init__(self, name: str, interval_seconds: int = 60):
        """
        Initialize health check.
        
        Args:
            name: Health check name
            interval_seconds: Check interval in seconds
        """
        self.name = name
        self.interval_seconds = interval_seconds
        self.last_check: Optional[datetime] = None
        self.last_result: Optional[bool] = None
        self.last_message: Optional[str] = None
    
    def check(self) -> bool:
        """
        Perform health check.
        
        Returns:
            True if healthy
        """
        try:
            result = self._do_check()
            self.last_check = datetime.now()
            self.last_result = result
            self.last_message = "Check passed" if result else "Check failed"
            return result
        except Exception as e:
            self.last_check = datetime.now()
            self.last_result = False
            self.last_message = f"Check error: {str(e)}"
            return False
    
    def _do_check(self) -> bool:
        """
        Perform the actual health check.
        
        Returns:
            True if healthy
        """
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get health check status.
        
        Returns:
            Health check status dictionary
        """
        return {
            "name": self.name,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_result": self.last_result,
            "last_message": self.last_message,
            "interval_seconds": self.interval_seconds
        }


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""
    
    def __init__(self, threshold_percent: float = 80.0, **kwargs):
        """
        Initialize memory health check.
        
        Args:
            threshold_percent: Memory usage threshold percentage
            **kwargs: Additional health check arguments
        """
        super().__init__("memory", **kwargs)
        self.threshold_percent = threshold_percent
    
    def _do_check(self) -> bool:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        return memory.percent < self.threshold_percent


class CPUHealthCheck(HealthCheck):
    """Health check for CPU usage."""
    
    def __init__(self, threshold_percent: float = 80.0, **kwargs):
        """
        Initialize CPU health check.
        
        Args:
            threshold_percent: CPU usage threshold percentage
            **kwargs: Additional health check arguments
        """
        super().__init__("cpu", **kwargs)
        self.threshold_percent = threshold_percent
    
    def _do_check(self) -> bool:
        """Check CPU usage."""
        return psutil.cpu_percent() < self.threshold_percent


class DiskHealthCheck(HealthCheck):
    """Health check for disk usage."""
    
    def __init__(self, path: str = "/", threshold_percent: float = 90.0, **kwargs):
        """
        Initialize disk health check.
        
        Args:
            path: Disk path to check
            threshold_percent: Disk usage threshold percentage
            **kwargs: Additional health check arguments
        """
        super().__init__("disk", **kwargs)
        self.path = path
        self.threshold_percent = threshold_percent
    
    def _do_check(self) -> bool:
        """Check disk usage."""
        disk = psutil.disk_usage(self.path)
        return (disk.used / disk.total) * 100 < self.threshold_percent


class CustomHealthCheck(HealthCheck):
    """Custom health check using a callback function."""
    
    def __init__(self, name: str, check_function: Callable[[], bool], **kwargs):
        """
        Initialize custom health check.
        
        Args:
            name: Health check name
            check_function: Function that returns True if healthy
            **kwargs: Additional health check arguments
        """
        super().__init__(name, **kwargs)
        self.check_function = check_function
    
    def _do_check(self) -> bool:
        """Perform custom health check."""
        return self.check_function()


class SystemMonitor:
    """
    System monitoring and health tracking.
    
    Provides comprehensive system monitoring with health checks,
    metrics collection, and status reporting.
    """
    
    def __init__(self, metrics_window_size: int = 1000):
        """
        Initialize system monitor.
        
        Args:
            metrics_window_size: Size of metrics rolling window
        """
        self.metrics_window_size = metrics_window_size
        self.health_checks: List[HealthCheck] = []
        self.metrics: Dict[str, deque] = {}
        self.callbacks: List[Callable[[HealthStatus], None]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Add default health checks
        self.add_health_check(MemoryHealthCheck())
        self.add_health_check(CPUHealthCheck())
        self.add_health_check(DiskHealthCheck())
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """
        Add a health check.
        
        Args:
            health_check: Health check to add
        """
        self.health_checks.append(health_check)
    
    def remove_health_check(self, name: str) -> bool:
        """
        Remove a health check by name.
        
        Args:
            name: Health check name
            
        Returns:
            True if health check was removed
        """
        self.health_checks = [hc for hc in self.health_checks if hc.name != name]
        return any(hc.name == name for hc in self.health_checks)
    
    def add_callback(self, callback: Callable[[HealthStatus], None]) -> None:
        """
        Add health status callback.
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval_seconds: int = 30) -> None:
        """
        Start system monitoring.
        
        Args:
            interval_seconds: Monitoring interval
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.stop_event.clear()
        
        def monitor_loop():
            while not self.stop_event.wait(interval_seconds):
                self._perform_health_checks()
                self._collect_system_metrics()
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_health_status(self) -> HealthStatus:
        """
        Get current system health status.
        
        Returns:
            Current health status
        """
        checks = {}
        metrics = {}
        overall_healthy = True
        
        for health_check in self.health_checks:
            result = health_check.check()
            checks[health_check.name] = result
            
            if not result:
                overall_healthy = False
            
            # Add relevant metrics
            if health_check.name == "memory":
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
            elif health_check.name == "cpu":
                metrics["cpu_percent"] = psutil.cpu_percent()
            elif health_check.name == "disk":
                disk = psutil.disk_usage("/")
                metrics["disk_percent"] = (disk.used / disk.total) * 100
        
        # Determine overall status
        if overall_healthy:
            status = "healthy"
            message = "All systems operational"
        else:
            failed_checks = [name for name, result in checks.items() if not result]
            status = "critical" if len(failed_checks) > 2 else "warning"
            message = f"Failed checks: {', '.join(failed_checks)}"
        
        return HealthStatus(
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
            checks=checks
        )
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> Union[List[SystemMetric], Dict[str, List[SystemMetric]]]:
        """
        Get collected metrics.
        
        Args:
            metric_name: Specific metric name (optional)
            since: Filter metrics since this time (optional)
            
        Returns:
            Metrics data
        """
        if metric_name:
            if metric_name not in self.metrics:
                return []
            
            metrics_list = list(self.metrics[metric_name])
            if since:
                metrics_list = [m for m in metrics_list if m.timestamp >= since]
            
            return metrics_list
        else:
            result = {}
            for name, metric_deque in self.metrics.items():
                metrics_list = list(metric_deque)
                if since:
                    metrics_list = [m for m in metrics_list if m.timestamp >= since]
                result[name] = metrics_list
            return result
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            tags: Metric tags
        """
        metric = SystemMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.metrics_window_size)
        
        self.metrics[name].append(metric)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            System information dictionary
        """
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_total": psutil.disk_usage("/").total,
            "disk_free": psutil.disk_usage("/").free,
            "disk_percent": (psutil.disk_usage("/").used / psutil.disk_usage("/").total) * 100,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "monitoring_active": self.monitoring
        }
    
    def _perform_health_checks(self) -> None:
        """Perform all health checks and notify callbacks."""
        health_status = self.get_health_status()
        
        for callback in self.callbacks:
            try:
                callback(health_status)
            except Exception:
                pass  # Don't let callback errors break monitoring
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        self.record_metric("cpu_percent", psutil.cpu_percent(), "%")
        
        memory = psutil.virtual_memory()
        self.record_metric("memory_percent", memory.percent, "%")
        self.record_metric("memory_available", memory.available, "bytes")
        
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric("disk_percent", disk_percent, "%")
        self.record_metric("disk_free", disk.free, "bytes")


# Global system monitor instance
_global_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """
    Get global system monitor.
    
    Returns:
        Global system monitor instance
    """
    return _global_monitor


def start_monitoring(interval_seconds: int = 30) -> None:
    """
    Start global system monitoring.
    
    Args:
        interval_seconds: Monitoring interval
    """
    _global_monitor.start_monitoring(interval_seconds)


def stop_monitoring() -> None:
    """Stop global system monitoring."""
    _global_monitor.stop_monitoring()


def record_metric(name: str, value: float, unit: str = "", **kwargs) -> None:
    """
    Record a metric globally.
    
    Args:
        name: Metric name
        value: Metric value
        unit: Metric unit
        **kwargs: Additional metric parameters
    """
    _global_monitor.record_metric(name, value, unit, **kwargs)


def get_health_status() -> HealthStatus:
    """
    Get global health status.
    
    Returns:
        Current health status
    """
    return _global_monitor.get_health_status()
