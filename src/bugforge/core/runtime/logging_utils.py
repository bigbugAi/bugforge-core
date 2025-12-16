"""
Logging utilities for BugForge intelligence systems.

LoggingUtils provides structured logging with context management,
performance tracking, and log aggregation.
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from pathlib import Path


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    message: str
    component: str
    context: Dict[str, Any]
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class StructuredLogger:
    """
    Structured logger with context and performance tracking.
    
    Provides structured logging with JSON formatting,
    context management, and performance metrics.
    """
    
    def __init__(self, 
                 name: str,
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_json: bool = True):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path (optional)
            format_json: Whether to format logs as JSON
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.format_json = format_json
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        if format_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """
        Set logging context.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context."""
        self._context.clear()
    
    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for temporary logging context.
        
        Args:
            **kwargs: Temporary context key-value pairs
        """
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        if error:
            kwargs["error"] = str(error)
            kwargs["error_type"] = type(error).__name__
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        if error:
            kwargs["error"] = str(error)
            kwargs["error_type"] = type(error).__name__
        self._log(logging.CRITICAL, message, **kwargs)
    
    def performance(self, operation: str, duration_ms: float, **kwargs) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            **kwargs: Additional performance data
        """
        self.info(f"Performance: {operation}", 
                 operation=operation, 
                 duration_ms=duration_ms,
                 **kwargs)
    
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """
        Context manager for timing operations.
        
        Args:
            operation: Operation name
            **kwargs: Additional context
        """
        start_time = time.time()
        try:
            with self.context(**kwargs):
                yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.performance(operation, duration_ms)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal logging method."""
        # Merge context with kwargs
        log_data = {**self._context, **kwargs}
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=logging.getLevelName(level),
            message=message,
            component=self.name,
            context=log_data
        )
        
        # Log using standard logging
        if self.format_json:
            log_record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, (), None
            )
            log_record.__dict__.update(asdict(entry))
            self.logger.handle(log_record)
        else:
            # Standard logging with context
            formatted_message = f"{message}"
            if log_data:
                formatted_message += f" | Context: {json.dumps(log_data)}"
            
            self.logger.log(level, formatted_message)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add structured data if available
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", 
                          "pathname", "filename", "module", "lineno", 
                          "funcName", "created", "msecs", "relativeCreated", 
                          "thread", "threadName", "processName", "process",
                          "getMessage", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class PerformanceTracker:
    """
    Performance tracking utility.
    
    Tracks operation performance with statistics and reporting.
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize performance tracker.
        
        Args:
            logger: Logger for performance events
        """
        self.logger = logger or StructuredLogger("performance")
        self._operations: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def track(self, operation: str, **kwargs):
        """
        Context manager for tracking operation performance.
        
        Args:
            operation: Operation name
            **kwargs: Additional context
        """
        start_time = time.time()
        operation_id = f"{operation}_{start_time}"
        
        try:
            with self.logger.context(**kwargs):
                yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(operation, duration_ms)
            self.logger.performance(operation, duration_ms, operation_id=operation_id)
    
    def start_operation(self, operation: str) -> str:
        """
        Start tracking an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Operation ID
        """
        operation_id = f"{operation}_{time.time()}"
        self._start_times[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str, operation: str) -> float:
        """
        End tracking an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            operation: Operation name
            
        Returns:
            Duration in milliseconds
        """
        if operation_id not in self._start_times:
            return 0.0
        
        start_time = self._start_times.pop(operation_id)
        duration_ms = (time.time() - start_time) * 1000
        self._record_operation(operation, duration_ms)
        
        return duration_ms
    
    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            operation: Specific operation to get stats for (optional)
            
        Returns:
            Performance statistics
        """
        if operation:
            if operation not in self._operations:
                return {}
            
            durations = self._operations[operation]
            return {
                "operation": operation,
                "count": len(durations),
                "total_ms": sum(durations),
                "average_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations)
            }
        else:
            stats = {}
            for op_name, durations in self._operations.items():
                stats[op_name] = {
                    "count": len(durations),
                    "total_ms": sum(durations),
                    "average_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations)
                }
            return stats
    
    def reset(self) -> None:
        """Reset all performance tracking."""
        self._operations.clear()
        self._start_times.clear()
    
    def _record_operation(self, operation: str, duration_ms: float) -> None:
        """Record operation duration."""
        if operation not in self._operations:
            self._operations[operation] = []
        self._operations[operation].append(duration_ms)


class LoggingUtils:
    """
    Utility functions for logging configuration and management.
    """
    
    @staticmethod
    def setup_logging(config: Dict[str, Any]) -> Dict[str, StructuredLogger]:
        """
        Setup logging from configuration.
        
        Args:
            config: Logging configuration
            
        Returns:
            Dictionary of configured loggers
        """
        loggers = {}
        
        for logger_config in config.get("loggers", []):
            name = logger_config["name"]
            level = logger_config.get("level", "INFO")
            log_file = logger_config.get("log_file")
            format_json = logger_config.get("format_json", True)
            
            logger = StructuredLogger(name, level, log_file, format_json)
            
            # Set initial context
            if "context" in logger_config:
                logger.set_context(**logger_config["context"])
            
            loggers[name] = logger
        
        return loggers
    
    @staticmethod
    def create_logger(name: str, **kwargs) -> StructuredLogger:
        """
        Create a structured logger.
        
        Args:
            name: Logger name
            **kwargs: Logger configuration
            
        Returns:
            Configured logger
        """
        return StructuredLogger(name, **kwargs)
    
    @staticmethod
    def get_default_logger() -> StructuredLogger:
        """
        Get default logger.
        
        Returns:
            Default structured logger
        """
        return StructuredLogger("bugforge")


# Global performance tracker
_global_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """
    Get global performance tracker.
    
    Returns:
        Global performance tracker instance
    """
    return _global_tracker


def track_performance(operation: str, **kwargs):
    """
    Decorator for tracking function performance.
    
    Args:
        operation: Operation name
        **kwargs: Additional context
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            with _global_tracker.track(operation, **kwargs):
                return func(*args, **func_kwargs)
        return wrapper
    return decorator
