"""
Logger - ENHANCED from Round 1A for Round 1B.
Comprehensive logging infrastructure for persona-driven document intelligence.

This module provides structured logging for all components of the Round 1B system
including PDF parsing, multilingual processing, persona analysis, and performance monitoring.

Features:
- Configurable log levels and formats
- Performance monitoring integration
- Structured logging for debugging
- Error tracking and reporting
- Batch processing log management
"""

import logging
import logging.handlers
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Import configuration
try:
    from ...config.settings import LOG_LEVEL, LOG_FORMAT
except ImportError:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PerformanceLogger:
    """Logger with built-in performance monitoring capabilities."""
    
    def __init__(self, name: str):
        """Initialize performance logger."""
        self.logger = logging.getLogger(name)
        self.start_times = {}
        self.performance_data = {}
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timing(self, operation: str) -> float:
        """End timing an operation and return elapsed time."""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.performance_data[operation] = elapsed
            self.logger.debug(f"Completed {operation} in {elapsed:.3f}s")
            del self.start_times[operation]
            return elapsed
        else:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
    
    def log_performance_summary(self) -> None:
        """Log summary of all recorded performance metrics."""
        if self.performance_data:
            total_time = sum(self.performance_data.values())
            self.logger.info(f"Performance Summary - Total time: {total_time:.3f}s")
            for operation, time_taken in self.performance_data.items():
                percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
                self.logger.info(f"  {operation}: {time_taken:.3f}s ({percentage:.1f}%)")
    
    def get_performance_data(self) -> Dict[str, float]:
        """Get current performance data."""
        return self.performance_data.copy()


class StructuredLogger:
    """Logger that supports structured logging with JSON output."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def set_context(self, **kwargs) -> None:
        """Set logging context that will be included in all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all logging context."""
        self.context.clear()
    
    def log_structured(self, level: str, message: str, **kwargs) -> None:
        """Log a structured message with context and additional data."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": self.context,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json.dumps(log_data, ensure_ascii=False))
    
    def info_structured(self, message: str, **kwargs) -> None:
        """Log structured info message."""
        self.log_structured("info", message, **kwargs)
    
    def error_structured(self, message: str, **kwargs) -> None:
        """Log structured error message."""
        self.log_structured("error", message, **kwargs)
    
    def debug_structured(self, message: str, **kwargs) -> None:
        """Log structured debug message."""
        self.log_structured("debug", message, **kwargs)


def setup_logging(
    log_level: str = LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    log_file: Optional[Path] = None,
    enable_console: bool = True,
    enable_performance: bool = True
) -> None:
    """
    Set up comprehensive logging for the Adobe Hackathon Round 1B system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file: Optional file path for log output
        enable_console: Whether to enable console logging
        enable_performance: Whether to enable performance logging
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific loggers for different components
    component_loggers = [
        "pdf_parser",
        "outline_extractor", 
        "intelligence",
        "nlp",
        "processors",
        "utils"
    ]
    
    for component in component_loggers:
        logger = logging.getLogger(component)
        logger.setLevel(numeric_level)
    
    # Log setup completion
    root_logger.info(f"Logging initialized - Level: {log_level}, Console: {enable_console}, File: {log_file}")


def get_logger(name: str, enable_performance: bool = False, enable_structured: bool = False) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        enable_performance: Whether to return a PerformanceLogger
        enable_structured: Whether to return a StructuredLogger
        
    Returns:
        Configured logger instance
    """
    if enable_performance:
        return PerformanceLogger(name)
    elif enable_structured:
        return StructuredLogger(name)
    else:
        return logging.getLogger(name)


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    logger = logging.getLogger("system")
    
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Memory Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    logger.info("=== End System Information ===")


def log_processing_start(document_count: int, persona: str, job: str) -> None:
    """Log the start of document processing with context."""
    logger = logging.getLogger("processing")
    logger.info("=== Processing Started ===")
    logger.info(f"Document Count: {document_count}")
    logger.info(f"Persona: {persona}")
    logger.info(f"Job: {job}")
    logger.info(f"Start Time: {datetime.now().isoformat()}")


def log_processing_end(processing_time: float, success: bool, results: Optional[Dict[str, Any]] = None) -> None:
    """Log the end of document processing with results."""
    logger = logging.getLogger("processing")
    logger.info("=== Processing Completed ===")
    logger.info(f"Success: {success}")
    logger.info(f"Processing Time: {processing_time:.3f}s")
    logger.info(f"End Time: {datetime.now().isoformat()}")
    
    if results:
        logger.info(f"Extracted Sections: {len(results.get('extracted_sections', []))}")
        logger.info(f"Subsection Analysis: {len(results.get('subsection_analysis', []))}")
    
    # Check performance constraints
    if processing_time > 60:
        logger.warning(f"Processing time ({processing_time:.1f}s) exceeds 60s constraint")
    else:
        logger.info("Processing time within 60s constraint")


def configure_third_party_loggers() -> None:
    """Configure third-party library loggers to reduce noise."""
    # Reduce verbosity of common libraries
    third_party_loggers = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "PIL": logging.WARNING,
        "matplotlib": logging.WARNING,
        "transformers": logging.ERROR,  # Reduce transformer model loading noise
        "sentence_transformers": logging.WARNING,
    }
    
    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


# Auto-configure on import
def _auto_configure():
    """Automatically configure basic logging on module import."""
    if not logging.getLogger().handlers:
        setup_logging()
        configure_third_party_loggers()


# Run auto-configuration
_auto_configure()
