"""
Performance monitor - NEW: Processing time and resource tracking for Round 1B.

Purpose:
    Implements the PerformanceMonitor class for tracking processing time, CPU, and memory usage for named operations or pipeline stages.
    Aggregates and reports performance metrics for the full pipeline and individual components.

Structure:
    - PerformanceMonitor class: main monitoring logic

Dependencies:
    - config.settings (for monitoring config)
    - src.utils.logger (for logging)
    - time, psutil (for timing and resource usage)

Integration Points:
    - Used by pipeline orchestrators and processors to monitor performance
    - Reports metrics for logging, debugging, and optimization

NOTE: No hardcoding; all configuration is loaded from config files or environment variables.
"""

from typing import Any, Dict, Optional
import time
import logging
import psutil
from config import settings
from src.utils.logger import get_logger


class PerformanceMonitor:
    """
    PerformanceMonitor tracks processing time and resource usage for named operations.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        _timings (dict): Stores start/stop times for operations.
        _metrics (dict): Stores aggregated metrics for operations.

    Raises:
        ValueError: If operation name is missing or invalid.

    Limitations:
        GPU usage tracking is not implemented (CPU/memory only).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the PerformanceMonitor.

        Args:
            config (dict, optional): Project configuration. If None, loads from settings.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self._timings: Dict[str, Dict[str, float]] = {}
        self._metrics: Dict[str, Dict[str, Any]] = {}

    def start(self, operation: str) -> None:
        """
        Start timing and resource tracking for a named operation.

        Args:
            operation (str): Name of the operation or pipeline stage.

        Raises:
            ValueError: If operation name is missing.
        """
        if not operation:
            raise ValueError("Operation name is required.")
        self._timings[operation] = {
            'start_time': time.time(),
            'start_cpu': psutil.cpu_percent(interval=None),
            'start_mem': psutil.virtual_memory().used,
        }
        self.logger.debug(f"Started monitoring operation: {operation}")

    def stop(self, operation: str) -> None:
        """
        Stop timing and resource tracking for a named operation and record metrics.

        Args:
            operation (str): Name of the operation or pipeline stage.

        Raises:
            ValueError: If operation name is missing or not started.
        """
        if not operation or operation not in self._timings:
            raise ValueError(f"Operation '{operation}' was not started.")
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_mem = psutil.virtual_memory().used
        start = self._timings[operation]
        duration = end_time - start['start_time']
        cpu_delta = end_cpu - start['start_cpu']
        mem_delta = end_mem - start['start_mem']
        self._metrics[operation] = {
            'duration_sec': duration,
            'cpu_percent_delta': cpu_delta,
            'mem_bytes_delta': mem_delta,
        }
        self.logger.info(f"Operation '{operation}' completed in {duration:.2f}s, CPU Δ: {cpu_delta:.2f}%, Mem Δ: {mem_delta/1024/1024:.2f}MB")

    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific operation or all operations.

        Args:
            operation (str, optional): Name of the operation. If None, returns all metrics.

        Returns:
            dict: Metrics for the specified operation or all operations.
        """
        if operation:
            return self._metrics.get(operation, {})
        return self._metrics.copy()

    def report(self) -> None:
        """
        Log a summary report of all recorded performance metrics.

        Returns:
            None
        """
        for op, metrics in self._metrics.items():
            self.logger.info(f"Performance for '{op}': {metrics}")

    # TODO: Add support for GPU usage tracking if needed.
