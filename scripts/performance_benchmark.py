"""
Performance benchmark script.

Purpose:
    Test the pipeline's end-to-end performance against the 60-second constraint.

Structure:
    - Loads test documents and config dynamically (no hardcoding)
    - Runs the full pipeline (PDF parsing, outline extraction, segmentation, NLP, intelligence)
    - Measures and logs per-stage timing, CPU/memory usage, and errors
    - Outputs a structured performance report (JSON)

Dependencies:
    - config.settings
    - src.utils.logger, src.utils.performance_monitor
    - src.intelligence.persona_driven_intelligence.PersonaDrivenIntelligence
    - src.pdf_parser, src.outline_extractor, src.nlp, etc.
    - psutil (for resource monitoring)

Integration Points:
    - Used in CI, Docker build, and manual QA to ensure pipeline meets performance requirements

NOTE: Add new stages as pipeline evolves. All config and paths are dynamic.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import psutil
from config import settings
from src.utils.logger import get_logger
from src.utils.performance_monitor import PerformanceMonitor
from src.intelligence.persona_driven_intelligence import PersonaDrivenIntelligence

def load_test_input(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Load test input JSON for benchmarking.
    Args:
        config (dict): Project config
        logger (Logger): Logger instance
    Returns:
        dict: Input spec
    Raises:
        SystemExit: If input is missing or invalid
    """
    input_dir = Path(config.get('input_dir', '/app/input'))
    input_json = config.get('input_json', 'challenge1b_input.json')
    input_path = input_dir / input_json
    if not input_path.exists():
        logger.error(f"Test input not found: {input_path}")
        sys.exit(1)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_spec = json.load(f)
        return input_spec
    except Exception as e:
        logger.error(f"Failed to load test input: {e}", exc_info=True)
        sys.exit(1)

def select_test_documents(input_spec: Dict[str, Any], n: int = 3) -> List[Dict[str, Any]]:
    """
    Select a representative batch of documents for benchmarking.
    Args:
        input_spec (dict): Input spec
        n (int): Number of documents to select
    Returns:
        list: List of document dicts
    """
    docs = input_spec.get('documents', [])
    return docs[:n] if len(docs) >= n else docs

def benchmark_pipeline(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Run the full pipeline and measure performance.
    Args:
        config (dict): Project config
        logger (Logger): Logger instance
    Returns:
        dict: Performance report
    """
    perf = PerformanceMonitor(logger=logger)
    perf.start()
    process = psutil.Process(os.getpid())
    report = {
        'stages': {},
        'errors': [],
        'cpu_percent': [],
        'memory_mb': [],
        'start_time': time.time(),
    }
    try:
        # Load and subset input
        input_spec = load_test_input(config, logger)
        test_docs = select_test_documents(input_spec, n=3)
        input_spec['documents'] = test_docs

        # Stage: Pipeline
        t0 = time.time()
        intelligence = PersonaDrivenIntelligence(config=config, logger=logger)
        t1 = time.time()
        report['stages']['init'] = t1 - t0

        # Stage: Processing
        t2 = time.time()
        result = intelligence.process_documents(input_spec)
        t3 = time.time()
        report['stages']['process_documents'] = t3 - t2

        # Resource usage
        report['cpu_percent'] = process.cpu_percent(interval=1)
        mem_info = process.memory_info()
        report['memory_mb'] = mem_info.rss // (1024 * 1024)

        # Total time
        perf.stop()
        report['total_time'] = perf.get_total_time()
        report['status'] = 'pass' if report['total_time'] <= config.get('max_processing_time', 60) else 'fail'
        report['result_summary'] = {
            'extracted_sections': len(result.get('extracted_sections', [])),
            'subsection_analysis': len(result.get('subsection_analysis', [])),
        }
    except Exception as e:
        logger.error(f"Benchmarking error: {e}", exc_info=True)
        report['errors'].append(str(e))
        report['status'] = 'fail'
        perf.stop()
        report['total_time'] = perf.get_total_time()
    report['end_time'] = time.time()
    return report

def save_report(report: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Save the performance report to disk as JSON.
    Args:
        report (dict): Performance report
        config (dict): Project config
        logger (Logger): Logger instance
    """
    output_dir = Path(config.get('output_dir', '/app/output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'performance_report.json'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Performance report saved: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save performance report: {e}", exc_info=True)

def main():
    """
    Main entry point for performance benchmarking.
    Loads config, runs benchmark, and outputs report.
    """
    logger = get_logger("performance_benchmark")
    logger.info("Starting performance benchmark...")
    config = settings.load_config()
    report = benchmark_pipeline(config, logger)
    save_report(report, config, logger)
    print("\nPerformance Benchmark Summary:")
    print(json.dumps(report, indent=2))
    if report['status'] == 'pass':
        logger.info("Pipeline meets the 60-second constraint.")
        sys.exit(0)
    else:
        logger.error("Pipeline does NOT meet the 60-second constraint.")
        sys.exit(1)

if __name__ == "__main__":
    main()
