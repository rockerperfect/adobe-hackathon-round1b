
"""
Adobe Hackathon Round 1B: Persona-Driven Document Intelligence

Entry point for Docker container execution with official JSON format support.

Purpose:
    Orchestrates the end-to-end pipeline: loads config, validates input, initializes logging, performance monitoring, caching, and runs persona-driven document intelligence.

Structure:
    - Loads all config dynamically (no hardcoding)
    - Uses robust logging, error handling, and performance monitoring
    - Validates input/output JSON formats
    - Integrates all core utilities (logger, config, performance_monitor, cache_manager, json_validator, file_handler)

Dependencies:
    - config.settings, src.utils.logger, src.utils.performance_monitor, src.utils.json_validator, src.utils.file_handler, src.utils.cache_manager
    - src.intelligence.persona_driven_intelligence.PersonaDrivenIntelligence

Integration Points:
    - All core modules and utilities
    - Designed for extensibility and maintainability

NOTE: All configuration, paths, and credentials are loaded dynamically from config files or environment variables.
"""


import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config import settings
from src.utils.logger import get_logger
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.json_validator import validate_json
from src.utils.file_handler import read_file, write_file
from src.utils.cache_manager import CacheManager
from src.intelligence.persona_driven_intelligence import PersonaDrivenIntelligence
from src.pdf_parser.batch_processor import BatchProcessor


def load_official_input(input_path: Path, config: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    Load and validate official Round 1B input JSON format.
    Args:
        input_path (Path): Path to input JSON file.
        config (dict): Project configuration.
        logger (logging.Logger): Logger instance.
    Returns:
        dict: Loaded and validated input spec.
    Raises:
        SystemExit: If validation fails.
    """
    try:
        input_spec = json.loads(read_file(input_path))
        # Validate required fields
        required_fields = ['challenge_info', 'documents', 'persona', 'job_to_be_done']
        for field in required_fields:
            if field not in input_spec:
                raise ValueError(f"Missing required field: {field}")
        # Optionally validate against schema if available
        schema_path = config.get('input_schema_path')
        if schema_path:
            validate_json(input_spec, schema_path)
        return input_spec
    except Exception as e:
        logger.error(f"Error loading input specification: {e}", exc_info=True)
        sys.exit(1)


def validate_output_format(output_data: Dict[str, Any], logger) -> bool:
    """
    Validate output matches official Round 1B format.
    Args:
        output_data (dict): Output data to validate.
        logger (logging.Logger): Logger instance.
    Returns:
        bool: True if valid, False otherwise.
    """
    required_fields = ['metadata', 'extracted_sections', 'subsection_analysis']
    for field in required_fields:
        if field not in output_data:
            logger.error(f"Missing output field: {field}")
            return False
    metadata = output_data['metadata']
    metadata_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
    if not all(field in metadata for field in metadata_fields):
        logger.error("Metadata missing required fields.")
        return False
    for section in output_data['extracted_sections']:
        section_fields = ['document', 'section_title', 'importance_rank', 'page_number']
        if not all(field in section for field in section_fields):
            logger.error(f"Section missing required fields: {section}")
            return False
    for subsection in output_data['subsection_analysis']:
        subsection_fields = ['document', 'refined_text', 'page_number']
        if not all(field in subsection for field in subsection_fields):
            logger.error(f"Subsection missing required fields: {subsection}")
            return False
    return True


def process_single_collection(input_json_path: Path, output_file: Path, config: Dict[str, Any], logger) -> bool:
    """
    Process a single collection (one input JSON + PDFs + output JSON).
    
    Args:
        input_json_path: Path to challenge1b_input.json
        output_file: Path to challenge1b_output.json  
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Performance monitoring with EARLY TERMINATION
        perf = PerformanceMonitor(logger=logger)
        perf.start("collection_pipeline")
        
        # CRITICAL: Set aggressive timeout
        start_time = time.time()
        MAX_PROCESSING_TIME = config.get('MAX_PROCESSING_TIME', 45)  # 45 seconds max

        # Load and validate input
        if not input_json_path.exists():
            logger.error(f"Input specification file not found: {input_json_path}")
            return False
        
        input_spec = load_official_input(input_json_path, config, logger)
        input_dir = input_json_path.parent

        logger.info(f"Challenge: {input_spec['challenge_info']['test_case_name']}")
        logger.info(f"Documents: {len(input_spec['documents'])} PDFs")
        logger.info(f"Persona: {input_spec['persona']['role']}")
        logger.info(f"Job: {input_spec['job_to_be_done']['task']}")

        # Verify all PDF files exist (look in PDFs subdirectory for collections)
        pdf_paths = []
        pdfs_dir = input_dir / "PDFs"
        if pdfs_dir.exists():
            # Collection format: PDFs are in PDFs/ subdirectory
            for doc_info in input_spec['documents']:
                pdf_path = pdfs_dir / doc_info['filename']
                if pdf_path.exists():
                    pdf_paths.append(pdf_path)
                    logger.info(f"Found PDF: {pdf_path}")
                else:
                    logger.warning(f"PDF not found: {pdf_path}")
        else:
            # Fallback: PDFs are in same directory as input JSON
            for doc_info in input_spec['documents']:
                pdf_path = input_dir / doc_info['filename']
                if pdf_path.exists():
                    pdf_paths.append(pdf_path)
                    logger.info(f"Found PDF: {pdf_path}")
                else:
                    logger.warning(f"PDF not found: {pdf_path}")

        if not pdf_paths:
            raise ValueError("No PDF files found to process")

        # Initialize persona-driven intelligence system - OPTIMIZED
        logger.info("Initializing persona-driven intelligence system...")
        
        # Pre-load model to avoid multiple loadings during processing
        try:
            from src.nlp.model_singleton import ModelSingleton
            singleton = ModelSingleton()
            singleton.get_model()  # Pre-load the model once
        except Exception as e:
            logger.warning(f"Model pre-loading failed: {e}")
        
        # PERFORMANCE CHECK
        elapsed = time.time() - start_time
        if elapsed > MAX_PROCESSING_TIME * 0.3:  # 30% of time budget
            logger.warning(f"Initialization taking too long: {elapsed:.1f}s")
        
        intelligence_system = PersonaDrivenIntelligence(config=config, logger=logger)

        # Parse PDF documents - PARALLEL PROCESSING
        logger.info("Parsing PDF documents...")
        
        # PERFORMANCE MONITORING
        elapsed = time.time() - start_time
        logger.info(f"PDF parsing phase - elapsed time: {elapsed:.1f}s")
        
        pdf_processor = BatchProcessor(max_workers=4)  # INCREASED workers for speed
        
        # Process PDFs to extract content and sections
        parsed_pdfs = pdf_processor.process_documents(pdf_paths)
        logger.info(f"Parsed {len(parsed_pdfs)} PDF documents")
        
        # Combine all parsed PDFs into a single structure for the intelligence system
        combined_pdf_data = {
            "documents": parsed_pdfs,
            "sections": []
        }
        
        # Extract sections from all parsed PDFs
        for parsed_pdf in parsed_pdfs:
            if "sections" in parsed_pdf:
                combined_pdf_data["sections"].extend(parsed_pdf["sections"])
        
        logger.info(f"Total sections extracted: {len(combined_pdf_data['sections'])}")

        # Process documents - WITH TIMEOUT PROTECTION
        logger.info("Processing documents with multilingual NLP...")
        
        # PERFORMANCE MONITORING  
        elapsed = time.time() - start_time
        logger.info(f"NLP processing phase - elapsed time: {elapsed:.1f}s")
        
        pipeline_result = intelligence_system.run_pipeline(
            parsed_pdf=combined_pdf_data,
            persona_context=input_spec['persona'],
            job_context=input_spec['job_to_be_done']
        )
        
        # CHECK FINAL TIME
        elapsed = time.time() - start_time
        if elapsed > MAX_PROCESSING_TIME:
            logger.warning(f"Processing time ({elapsed:.1f}s) exceeds constraint ({MAX_PROCESSING_TIME}s)")
        else:
            logger.info(f"Processing completed within time constraint: {elapsed:.1f}s")

        # Transform pipeline output to match expected format
        extracted_subsections = pipeline_result.get('extracted_subsections', [])
        
        # Transform extracted sections to match expected format (simplified structure)
        transformed_sections = []
        for i, section in enumerate(pipeline_result.get('ranked_sections', [])):
            transformed_section = {
                "document": section.get('document', 'unknown.pdf'),
                "section_title": section.get('section_title', section.get('title', '')),
                "importance_rank": i + 1,
                "page_number": section.get('page_number', 1)
            }
            transformed_sections.append(transformed_section)

        # Transform subsections to match expected format
        transformed_subsections = []
        for subsection in extracted_subsections:
            transformed_subsection = {
                "document": subsection.get('section_metadata', {}).get('document', 'unknown.pdf'),
                "refined_text": subsection.get('passage', ''),
                "page_number": subsection.get('section_metadata', {}).get('page_number', 1)
            }
            transformed_subsections.append(transformed_subsection)
        
        result = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in input_spec['documents']],
                "persona": input_spec['persona']['role'],
                "job_to_be_done": input_spec['job_to_be_done']['task'],
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": transformed_sections,
            "subsection_analysis": transformed_subsections
        }

        perf.stop("collection_pipeline")
        metrics = perf.get_metrics("collection_pipeline")
        processing_time = metrics.get('duration_sec', 0)

        # Validate output format
        if not validate_output_format(result, logger):
            raise ValueError("Generated output does not match required format")

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_file(output_file, json.dumps(result, indent=2, ensure_ascii=False))

        # Print/log success summary
        logger.info(f"Processing completed successfully in {processing_time:.2f} seconds.")
        logger.info(f"Extracted sections: {len(result['extracted_sections'])}")
        logger.info(f"Detailed subsections: {len(result['subsection_analysis'])}")
        logger.info(f"Output saved to: {output_file}")

        # Performance validation
        if processing_time > config.get('max_processing_time', 60):
            logger.warning(f"Processing time ({processing_time:.2f}s) exceeds constraint.")
        else:
            logger.info("Performance constraint satisfied.")

        return True

    except Exception as e:
        logger.error(f"Error processing collection: {str(e)}", exc_info=True)
        
        # Generate minimal fallback output
        try:
            input_spec = load_official_input(input_json_path, config, logger)
            fallback_result = {
                "metadata": {
                    "input_documents": [doc['filename'] for doc in input_spec['documents']],
                    "persona": input_spec['persona']['role'],
                    "job_to_be_done": input_spec['job_to_be_done']['task'],
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            write_file(output_file, json.dumps(fallback_result, indent=2, ensure_ascii=False))
            logger.info(f"Fallback output saved to: {output_file}")
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback output: {fallback_error}")
        
        return False


def process_collections(collections_dir: Path, config: Dict[str, Any], logger) -> None:
    """
    Process multiple collections in the required format.
    
    Expected structure:
    collections_dir/
    ├── Collection 1/
    │   ├── PDFs/
    │   ├── challenge1b_input.json
    │   └── challenge1b_output.json
    ├── Collection 2/
    │   ├── PDFs/
    │   ├── challenge1b_input.json
    │   └── challenge1b_output.json
    ...
    """
    logger.info(f"Processing collections in directory: {collections_dir}")
    
    # Find all collection directories
    collection_dirs = []
    for item in collections_dir.iterdir():
        if item.is_dir() and item.name.lower().startswith('collection'):
            collection_dirs.append(item)
    
    if not collection_dirs:
        logger.error("No collection directories found (looking for directories starting with 'Collection')")
        sys.exit(1)
    
    collection_dirs.sort()  # Process in order
    logger.info(f"Found {len(collection_dirs)} collections to process")
    
    successful_collections = 0
    failed_collections = 0
    
    for collection_dir in collection_dirs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {collection_dir.name}")
        logger.info(f"{'='*60}")
        
        input_json = collection_dir / "challenge1b_input.json"
        output_json = collection_dir / "challenge1b_output.json"
        
        if not input_json.exists():
            logger.error(f"Input JSON not found in {collection_dir}: {input_json}")
            failed_collections += 1
            continue
        
        success = process_single_collection(input_json, output_json, config, logger)
        if success:
            successful_collections += 1
        else:
            failed_collections += 1
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COLLECTION PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total collections: {len(collection_dirs)}")
    logger.info(f"Successful: {successful_collections}")
    logger.info(f"Failed: {failed_collections}")
    
    if failed_collections > 0:
        logger.warning(f"{failed_collections} collections failed processing")
        sys.exit(1)
    else:
        logger.info("All collections processed successfully!")


def main():
    """
    Main orchestrator for Round 1B persona-driven document intelligence.
    Supports both single collection and multi-collection processing.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Adobe Hackathon Round 1B: Persona-Driven Document Intelligence')
    parser.add_argument('--input', default=None, help='Input JSON file path (single collection mode)')
    parser.add_argument('--output', default=None, help='Output JSON file path (single collection mode)')
    parser.add_argument('--collections_dir', default=None, help='Directory containing multiple collections (multi-collection mode)')
    args = parser.parse_args()
    
    logger = get_logger("main")
    logger.info("Adobe Hackathon Round 1B: Persona-Driven Document Intelligence - Starting pipeline.")

    # Load config dynamically
    config = settings.load_config()
    
    # Determine processing mode
    if args.collections_dir:
        # Multi-collection mode
        collections_dir = Path(args.collections_dir)
        if not collections_dir.exists():
            logger.error(f"Collections directory not found: {collections_dir}")
            sys.exit(1)
        process_collections(collections_dir, config, logger)
    else:
        # Single collection mode (backward compatibility)
        if args.input:
            input_json_path = Path(args.input)
            input_dir = input_json_path.parent
        else:
            input_dir = Path(config.get('input_dir', '/app/input'))
            input_json_path = input_dir / config.get('input_json', 'challenge1b_input.json')
        
        if args.output:
            output_file = Path(args.output)
        else:
            output_dir = Path(config.get('output_dir', '/app/output'))
            output_file = output_dir / config.get('output_json', 'challenge1b_output.json')
        
        logger.info("Running in single collection mode")
        success = process_single_collection(input_json_path, output_file, config, logger)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
