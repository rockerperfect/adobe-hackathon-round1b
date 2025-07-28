"""
Batch document processor - NEW: Multi-document orchestration for Round 1B.

Purpose:
    Implements the BatchDocumentProcessor class for orchestrating the processing of multiple documents in parallel or sequentially.
    Integrates PDF parsing, outline extraction, content segmentation, multilingual processing, and intelligence analysis.

Structure:
    - BatchDocumentProcessor class: main orchestration logic

Dependencies:
    - config.settings (for orchestration config)
    - src.utils.logger (for logging)
    - src.pdf_parser.batch_processor (for PDF parsing)
    - src.outline_extractor.content_extractor (for section extraction)
    - src.processors.content_segmenter (for text segmentation)
    - src.processors.multilingual (for multilingual processing)
    - src.intelligence.persona_driven_intelligence (for intelligence pipeline)

Integration Points:
    - Accepts list of document file paths or parsed objects, configuration, and logger
    - Returns structured results for all documents, including metadata and errors

NOTE: No hardcoding; all configuration, model paths, and language codes are config-driven.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import settings
from src.utils.logger import get_logger
from src.pdf_parser.batch_processor import BatchProcessor
from src.outline_extractor.content_extractor import ContentExtractor
from src.processors.content_segmenter import ContentSegmenter
from src.processors.multilingual import MultilingualProcessor
from src.intelligence.persona_driven_intelligence import PersonaDrivenIntelligence


class BatchDocumentProcessor:
    """
    BatchDocumentProcessor orchestrates the processing of multiple documents.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        batch_processor (BatchProcessor): PDF batch processor.
        content_extractor (ContentExtractor): Section content extractor.
        content_segmenter (ContentSegmenter): Text segmenter.
        multilingual_processor (MultilingualProcessor): Multilingual processor.
        intelligence_pipeline (PersonaDrivenIntelligence): Intelligence pipeline orchestrator.

    Raises:
        ValueError: If input is missing or invalid.

    Limitations:
        Assumes all downstream modules are implemented and functional.
        Does not handle file I/O or output formatting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the BatchDocumentProcessor.

        Args:
            config (dict, optional): Project configuration. If None, loads from settings.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.batch_processor = BatchProcessor(self.config.get('max_workers', 4))
        self.content_extractor = ContentExtractor(self.config)
        self.content_segmenter = ContentSegmenter(self.config)
        self.multilingual_processor = MultilingualProcessor(self.config)
        self.intelligence_pipeline = PersonaDrivenIntelligence(self.config)
        self.parallel = self.config.get('parallel_processing', True)

    def process_documents(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        persona_context: Dict[str, Any],
        job_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Orchestrate the processing of multiple documents.

        Args:
            documents (list): List of document file paths or parsed document objects.
            persona_context (dict): Persona information for intelligence analysis.
            job_context (dict): Job information for intelligence analysis.

        Returns:
            list: List of structured results for each document, including metadata and errors.

        Raises:
            Exception: For unrecoverable pipeline errors (logged and re-raised).

        Edge Cases:
            - Handles empty document list (returns empty list).
            - Aggregates errors per document.
        """
        if not documents:
            self.logger.warning("No documents provided for batch processing; returning empty result list.")
            return []

        results = []
        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
                future_to_doc = {
                    executor.submit(self._process_single_document, doc, persona_context, job_context): doc
                    for doc in documents
                }
                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Document processing failed: {e}", exc_info=True)
                        results.append({
                            'document': doc,
                            'error': str(e),
                        })
        else:
            for doc in documents:
                try:
                    result = self._process_single_document(doc, persona_context, job_context)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Document processing failed: {e}", exc_info=True)
                    results.append({
                        'document': doc,
                        'error': str(e),
                    })
        self.logger.info(f"Processed {len(results)} documents.")
        return results

    def _process_single_document(
        self,
        document: Union[str, Dict[str, Any]],
        persona_context: Dict[str, Any],
        job_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single document through the full pipeline.

        Args:
            document (str or dict): Document file path or parsed document object.
            persona_context (dict): Persona information for intelligence analysis.
            job_context (dict): Job information for intelligence analysis.

        Returns:
            dict: Structured result for the document, including metadata and errors.

        Raises:
            Exception: For unrecoverable pipeline errors (logged and re-raised).

        Edge Cases:
            - Handles errors at each pipeline stage and aggregates them in the result.
        """
        result = {'document': document, 'errors': []}
        try:
            # 1. PDF parsing
            if isinstance(document, str):
                parsed_pdf = self.batch_processor.process([document])[0]
            else:
                parsed_pdf = document
            # 2. Outline/section extraction
            headings = parsed_pdf.get('headings', [])
            sections = self.content_extractor.extract_sections(parsed_pdf, headings)
            # 3. Content segmentation
            segmented_sections = [
                self.content_segmenter.segment(section['content'], section['metadata'])
                for section in sections
            ]
            # 4. Multilingual processing
            multilingual_sections = [
                self.multilingual_processor.process(seg, section['metadata'])
                for seg, section in zip(segmented_sections, sections)
            ]
            # 5. Intelligence analysis
            intelligence_result = self.intelligence_pipeline.run_pipeline(
                parsed_pdf=parsed_pdf,
                persona_context=persona_context,
                job_context=job_context,
                extra_metadata={'document': document}
            )
            result.update({
                'parsed_pdf': parsed_pdf,
                'sections': sections,
                'segmented_sections': segmented_sections,
                'multilingual_sections': multilingual_sections,
                'intelligence_result': intelligence_result,
            })
        except Exception as e:
            self.logger.error(f"Error processing document: {e}", exc_info=True)
            result['errors'].append(str(e))
        return result
