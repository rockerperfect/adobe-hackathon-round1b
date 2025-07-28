"""
Persona-driven intelligence - Main orchestrator for Round 1B.

Purpose:
    Orchestrates the full persona-driven document intelligence pipeline for Round 1B.
    Integrates PDF parsing, persona/job analysis, section ranking, relevance calculation, and subsection extraction.

Structure:
    - PersonaDrivenIntelligence class: main orchestrator
    - Integrates PersonaAnalyzer, JobAnalyzer, SectionRanker, RelevanceCalculator, SubsectionExtractor

Dependencies:
    - src.intelligence.persona_analyzer.PersonaAnalyzer
    - src.intelligence.job_analyzer.JobAnalyzer
    - src.intelligence.section_ranker.SectionRanker
    - src.intelligence.relevance_calculator.RelevanceCalculator
    - src.intelligence.subsection_extractor.SubsectionExtractor
    - config.settings, config.model_config, config.language_config
    - src.utils.logger

Integration Points:
    - Accepts parsed PDF data, persona/job context, and configuration as input
    - Returns structured results for output formatting
    - Logs all major steps and errors

NOTE: All configuration, model paths, and language codes must be loaded from config files or environment variables (no hardcoding).
"""

from typing import Any, Dict, List, Optional
import logging

from .persona_analyzer import PersonaAnalyzer
from .job_analyzer import JobAnalyzer
from .section_ranker import SectionRanker
from .relevance_calculator import RelevanceCalculator
from .subsection_extractor import SubsectionExtractor
from config import settings, model_config, language_config
from src.utils.logger import get_logger


class PersonaDrivenIntelligence:
    """
    Main orchestrator for persona-driven document intelligence (Round 1B).

    Coordinates persona/job analysis, section ranking, relevance calculation, and subsection extraction.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        persona_analyzer (PersonaAnalyzer): Persona analysis module.
        job_analyzer (JobAnalyzer): Job analysis module.
        section_ranker (SectionRanker): Section ranking module.
        relevance_calculator (RelevanceCalculator): Relevance scoring module.
        subsection_extractor (SubsectionExtractor): Subsection extraction module.
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If required modules or configuration are missing.

    Limitations:
        Assumes all downstream modules are implemented and functional.
        Does not handle file I/O or output formatting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the PersonaDrivenIntelligence orchestrator.

        Args:
            config (dict, optional): Project configuration. If None, loads from config files.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)

        # Initialize all intelligence modules
        self.persona_analyzer = PersonaAnalyzer(self.config)
        self.job_analyzer = JobAnalyzer(self.config)
        self.section_ranker = SectionRanker(self.config)
        self.relevance_calculator = RelevanceCalculator(self.config)
        self.subsection_extractor = SubsectionExtractor(self.config)

    def run_pipeline(
        self,
        parsed_pdf: Dict[str, Any],
        persona_context: Dict[str, Any],
        job_context: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full persona-driven intelligence pipeline - SPEED OPTIMIZED.

        Args:
            parsed_pdf (dict): Parsed PDF data (from PDF parsers).
            persona_context (dict): Persona information from input JSON.
            job_context (dict): Job-to-be-done information from input JSON.
            extra_metadata (dict, optional): Any additional metadata for logging or output.

        Returns:
            dict: Structured results including ranked sections and extracted subsections.

        Raises:
            Exception: For unrecoverable pipeline errors (logged and re-raised).

        Side Effects:
            Logs all major steps and errors.

        Limitations:
            Assumes all downstream modules are implemented and functional.
        """
        try:
            self.logger.info("Starting persona-driven intelligence pipeline.")
            
            # QUALITY RESTORED - No emergency mode needed with pre-warmed models
            self.logger.info("Running in HIGH QUALITY mode with pre-warmed models")

            # 1. Persona analysis
            persona_profile = self.persona_analyzer.analyze(persona_context)
            
            # 2. Job analysis
            job_profile = self.job_analyzer.analyze(job_context)

            # 3. Section ranking - RESTORED QUALITY PROCESSING
            sections = parsed_pdf.get("sections", [])
            
            # QUALITY RESTORED: Process all sections up to reasonable limit
            if len(sections) > 200:
                self.logger.info(f"Large section count ({len(sections)}), limiting to top 200 for quality balance")
                sections = sections[:200]
            
            # Pass original context along with profiles for filtering
            persona_profile_with_context = {**persona_profile, "original_text": persona_context}
            job_profile_with_context = {**job_profile, "original_text": job_context}
            
            ranked_sections = self.section_ranker.rank(sections, persona_profile_with_context, job_profile_with_context)

            # 4. Relevance calculation - FULL QUALITY RESTORED
            relevance_scores = self.relevance_calculator.calculate(ranked_sections, persona_profile_with_context, job_profile_with_context)

            # 5. Subsection extraction - FULL QUALITY RESTORED
            max_subsections = 30  # Restored for quality
            extracted_subsections = self.subsection_extractor.extract(
                ranked_sections[:max_subsections], 
                persona_profile_with_context, 
                job_profile_with_context
            )

            # Compose final result
            result = {
                "ranked_sections": ranked_sections,
                "relevance_scores": relevance_scores,
                "extracted_subsections": extracted_subsections,
                "metadata": extra_metadata or {},
            }
            self.logger.info("Persona-driven intelligence pipeline completed successfully.")
            return result
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            # TODO: Add more granular error handling as needed
            raise
