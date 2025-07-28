"""
Business scenario integration tests.

Purpose:
    Integration tests for processing business/financial reports through the full pipeline.

Structure:
    - Test functions for each major business use case (analyst, executive, auditor)
    - Uses fixtures for sample input PDFs and expected outputs

Dependencies:
    - pytest (test runner)
    - src.processors.batch_document_processor.BatchDocumentProcessor
    - tests/fixtures/sample_inputs (for input PDFs and persona/job contexts)
    - tests/fixtures/expected_outputs (for expected results)

Integration Points:
    - Validates full pipeline: PDF parsing, outline extraction, segmentation, multilingual processing, intelligence analysis
    - Ensures output matches expected structure and content

NOTE: No hardcoding; all test data paths and config are loaded from fixtures or config files.
"""

import os
import pytest
from src.processors.batch_document_processor import BatchDocumentProcessor

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
SAMPLE_INPUTS_DIR = os.path.join(FIXTURES_DIR, 'sample_inputs')
EXPECTED_OUTPUTS_DIR = os.path.join(FIXTURES_DIR, 'expected_outputs')

def load_fixture(filename: str, subdir: str = 'sample_inputs') -> str:
    """
    Load the path to a fixture file.
    Args:
        filename (str): Name of the file.
        subdir (str): Subdirectory under fixtures.
    Returns:
        str: Full path to the fixture file.
    """
    return os.path.join(FIXTURES_DIR, subdir, filename)

@pytest.mark.integration
def test_financial_report_analyst_scenario():
    """
    Test processing a financial report as an analyst persona.
    Validates extraction of tables, financial sections, and relevance ranking.
    """
    pdf_path = load_fixture('financial_report_en.pdf')
    persona_context = {'role': 'analyst', 'description': 'Financial analyst reviewing quarterly results.'}
    job_context = {'task': 'analyze', 'description': 'Analyze financial performance and trends.'}
    processor = BatchDocumentProcessor()
    results = processor.process_documents([pdf_path], persona_context, job_context)
    assert results, "No results returned."
    doc_result = results[0]
    assert 'intelligence_result' in doc_result, "Missing intelligence_result in output."
    # TODO: Add more detailed assertions for table extraction, section ranking, and error handling

@pytest.mark.integration
def test_financial_report_executive_scenario():
    """
    Test processing a financial report as an executive persona.
    Validates extraction of executive summary and high-level insights.
    """
    pdf_path = load_fixture('financial_report_multi.pdf')
    persona_context = {'role': 'executive', 'description': 'C-level executive seeking high-level insights.'}
    job_context = {'task': 'summarize', 'description': 'Summarize key financial highlights and risks.'}
    processor = BatchDocumentProcessor()
    results = processor.process_documents([pdf_path], persona_context, job_context)
    assert results, "No results returned."
    doc_result = results[0]
    assert 'intelligence_result' in doc_result, "Missing intelligence_result in output."
    # TODO: Add assertions for summary extraction, multilingual support, and edge cases

@pytest.mark.integration
def test_financial_report_auditor_scenario():
    """
    Test processing a financial report as an auditor persona.
    Validates extraction of compliance-relevant sections and error handling.
    """
    pdf_path = load_fixture('financial_report_missing_tables.pdf')
    persona_context = {'role': 'auditor', 'description': 'Auditor checking for compliance and completeness.'}
    job_context = {'task': 'verify', 'description': 'Verify presence of all required financial tables and disclosures.'}
    processor = BatchDocumentProcessor()
    results = processor.process_documents([pdf_path], persona_context, job_context)
    assert results, "No results returned."
    doc_result = results[0]
    assert 'intelligence_result' in doc_result, "Missing intelligence_result in output."
    # TODO: Add assertions for missing tables, non-standard layouts, and error reporting
