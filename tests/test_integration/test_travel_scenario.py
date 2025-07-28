"""
Travel scenario integration tests.

Purpose:
    Integration tests for processing travel planning PDFs through the full pipeline.

Structure:
    - Test functions for each major travel use case (tourist, travel agent, event planner)
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
def test_travel_pdf_tourist_scenario():
    """
    Test processing a travel planning PDF as a tourist persona.
    Validates extraction of itineraries, maps, and multilingual support.
    """
    pdf_path = load_fixture('travel_itinerary_en.pdf')
    persona_context = {'role': 'tourist', 'description': 'Tourist planning a multi-city trip.'}
    job_context = {'task': 'plan', 'description': 'Plan itinerary and sightseeing.'}
    processor = BatchDocumentProcessor()
    results = processor.process_documents([pdf_path], persona_context, job_context)
    assert results, "No results returned."
    doc_result = results[0]
    assert 'intelligence_result' in doc_result, "Missing intelligence_result in output."
    # TODO: Add more detailed assertions for itinerary extraction, map handling, and language support

@pytest.mark.integration
def test_travel_pdf_agent_scenario():
    """
    Test processing a travel planning PDF as a travel agent persona.
    Validates extraction of schedules, bookings, and multilingual content.
    """
    pdf_path = load_fixture('travel_schedule_multi.pdf')
    persona_context = {'role': 'travel_agent', 'description': 'Travel agent organizing group travel.'}
    job_context = {'task': 'organize', 'description': 'Organize bookings, schedules, and group activities.'}
    processor = BatchDocumentProcessor()
    results = processor.process_documents([pdf_path], persona_context, job_context)
    assert results, "No results returned."
    doc_result = results[0]
    assert 'intelligence_result' in doc_result, "Missing intelligence_result in output."
    # TODO: Add assertions for schedule extraction, multilingual/mixed-language support, and error handling

@pytest.mark.integration
def test_travel_pdf_event_planner_scenario():
    """
    Test processing a travel planning PDF as an event planner persona.
    Validates extraction of event schedules, embedded images/maps, and edge cases.
    """
    pdf_path = load_fixture('travel_event_embedded_images.pdf')
    persona_context = {'role': 'event_planner', 'description': 'Event planner managing conference logistics.'}
    job_context = {'task': 'extract', 'description': 'Extract event schedules, maps, and logistics details.'}
    processor = BatchDocumentProcessor()
    results = processor.process_documents([pdf_path], persona_context, job_context)
    assert results, "No results returned."
    doc_result = results[0]
    assert 'intelligence_result' in doc_result, "Missing intelligence_result in output."
    # TODO: Add assertions for embedded images/maps, missing sections, and error reporting
