"""
JSON validator - ENHANCED from Round 1A.

Purpose:
    Provides utilities for validating input and output JSON against the official Round 1B schema.

Structure:
    - validate_json: function to validate JSON against schema
    - load_schema: function to load schema from config or external file

Dependencies:
    - config.settings (for schema config)
    - jsonschema (for validation)
    - src.utils.logger (for logging)

Integration Points:
    - Used by pipeline entry/exit points to validate all input/output JSON
    - Returns detailed validation errors and warnings

NOTE: No hardcoding; all schema paths and config are loaded dynamically.
"""

from typing import Any, Dict, Optional, Tuple, List
import os
import json
import logging
from config import settings
from src.utils.logger import get_logger

try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore

def load_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a JSON schema from config or external file.

    Args:
        schema_path (str, optional): Path to the schema file. If None, loads from config.

    Returns:
        dict: Loaded JSON schema.

    Raises:
        FileNotFoundError: If schema file is not found.
        json.JSONDecodeError: If schema file is invalid JSON.
    """
    config_schema_path = settings.load_config().get('json_schema_path')
    path = schema_path or config_schema_path
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"JSON schema file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_json(data: Any, schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """
    Validate input or output JSON against the official Round 1B schema.

    Args:
        data (Any): JSON data to validate.
        schema (dict, optional): JSON schema. If None, loads from config.

    Returns:
        tuple: (is_valid, errors) where is_valid is a bool and errors is a list of error messages.

    Raises:
        RuntimeError: If jsonschema is not installed.

    Edge Cases:
        - Returns False and error messages if validation fails.
        - Returns True and empty list if validation passes.
    """
    logger = get_logger(__name__)
    if jsonschema is None:
        logger.error("jsonschema package is required for JSON validation.")
        raise RuntimeError("jsonschema package is required.")
    if schema is None:
        try:
            schema = load_schema()
        except Exception as e:
            logger.error(f"Failed to load JSON schema: {e}", exc_info=True)
            return False, [str(e)]
    validator = jsonschema.Draft7Validator(schema)
    errors = [f"{e.message} (at {list(e.path)})" for e in validator.iter_errors(data)]
    if errors:
        logger.warning(f"JSON validation failed: {errors}")
        return False, errors
    logger.info("JSON validation passed.")
    return True, []
