"""
File handler - ENHANCED from Round 1A.

Purpose:
    Provides utilities for reading, writing, validating, and managing multiple files (PDFs, JSON, models, etc.).
    Supports batch operations and error recovery.

Structure:
    - read_file: function to read a file (text or binary)
    - write_file: function to write data to a file
    - validate_file: function to check file existence and type
    - batch_read_files: function to read multiple files

Dependencies:
    - config.settings (for file config)
    - src.utils.logger (for logging)
    - os, json

Integration Points:
    - Used by pipeline components for file I/O and validation
    - Supports all file types used in the project

NOTE: No hardcoding; all paths and config are loaded dynamically.
"""

from typing import Any, Dict, List, Optional, Union
import os
import json
import logging
from config import settings
from src.utils.logger import get_logger

def read_file(path: str, mode: str = 'r', encoding: Optional[str] = 'utf-8') -> Union[str, bytes]:
    """
    Read a file (text or binary).

    Args:
        path (str): Path to the file.
        mode (str): File mode ('r', 'rb', etc.).
        encoding (str, optional): Encoding for text files.

    Returns:
        str or bytes: File contents.

    Raises:
        FileNotFoundError: If file does not exist.
        IOError: If file cannot be read.
    """
    logger = get_logger(__name__)
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    try:
        if 'b' in mode:
            with open(path, mode) as f:
                return f.read()
        else:
            with open(path, mode, encoding=encoding) as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}", exc_info=True)
        raise

def write_file(path: str, data: Union[str, bytes], mode: str = 'w', encoding: Optional[str] = 'utf-8') -> None:
    """
    Write data to a file (text or binary).

    Args:
        path (str): Path to the file.
        data (str or bytes): Data to write.
        mode (str): File mode ('w', 'wb', etc.).
        encoding (str, optional): Encoding for text files.

    Raises:
        IOError: If file cannot be written.
    """
    logger = get_logger(__name__)
    try:
        if 'b' in mode:
            with open(path, mode) as f:
                f.write(data)
        else:
            with open(path, mode, encoding=encoding) as f:
                f.write(data)
        logger.info(f"Wrote file: {path}")
    except Exception as e:
        logger.error(f"Failed to write file {path}: {e}", exc_info=True)
        raise

def validate_file(path: str, file_type: Optional[str] = None) -> bool:
    """
    Validate that a file exists and optionally matches a type.

    Args:
        path (str): Path to the file.
        file_type (str, optional): Expected file type ('pdf', 'json', etc.).

    Returns:
        bool: True if file exists (and matches type if specified), False otherwise.
    """
    logger = get_logger(__name__)
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return False
    if file_type:
        ext = os.path.splitext(path)[1].lower()
        if file_type == 'pdf' and ext != '.pdf':
            logger.warning(f"File {path} is not a PDF.")
            return False
        if file_type == 'json' and ext != '.json':
            logger.warning(f"File {path} is not a JSON file.")
            return False
    return True

def batch_read_files(paths: List[str], mode: str = 'r', encoding: Optional[str] = 'utf-8') -> List[Union[str, bytes, None]]:
    """
    Read multiple files in batch, returning contents or None for failures.

    Args:
        paths (list): List of file paths.
        mode (str): File mode ('r', 'rb', etc.).
        encoding (str, optional): Encoding for text files.

    Returns:
        list: List of file contents or None for failed reads.
    """
    results = []
    for path in paths:
        try:
            results.append(read_file(path, mode, encoding))
        except Exception as e:
            get_logger(__name__).error(f"Batch read failed for {path}: {e}")
            results.append(None)
    return results
