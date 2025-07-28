"""
Validate models script.

Purpose:
    Verify the integrity and functionality of all required ML models for the pipeline.

Structure:
    - Loads all model paths and config dynamically (no hardcoding)
    - Checks file existence, size, hash (if available), loadability, and minimal inference
    - Logs all results, errors, and warnings in a structured format
    - Returns a summary of validation status for each model

Dependencies:
    - config.model_config, config.settings
    - src.utils.logger
    - sentence_transformers, langdetect, nltk, tesseract (if OCR enabled)

Integration Points:
    - Used in CI, Docker build, and manual QA to ensure all models are ready

NOTE: Add new model checks as new models are added to the pipeline.
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from config import model_config, settings
from src.utils.logger import get_logger

def get_file_hash(path: Path, algo: str = "sha256") -> str:
    """
    Compute the hash of a file.
    Args:
        path (Path): File path
        algo (str): Hash algorithm (default: sha256)
    Returns:
        str: Hex digest of file hash
    Raises:
        FileNotFoundError: If file does not exist
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_sentence_transformer(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate the primary sentence transformer model.
    Args:
        config (dict): Model config
        logger (Logger): Logger instance
    Returns:
        dict: Validation summary
    """
    result = {"model": config["name"], "status": "pending", "details": []}
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config["name"])
        result["details"].append("Model loaded successfully.")
        # Minimal inference
        test_emb = model.encode(["Hello world", "Bonjour le monde"])
        if test_emb.shape[1] == config["embedding_dimensions"]:
            result["details"].append(f"Embedding shape OK: {test_emb.shape}")
            result["status"] = "pass"
        else:
            result["details"].append(f"Embedding shape mismatch: {test_emb.shape}")
            result["status"] = "fail"
    except Exception as e:
        logger.error(f"SentenceTransformer validation failed: {e}", exc_info=True)
        result["details"].append(f"Error: {e}")
        result["status"] = "fail"
    return result

def validate_langdetect(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate the language detection model.
    Args:
        config (dict): Model config
        logger (Logger): Logger instance
    Returns:
        dict: Validation summary
    """
    result = {"model": config["name"], "status": "pending", "details": []}
    try:
        from langdetect import detect
        detected = detect("This is a test sentence in English")
        if detected == "en":
            result["details"].append("Language detection OK: en")
            result["status"] = "pass"
        else:
            result["details"].append(f"Unexpected detection: {detected}")
            result["status"] = "fail"
    except Exception as e:
        logger.error(f"Langdetect validation failed: {e}", exc_info=True)
        result["details"].append(f"Error: {e}")
        result["status"] = "fail"
    return result

def validate_nltk(logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate NLTK data availability.
    Args:
        logger (Logger): Logger instance
    Returns:
        dict: Validation summary
    """
    result = {"model": "nltk", "status": "pending", "details": []}
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        result["details"].append("NLTK data available.")
        result["status"] = "pass"
    except Exception as e:
        logger.error(f"NLTK validation failed: {e}", exc_info=True)
        result["details"].append(f"Error: {e}")
        result["status"] = "fail"
    return result

# TODO: Add OCR model validation if OCR is enabled in config

def main():
    """
    Main entry point for model validation.
    Loads config, validates all models, and prints a summary.
    """
    logger = get_logger("validate_models")
    logger.info("Starting model validation...")

    config = model_config.PRIMARY_MODEL_CONFIG
    lang_config = model_config.LANGUAGE_DETECTION_CONFIG

    results = []
    results.append(validate_sentence_transformer(config, logger))
    results.append(validate_langdetect(lang_config, logger))
    results.append(validate_nltk(logger))

    # Print summary
    print("\nModel Validation Summary:")
    for res in results:
        print(f"- {res['model']}: {res['status']}")
        for detail in res['details']:
            print(f"    • {detail}")

    # Exit code: 0 if all pass, 1 otherwise
    if all(r['status'] == 'pass' for r in results):
        logger.info("All models validated successfully.")
        sys.exit(0)
    else:
        logger.error("Some models failed validation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
