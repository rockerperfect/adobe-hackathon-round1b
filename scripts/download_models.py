"""
Download models script.
Pre-download models during Docker build to avoid runtime delays.

This script downloads all required models for Round 1B including:
- Multilingual sentence transformers model
- Language detection models
- OCR language packs

Performance: Ensures models are cached locally to meet 60s processing constraint.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from config.model_config import PRIMARY_MODEL_CONFIG, MODEL_LOADING
from src.utils.logger import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_sentence_transformers_model():
    """Download the primary multilingual sentence transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = PRIMARY_MODEL_CONFIG["name"]
        logger.info(f"Downloading model: {model_name}")
        
        # Download and cache the model
        model = SentenceTransformer(model_name)
        logger.info(f"Model {model_name} downloaded successfully ({PRIMARY_MODEL_CONFIG['size_mb']}MB)")
        
        # Test the model to ensure it works
        test_text = ["Hello world", "Test sentence"]
        embeddings = model.encode(test_text)
        logger.info(f"Model test successful - Generated embeddings: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download sentence transformers model: {e}")
        return False


def download_language_detection_models():
    """Download language detection models."""
    try:
        import nltk
        
        # Download NLTK data for language processing
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("Language detection models downloaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download language detection models: {e}")
        return False


def verify_model_downloads():
    """Verify that all models are properly downloaded and functional."""
    logger.info("Verifying model downloads...")
    
    success = True
    
    # Test sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(PRIMARY_MODEL_CONFIG["name"])
        logger.info("✓ Sentence transformers model verified")
    except Exception as e:
        logger.error(f"✗ Sentence transformers verification failed: {e}")
        success = False
    
    # Test language detection
    try:
        from langdetect import detect
        test_result = detect("This is a test sentence in English")
        logger.info(f"✓ Language detection verified (detected: {test_result})")
    except Exception as e:
        logger.error(f"✗ Language detection verification failed: {e}")
        success = False
    
    return success


def main():
    """Main function to download all required models."""
    logger.info("Starting model download process...")
    
    success = True
    
    # Download models
    logger.info("=== Downloading Sentence Transformers Model ===")
    if not download_sentence_transformers_model():
        success = False
    
    logger.info("=== Downloading Language Detection Models ===")
    if not download_language_detection_models():
        success = False
    
    # Verify downloads
    logger.info("=== Verifying Downloads ===")
    if not verify_model_downloads():
        success = False
    
    if success:
        logger.info("✅ All models downloaded and verified successfully!")
        return 0
    else:
        logger.error("❌ Some models failed to download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
