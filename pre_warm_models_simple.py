#!/usr/bin/env python3
"""
Pre-warm all models during Docker build to eliminate runtime loading time.
This script should eliminate the 55-57 second model loading overhead,
allowing for 20-30 second processing times with full quality.
"""

import logging
import sys
import os
import traceback

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Pre-warm all critical models for optimal runtime performance."""
    try:
        logger.info("🔥 Starting model pre-warming for Adobe Hackathon Pipeline...")
        
        # Pre-warm sentence transformer model (most critical - takes 55+ seconds)
        logger.info("Pre-warming sentence transformer model...")
        from src.nlp.model_singleton import ModelSingleton
        
        # This will force the model to load and cache
        singleton = ModelSingleton()
        model = singleton.get_model()
        logger.info("✅ Singleton model pre-warmed successfully")
        
        # Test a quick embedding to ensure everything works
        logger.info("Testing model with sample text...")
        test_embedding = model.encode(["Testing model pre-warming"], show_progress_bar=False)
        logger.info(f"✅ Test embedding successful - shape: {test_embedding.shape}")
        
        logger.info("🚀 All models pre-warmed successfully! Runtime should be 20-30s per collection.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model pre-warming failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    logger.info("✅ Pre-warming completed successfully!")
