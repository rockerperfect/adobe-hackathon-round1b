#!/usr/bin/env python3
"""Simple validation script to check all our dependencies and imports."""

import sys
import os

def test_dependencies():
    """Test all required dependencies."""
    print("🔍 Testing Dependencies...")
    
    try:
        # Test basic Python modules
        import json
        import pathlib
        print("✅ Python stdlib: OK")
        
        # Test ML libraries
        import numpy as np
        import pandas as pd
        print("✅ Data science libs: OK")
        
        # Test NLP libraries
        import sentence_transformers
        import transformers
        import torch
        import langdetect
        import nltk
        print("✅ NLP libraries: OK")
        
        # Test PDF processing
        import fitz  # PyMuPDF
        import pdfminer
        print("✅ PDF processing: OK")
        
        # Test image processing
        import cv2
        import PIL
        print("✅ Image processing: OK")
        
        # Test utilities
        import psutil
        import jsonschema
        print("✅ Utility libraries: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False

def test_project_structure():
    """Test project structure and imports."""
    print("\n🏗️ Testing Project Structure...")
    
    try:
        # Add project root to path
        sys.path.insert(0, '.')
        
        # Test config imports
        from config import settings, model_config
        print("✅ Config modules: OK")
        
        # Test utils imports
        from src.utils import logger, performance_monitor, cache_manager
        print("✅ Utility modules: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Project Import Error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Adobe Hackathon Round 1B - Dependency Check")
    print("=" * 50)
    
    deps_ok = test_dependencies()
    structure_ok = test_project_structure()
    
    print("\n📋 Summary:")
    if deps_ok and structure_ok:
        print("🎉 All checks passed! Your environment is ready.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
