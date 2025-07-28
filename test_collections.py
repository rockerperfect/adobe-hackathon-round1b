#!/usr/bin/env python3
"""
Test script for multi-collection support.

Purpose:
    Creates a sample collection structure and tests the multi-collection processing functionality.

Structure:
    Creates test collections in the required format and validates processing.

Usage:
    python test_collections.py
"""

import json
import os
import shutil
from pathlib import Path
import tempfile

def create_test_collections():
    """Create test collection structure for validation."""
    
    # Create temporary directory for testing
    test_dir = Path("test_collections_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    print(f"Creating test collections in: {test_dir.absolute()}")
    
    # Collection 1: Travel Planning (copy from current setup)
    collection1_dir = test_dir / "Collection 1"
    collection1_dir.mkdir()
    
    # Create PDFs subdirectory
    pdfs_dir = collection1_dir / "PDFs"
    pdfs_dir.mkdir()
    
    # Copy existing PDFs to PDFs subdirectory
    input_dir = Path("input")
    if input_dir.exists():
        for pdf_file in input_dir.glob("*.pdf"):
            shutil.copy2(pdf_file, pdfs_dir)
    
    # Copy existing input JSON
    existing_input = Path("input/challenge1b_input.json")
    if existing_input.exists():
        shutil.copy2(existing_input, collection1_dir / "challenge1b_input.json")
    else:
        # Create minimal input JSON
        sample_input = {
            "challenge_info": {
                "test_case_name": "travel_planner",
                "challenge_id": "round_1b_002"
            },
            "documents": [
                {"filename": "South of France - Cities.pdf"},
                {"filename": "South of France - Cuisine.pdf"},
                {"filename": "South of France - History.pdf"},
                {"filename": "South of France - Restaurants and Hotels.pdf"},
                {"filename": "South of France - Things to Do.pdf"},
                {"filename": "South of France - Tips and Tricks.pdf"},
                {"filename": "South of France - Traditions and Culture.pdf"}
            ],
            "persona": {
                "role": "Travel Planner"
            },
            "job_to_be_done": {
                "task": "Plan a trip of 4 days for a group of 10 college friends."
            }
        }
        with open(collection1_dir / "challenge1b_input.json", 'w') as f:
            json.dump(sample_input, f, indent=2)
    
    # Collection 2: Sample HR Collection (minimal test)
    collection2_dir = test_dir / "Collection 2"
    collection2_dir.mkdir()
    
    pdfs_dir2 = collection2_dir / "PDFs"
    pdfs_dir2.mkdir()
    
    # Create a minimal input for collection 2
    sample_input2 = {
        "challenge_info": {
            "test_case_name": "hr_professional", 
            "challenge_id": "round_1b_003"
        },
        "documents": [
            {"filename": "sample_hr_guide.pdf"}
        ],
        "persona": {
            "role": "HR Professional"
        },
        "job_to_be_done": {
            "task": "Create and manage fillable forms for onboarding and compliance"
        }
    }
    
    with open(collection2_dir / "challenge1b_input.json", 'w') as f:
        json.dump(sample_input2, f, indent=2)
    
    # Create a dummy PDF for collection 2 (copy one from collection 1)
    if list(pdfs_dir.glob("*.pdf")):
        first_pdf = list(pdfs_dir.glob("*.pdf"))[0]
        shutil.copy2(first_pdf, pdfs_dir2 / "sample_hr_guide.pdf")
    
    print(f"Created test collections:")
    print(f"  - {collection1_dir}")
    print(f"  - {collection2_dir}")
    
    return test_dir

def test_collections_processing(collections_dir):
    """Test the collections processing functionality."""
    print(f"\nTesting collections processing...")
    
    # Import and run the main function with collections argument
    import subprocess
    import sys
    
    cmd = [sys.executable, "main.py", "--collections_dir", str(collections_dir)]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if output files were created
        collection1_output = collections_dir / "Collection 1" / "challenge1b_output.json"
        collection2_output = collections_dir / "Collection 2" / "challenge1b_output.json"
        
        if collection1_output.exists():
            print(f"✅ Collection 1 output created: {collection1_output}")
            with open(collection1_output) as f:
                output_data = json.load(f)
                print(f"   - Extracted sections: {len(output_data.get('extracted_sections', []))}")
                print(f"   - Subsection analysis: {len(output_data.get('subsection_analysis', []))}")
        else:
            print(f"❌ Collection 1 output missing: {collection1_output}")
        
        if collection2_output.exists():
            print(f"✅ Collection 2 output created: {collection2_output}")
            with open(collection2_output) as f:
                output_data = json.load(f)
                print(f"   - Extracted sections: {len(output_data.get('extracted_sections', []))}")
                print(f"   - Subsection analysis: {len(output_data.get('subsection_analysis', []))}")
        else:
            print(f"❌ Collection 2 output missing: {collection2_output}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (300 seconds)")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def cleanup_test_collections(test_dir):
    """Clean up test collections directory."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")

def main():
    """Main test function."""
    print("="*60)
    print("MULTI-COLLECTION SUPPORT TEST")
    print("="*60)
    
    try:
        # Create test collections
        test_dir = create_test_collections()
        
        # Test processing
        success = test_collections_processing(test_dir)
        
        if success:
            print("\n✅ Multi-collection support test PASSED!")
        else:
            print("\n❌ Multi-collection support test FAILED!")
            return 1
        
    finally:
        # Cleanup
        if 'test_dir' in locals():
            cleanup_test_collections(test_dir)
    
    return 0

if __name__ == "__main__":
    exit(main())
