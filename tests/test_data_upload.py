"""
Test module for the data_upload component.

Run this directly with:
python -m tests.test_data_upload
"""

import sys
import os

# Add project root to path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

# This will test importing the component 
from ..src.ui.components.data_upload import render_file_upload, render_sample_data_option, render_data_upload

def main():
    """Verify that the data_upload component imports work correctly."""
    print("✅ Successfully imported data_upload components")
    
    # Check function signatures
    print(f"render_file_upload: {render_file_upload.__annotations__}")
    print(f"render_sample_data_option: {render_sample_data_option.__annotations__}")
    print(f"render_data_upload: {render_data_upload.__annotations__}")
    
    print("✅ Test completed successfully")
    
if __name__ == "__main__":
    main()
