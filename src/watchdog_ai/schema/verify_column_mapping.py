#!/usr/bin/env python3
"""
Simple smoke test for column mapping functionality.
"""

import pandas as pd
from src.llm_engine import LLMEngine
from src.utils.config import DROP_UNMAPPED_COLUMNS
import json

# Import status formatter for UI display (can be used when integrating with UI)
try:
    from watchdog_ai.ui.utils.status_formatter import StatusType, format_status_text
    HAS_STATUS_FORMATTER = True
except ImportError:
    HAS_STATUS_FORMATTER = False

def test_column_mapping():
    """Test column mapping with a simple example."""
    # Create test data
    test_columns = ["lead_source", "profit", "sold_price", "vehicle_year", "vehicle_make"]
    
    # Initialize LLM engine with mock mode and disable Redis
    engine = LLMEngine(use_mock=True, use_redis_cache=False)
    
    # Call column mapping
    print("Calling column mapping...")
    try:
        result = engine.map_columns_jeopardy(test_columns)
        print("[PASS] Column mapping function call succeeded")
        
        # Print info about DROP_UNMAPPED_COLUMNS setting
        print(f"DROP_UNMAPPED_COLUMNS setting: {DROP_UNMAPPED_COLUMNS}")
        
        # Basic result checking
        if isinstance(result, dict):
            print("[PASS] Result is a dictionary")
            return True
        else:
            print("[FAIL] Result is not a dictionary")
            return False
    
    except Exception as e:
        print(f"[FAIL] Column mapping failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("Running column mapping smoke test...")
    success = test_column_mapping()
    print("Test completed.")
    
    # Example of how to use StatusFormatter when integrated with UI
    if HAS_STATUS_FORMATTER:
        status_type = StatusType.SUCCESS if success else StatusType.ERROR
        status_message = "Column mapping verification completed successfully" if success else "Column mapping verification failed"
        formatted_message = format_status_text(status_type, custom_text=status_message)
        print(f"UI status message would be: {formatted_message}")
