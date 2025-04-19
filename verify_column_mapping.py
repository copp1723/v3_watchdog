#!/usr/bin/env python3
"""
Simple smoke test for column mapping functionality.
"""

import pandas as pd
from src.llm_engine import LLMEngine
from src.utils.config import DROP_UNMAPPED_COLUMNS
import json

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
        print("✅ PASS: Column mapping function call succeeded")
        
        # Print info about DROP_UNMAPPED_COLUMNS setting
        print(f"DROP_UNMAPPED_COLUMNS setting: {DROP_UNMAPPED_COLUMNS}")
        
        # Basic result checking
        if isinstance(result, dict):
            print("✅ PASS: Result is a dictionary")
            return True
        else:
            print("❌ FAIL: Result is not a dictionary")
            return False
    
    except Exception as e:
        print(f"❌ FAIL: Column mapping failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("Running column mapping smoke test...")
    success = test_column_mapping()
    print("Test completed.")