#!/usr/bin/env python3
"""
Smoke test script for Watchdog AI.
Verifies core functionality after Phase 0 changes.
"""

import os
import sys
import pandas as pd
import json
import hashlib
import traceback
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('smoke_test.log')
    ]
)

logger = logging.getLogger("smoke_test")

def print_header(text):
    """Print a section header."""
    border = "=" * (len(text) + 4)
    print("\n" + border)
    print(f"  {text}  ")
    print(border + "\n")

def test_normalization():
    """Test term normalization functionality."""
    print_header("TESTING TERM NORMALIZATION")
    
    try:
        from src.utils.term_normalizer import normalize_terms
        
        # Create test DataFrame with term variations
        test_df = pd.DataFrame({
            'LeadSource': ['autotrader', 'Auto-Trader', 'AT.com', 'auto trader', 'AutoTrader'],
            'SalesRep': ['John Smith', 'sales_rep', 'Jane Doe', 'sales representative', 'salesperson']
        })
        
        # Apply normalization
        normalized_df = normalize_terms(test_df)
        
        # Print results
        print("Original values:")
        print(test_df['LeadSource'].tolist())
        print("\nNormalized values:")
        print(normalized_df['LeadSource'].tolist())
        
        # Verify normalization worked
        assert 'Auto Trader' in normalized_df['LeadSource'].tolist(), "Term normalization failed for 'autotrader'"
        assert 'Sales Rep' in normalized_df['SalesRep'].tolist(), "Term normalization failed for 'sales_rep'"
        
        print("\n✅ Term normalization test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Term normalization test FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_encryption():
    """Test file encryption functionality."""
    print_header("TESTING FILE ENCRYPTION")
    
    try:
        from src.utils.encryption import encrypt_bytes, decrypt_bytes
        
        # Create test data
        test_data = "Test data for encryption and decryption".encode('utf-8')
        
        # Encrypt data
        encrypted_data = encrypt_bytes(test_data)
        print(f"Original data: {test_data}")
        print(f"Encrypted data: {encrypted_data[:30]}...")
        
        # Verify data was encrypted (should be different)
        assert encrypted_data != test_data, "Data was not encrypted"
        
        # Decrypt data
        decrypted_data = decrypt_bytes(encrypted_data)
        print(f"Decrypted data: {decrypted_data}")
        
        # Verify decryption
        assert decrypted_data == test_data, "Decryption failed to restore original data"
        
        print("\n✅ Encryption test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Encryption test FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_file_upload():
    """Test file upload with encryption."""
    print_header("TESTING FILE UPLOAD WITH ENCRYPTION")
    
    try:
        # Create a temporary test CSV
        import tempfile
        
        test_data = pd.DataFrame({
            'VIN': ['1HGCM82633A123456', '5TFBW5F13AX123457'],
            'Make': ['Honda', 'Toyota'],
            'Model': ['Accord', 'Tundra'],
            'Year': [2019, 2020],
            'Sale_Date': ['2023-01-15', '2023-02-20'],
            'Sale_Price': [28500.00, 45750.00],
            'Gross_Profit': [3500.00, 5750.00],
            'Lead_Source': ['autotrader', 'Walk-in']
        })
        
        temp_dir = tempfile.mkdtemp()
        test_csv_path = os.path.join(temp_dir, "test_upload.csv")
        test_data.to_csv(test_csv_path, index=False)
        
        print(f"Created test CSV at {test_csv_path}")
        
        # Create a mock file object similar to what Streamlit's file_uploader provides
        class MockUploadedFile:
            def __init__(self, path, name):
                self.path = path
                self.name = name
                self._file = open(path, 'rb')
            
            def read(self):
                self._file.seek(0)
                return self._file.read()
            
            def getvalue(self):
                self._file.seek(0)
                return self._file.read()
                
            def seek(self, position):
                self._file.seek(position)
                
            def tell(self):
                return self._file.tell()
                
            def close(self):
                self._file.close()
        
        # Mock the Streamlit file uploader
        mock_file = MockUploadedFile(test_csv_path, "test_upload.csv")
        
        # Import and use data_io module
        from src.utils.data_io import load_data
        
        # Monkey patch st.cache_data to be a no-op decorator
        import streamlit as st
        if not hasattr(st, 'cache_data'):
            st.cache_data = lambda f: f
        
        # Load and process the data
        df = load_data(mock_file)
        
        print("\nLoaded data:")
        print(df.head())
        
        # Check for encrypted file
        uploads_dir = os.path.join('data', 'uploads')
        encrypted_files = []
        for root, _, files in os.walk(uploads_dir):
            for file in files:
                if file.endswith('.enc'):
                    encrypted_files.append(os.path.join(root, file))
        
        print(f"\nFound {len(encrypted_files)} encrypted files:")
        for file in encrypted_files:
            print(f"- {file}")
        
        assert len(encrypted_files) > 0, "No encrypted files found"
        
        # Clean up
        mock_file.close()
        
        print("\n✅ File upload with encryption test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ File upload with encryption test FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_consolidated_modules():
    """Test consolidated modules."""
    print_header("TESTING CONSOLIDATED MODULES")
    
    try:
        # Import consolidated modules
        from src.insight_card_consolidated import InsightOutputFormatter
        
        try:
            from src.insight_conversation_consolidated import ConversationManager
            from src.ui.components.chat_interface_consolidated import ChatInterface
            # Verify ConversationManager
            conversation_manager = ConversationManager(use_mock=True)
            print("Created ConversationManager")
        except Exception as e:
            print(f"Note: Couldn't create ConversationManager due to missing dependencies: {str(e)}")
            print("This is expected during initial testing")
        
        # Verify InsightOutputFormatter
        formatter = InsightOutputFormatter()
        test_response = {
            "summary": "Test summary",
            "value_insights": ["Test insight 1", "Test insight 2"],
            "actionable_flags": ["Test action 1"],
            "confidence": "high"
        }
        formatted = formatter.format_response(json.dumps(test_response))
        print("\nFormatted response:", formatted["summary"])
        
        # Create ChatInterface (testing imports only)
        chat_interface = ChatInterface()
        print("Created ChatInterface")
        
        print("\n✅ Consolidated modules test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Consolidated modules test FAILED: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all smoke tests."""
    print_header("WATCHDOG AI PHASE 0 SMOKE TEST")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "normalization": test_normalization(),
        "encryption": test_encryption(),
        "file_upload": test_file_upload(),
        "consolidated_modules": test_consolidated_modules()
    }
    
    print_header("SMOKE TEST RESULTS")
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All smoke tests PASSED!")
        return 0
    else:
        print("\n⚠️ Some smoke tests FAILED! Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())