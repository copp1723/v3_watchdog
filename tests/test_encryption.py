"""
Tests for the encryption utilities.
"""

import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils.encryption import (
    encrypt_bytes, decrypt_bytes, encrypt_file, decrypt_file,
    read_encrypted_csv, save_encrypted_csv
)


class TestEncryption(unittest.TestCase):
    """Test cases for the encryption utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Sample data
        self.sample_text = "This is sensitive test data for encryption."
        self.sample_bytes = self.sample_text.encode('utf-8')
        
        # Sample CSV data
        self.sample_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def test_encrypt_decrypt_bytes(self):
        """Test encrypting and decrypting bytes."""
        # Encrypt the sample data
        encrypted_data = encrypt_bytes(self.sample_bytes)
        
        # The encrypted data should be different from the original
        self.assertNotEqual(encrypted_data, self.sample_bytes)
        
        # Decrypt the data
        decrypted_data = decrypt_bytes(encrypted_data)
        
        # The decrypted data should match the original
        self.assertEqual(decrypted_data, self.sample_bytes)
        self.assertEqual(decrypted_data.decode('utf-8'), self.sample_text)
    
    def test_encrypt_decrypt_file(self):
        """Test encrypting and decrypting a file."""
        # Create a test file
        test_file_path = os.path.join(self.test_dir, 'test_file.txt')
        with open(test_file_path, 'w') as f:
            f.write(self.sample_text)
        
        # Encrypt the file
        encrypted_file_path = encrypt_file(test_file_path)
        
        # Check that the encrypted file exists
        self.assertTrue(os.path.exists(encrypted_file_path))
        
        # The encrypted file should be different from the original
        with open(encrypted_file_path, 'rb') as f:
            encrypted_content = f.read()
        self.assertNotEqual(encrypted_content, self.sample_bytes)
        
        # Decrypt the file
        decrypted_file_path = decrypt_file(encrypted_file_path)
        
        # Check that the decrypted file exists
        self.assertTrue(os.path.exists(decrypted_file_path))
        
        # The decrypted content should match the original
        with open(decrypted_file_path, 'r') as f:
            decrypted_content = f.read()
        self.assertEqual(decrypted_content, self.sample_text)
    
    def test_csv_encryption(self):
        """Test pandas DataFrame encryption and decryption to CSV."""
        # Create a file path for the encrypted CSV
        csv_path = os.path.join(self.test_dir, 'test_data.csv.enc')
        
        # Save the DataFrame as an encrypted CSV
        save_encrypted_csv(self.sample_df, csv_path, index=False)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(csv_path))
        
        # Read the encrypted CSV
        loaded_df = read_encrypted_csv(csv_path)
        
        # The loaded DataFrame should match the original
        pd.testing.assert_frame_equal(loaded_df, self.sample_df)
    
    def test_invalid_decrypt(self):
        """Test decrypting invalid data."""
        # Create some invalid encrypted data
        invalid_data = b'invalid-encrypted-data'
        
        # Attempting to decrypt it should raise an exception
        with self.assertRaises(ValueError):
            decrypt_bytes(invalid_data)
    
    def test_empty_data(self):
        """Test handling empty data."""
        # Empty data should be returned as-is
        self.assertEqual(encrypt_bytes(b''), b'')
        self.assertEqual(decrypt_bytes(b''), b'')


if __name__ == '__main__':
    unittest.main()