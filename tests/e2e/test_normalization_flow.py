import io
import unittest
import pandas as pd

from src.utils.data_io import load_data


class DummyUploadedFile:
    def __init__(self, name, content, mime_type="text/csv"):
        self.name = name
        self._content = content
        self.type = mime_type
        self._pointer = 0
    
    def getvalue(self):
        return self._content
    
    def seek(self, pos):
        self._pointer = pos


class TestNormalizationFlow(unittest.TestCase):
    def test_csv_upload_normalization(self):
        # Prepare CSV content with variant term 'autotrdr'
        csv_content = "LeadSource,TotalGross,VIN,SaleDate,SalePrice\nautotrdr,10000,1HGCM82633A004352,2023-10-01,20000\n"
        # Convert to bytes
        csv_bytes = csv_content.encode('utf-8')

        # Create dummy uploaded file object
        uploaded_file = DummyUploadedFile(name="test_upload.csv", content=csv_bytes)

        # Call load_data to process the upload
        df = load_data(uploaded_file)

        # Assert that 'LeadSource' column is normalized to canonical value 'Auto Trader'
        # Note: This expects that the normalization rules convert "autotrdr" to "Auto Trader".
        self.assertIn('LeadSource', df.columns)
        normalized_value = df['LeadSource'].iloc[0]
        self.assertEqual(normalized_value, "Auto Trader")


if __name__ == '__main__':
    unittest.main() 