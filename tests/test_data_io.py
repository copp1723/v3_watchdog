import pytest
import pandas as pd
import io
from pandas.errors import ParserError

# Import the load_data function and ValidationError
from src.utils.data_io import load_data, ValidationError

# Helper class to mock Streamlit UploadedFile
class MockUploadedFile:
    def __init__(self, content: bytes, name: str, mime: str = None):
        self._io = io.BytesIO(content)
        self.name = name
        if mime:
            self.type = mime
    def getvalue(self) -> bytes:
        return self._io.getvalue()
    def seek(self, pos: int):
        return self._io.seek(pos)
    def __iter__(self):
        return iter(self._io)

# Utility to create a valid CSV upload
def make_csv_file(columns, rows, name="test.csv", mime="text/csv"):
    df = pd.DataFrame(rows, columns=columns)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue().encode('utf-8')
    return MockUploadedFile(data, name, mime)

# Utility to create a valid XLSX upload
def make_xlsx_file(columns, rows, name="test.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
    df = pd.DataFrame(rows, columns=columns)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return MockUploadedFile(buf.getvalue(), name, mime)

# Test successful CSV upload
def test_load_data_valid_csv_success(monkeypatch):
    columns = ['LeadSource','TotalGross','VIN','SaleDate','SalePrice']
    rows = [['web', 100.0, 'V1', '2023-01-01', 15000]]
    uploaded = make_csv_file(columns, rows)

    # Patch Sentry logging
    sentry_calls = []
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.capture_message', lambda msg, level=None: sentry_calls.append((msg, level)))

    df = load_data(uploaded)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == columns
    # Ensure Sentry success logged
    assert any("Upload succeeded: test.csv" in msg for msg, level in sentry_calls)

# Test successful XLSX upload
def test_load_data_valid_xlsx_success(monkeypatch):
    columns = ['LeadSource','TotalGross','VIN','SaleDate','SalePrice']
    rows = [['referral', 200.5, 'V2', '2023-02-02', 20000]]
    uploaded = make_xlsx_file(columns, rows)

    sentry_calls = []
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.capture_message', lambda msg, level=None: sentry_calls.append((msg, level)))

    df = load_data(uploaded)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == columns
    assert any("Upload succeeded: test.xlsx" in msg for msg, level in sentry_calls)

# Test unsupported file format
def test_load_data_unsupported_format(monkeypatch):
    content = b"dummy data"
    uploaded = MockUploadedFile(content, "bad.txt", mime="text/plain")

    sentry_errors = []
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.capture_message', lambda msg, level=None: sentry_errors.append((msg, level)))

    with pytest.raises(ValueError) as exc:
        load_data(uploaded)
    assert "Unsupported file format" in str(exc.value)
    assert any("Upload failed: bad.txt" in msg for msg, level in sentry_errors)

# Test parse error for corrupted CSV
def test_load_data_corrupted_csv_parse_error(monkeypatch):
    # Create a valid-like CSV but patch read_csv to throw
    columns = ['LeadSource','TotalGross','VIN','SaleDate','SalePrice']
    rows = [['web', 100.0, 'V1', '2023-01-01', 15000]]
    uploaded = make_csv_file(columns, rows)

    # Force ParserError
    monkeypatch.setattr('src.utils.data_io.pd.read_csv', lambda *args, **kwargs: (_ for _ in ()).throw(ParserError("bad CSV")))

    sentry_excs = []
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.capture_exception', lambda e: sentry_excs.append(e))

    with pytest.raises(ValueError) as exc:
        load_data(uploaded)
    assert "Error parsing CSV file" in str(exc.value)
    assert len(sentry_excs) >= 1

# Test parse error for corrupted Excel
def test_load_data_corrupted_excel_parse_error(monkeypatch):
    # Create a dummy XLSX-like file
    uploaded = MockUploadedFile(b"not an excel file", "bad.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Force read_excel to throw
    monkeypatch.setattr('src.utils.data_io.pd.read_excel', lambda *args, **kwargs: (_ for _ in ()).throw(OSError("bad xlsx")))

    sentry_excs = []
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.capture_exception', lambda e: sentry_excs.append(e))

    with pytest.raises(ValueError) as exc:
        load_data(uploaded)
    assert "Error reading Excel file" in str(exc.value)
    assert len(sentry_excs) >= 1

def test_load_data_post_schema_normalization(monkeypatch):
    # Create CSV with correct headers but alias cell value for LeadSource
    columns = ['LeadSource','TotalGross','VIN','SaleDate','SalePrice']
    rows = [['suv', 150.0, 'V1', '2023-01-01', 10000]]
    uploaded = make_csv_file(columns, rows)

    # Capture Sentry tags
    sentry_tags = []
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.set_tag', lambda key, value: sentry_tags.append((key, value)))
    # Prevent duplicate capture_message noise
    monkeypatch.setattr('src.utils.data_io.sentry_sdk.capture_message', lambda msg, level=None: None)

    df = load_data(uploaded)
    # Cell-level normalization should convert 'suv' -> 'SUV'
    assert df['LeadSource'].iloc[0] == 'SUV'
    # Sentry tags for normalization should have been set
    assert any(tag == 'normalization_step' for tag, _ in sentry_tags)
    assert any(tag == 'normalization_rules_version' for tag, _ in sentry_tags) 