# Watchdog AI Test Suite

This directory contains the comprehensive test suite for the Watchdog AI data processing engine.

## Test Structure

The test suite is organized to match the module structure of the codebase:

- `test_data_cleaner.py`: Tests for data cleaning and normalization functionality
- `test_type_inference.py`: Tests for type inference and schema generation
- `test_metadata.py`: Tests for metadata extraction and management
- `test_data_parser.py`: Tests for file parsing across different formats (CSV, Excel, PDF)

## Running Tests

To run the full test suite:

```bash
cd v3watchdog_ai
pytest
```

To run a specific test module:

```bash
pytest tests/test_data_cleaner.py
```

To run a specific test class:

```bash
pytest tests/test_data_cleaner.py::TestColumnNameCleaning
```

To run a specific test function:

```bash
pytest tests/test_data_cleaner.py::TestColumnNameCleaning::test_clean_simple_column_names
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`, including:

- `clean_csv_data`: A clean DataFrame with automotive data
- `messy_csv_data`: A DataFrame with data quality issues for testing cleaners
- `sample_excel_path`: Path to a sample Excel file with multiple sheets
- `sample_clean_csv_path`: Path to a clean CSV file
- `sample_messy_csv_path`: Path to a messy CSV file

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of Arrange-Act-Assert
2. Add appropriate docstrings to test classes and methods
3. Use parametrized tests for testing multiple similar cases
4. Ensure tests handle edge cases (empty data, malformed input, etc.)
5. Add fixtures for complex test setups to `conftest.py`

## Test Data

Test data is stored in the `assets` directory:

- CSV files with clean and messy data
- Excel files with multiple sheets and varying quality
- PDF files for testing document extraction
