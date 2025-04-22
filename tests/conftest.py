"""
Pytest configuration file for Watchdog AI testing.

This file contains shared fixtures and configuration for all test modules.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Generator, Union, ContextManager
from datetime import datetime, timedelta
import asyncio
import logging
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import sqlalchemy as sa
import io
import random
from pathlib import Path
from faker import Faker

# Ensure src is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import validator components - these imports will be available to all tests
from src.validators.base_validator import BaseValidator
from src.validators.validator_registry import get_validator_classes, get_validators, get_validator_by_name
from src.validators.validator_service import ValidatorService

# Configure logging to prevent log noise during tests
logging.basicConfig(level=logging.INFO)
logging.getLogger("watchdog_ai").setLevel(logging.WARNING)

# Initialize faker instance for data generation
faker = Faker()

# Get project root directory
PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())

# Paths for test assets
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)

# Export all fixtures and utilities
__all__ = [
    # Existing fixtures
    'event_loop', 'sample_dataframe', 'mock_file', 'mock_validator', 'validator_service',
    'mock_llm_response', 'cleanup_test_files', 'test_data_dir', 'performance_thresholds',
    'mock_feedback_data', 'assets_dir', 'clean_csv_data', 'messy_csv_data',
    'create_sample_csv', 'sample_clean_csv_path', 'sample_messy_csv_path',
    'sample_excel_path', 'empty_dataframe', 'sample_metadata',
    # New fixtures
    'mock_db_connection', 'capture_detailed_logs', 'perf_monitor', 'data_factory',
    'mock_redis', 'mock_mongodb', 'mock_requests', 'temp_directory', 'timing',
    'mock_insight', 'mock_streamlit', 'sample_sales_data', 'sample_metrics', 'mock_openai',
    'error_tracker'
]

# Test Categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "dashboard: marks tests related to dashboard functionality"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers",
        "performance: marks performance tests"
    )
    config.addinivalue_line(
        "markers",
        "error_handling: marks error handling tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end tests"
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers",
        "flaky: Test that may fail occasionally"
    )
    config.addinivalue_line(
        "markers",
        "critical: Critical path tests that must never fail"
    )
    config.addinivalue_line(
        "markers",
        "data_validation: Tests that validate data integrity and quality"
    )
    config.addinivalue_line(
        "markers",
        "security: Tests for security aspects of the codebase"
    )
    config.addinivalue_line(
        "markers",
        "api: Tests that verify API endpoints and behavior"
    )

# --- Sample Data Fixtures ---

@pytest.fixture
def mock_insight() -> Dict[str, Any]:
    """Create mock insight data."""
    return {
        "summary": "Test insight summary",
        "metrics": {"value": 100},
        "recommendations": ["Test recommendation"],
        "confidence": "high",
        "chart_data": pd.DataFrame({
            'x': range(5),
            'y': range(5)
        })
    }

@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'product': np.random.choice(['A', 'B', 'C'], 100),
        'quantity': np.random.randint(1, 100, 100),
        'price': np.random.uniform(10, 1000, 100),
        'cost': np.random.uniform(5, 800, 100)
    })

@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing."""
    return {
        'total_sales': 150000,
        'avg_profit_margin': 0.25,
        'conversion_rate': 0.15,
        'top_product': 'Product A',
        'growth_rate': 0.08
    }

# --- Streamlit mock fixtures ---

@pytest.fixture
def mock_streamlit():
    """Mock streamlit for testing UI components."""
    with pytest.MonkeyPatch() as mp:
        # Create mock streamlit functions
        mocks = {}
        
        # Mock basic streamlit functions
        for func in ['write', 'markdown', 'header', 'subheader', 'text', 'warning', 'error', 'info', 'success']:
            mocks[func] = mp.setattr(f'streamlit.{func}', lambda *args, **kwargs: None)
            
        # Mock interactive components
        mocks['button'] = mp.setattr('streamlit.button', lambda *a, **k: False)
        mocks['checkbox'] = mp.setattr('streamlit.checkbox', lambda *a, **k: False)
        mocks['selectbox'] = mp.setattr('streamlit.selectbox', lambda *a, **k: None)
        
        # Mock containers and columns
        mocks['container'] = mp.setattr('streamlit.container', lambda: None)
        mocks['columns'] = mp.setattr('streamlit.columns', lambda *a, **k: [None] * len(a[0]))
        
        # Mock session state
        mp.setattr('streamlit.session_state', {})
        
        yield mocks


@pytest.fixture
def sales_data_dir():
    """Return the path to the sales data directory."""
    return SALES_DATA_DIR


# Data loading fixtures
@pytest.fixture
def valid_sales_df():
    """Load the valid sales CSV into a DataFrame."""
    df = pd.read_csv(VALID_SALES_FILE)
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    return df


@pytest.fixture
def invalid_sales_df():
    """Load the invalid sales CSV into a DataFrame."""
    try:
        return pd.read_csv(INVALID_SALES_FILE)
    except Exception:
        # Return a malformed DataFrame if CSV can't be parsed
        return pd.DataFrame({
            'SaleDate': ['invalid-date', 'yesterday'],
            'LeadSource': [123, 456],
            'SalesPerson': ['John Doe', 'Jane Smith'],
            'TotalGross': ['not-a-number', '$1500.75'],
            'IsSale': ['maybe', 'probably']
        })


@pytest.fixture
def missing_columns_df():
    """Load the missing columns CSV into a DataFrame."""
    return pd.read_csv(MISSING_COLUMNS_FILE)


@pytest.fixture
def empty_df():
    """Return an empty DataFrame."""
    return pd.DataFrame()


# Mock fixtures
@pytest.fixture
def mock_csv_reader(monkeypatch, request):
    """Mock CSV reader that returns the specified DataFrame."""
    df = request.param if hasattr(request, 'param') else pd.DataFrame()
    
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    with patch('pandas.read_csv', return_value=df):
        yield


# Helper functions
def load_test_csv(file_path):
    """
    Load a CSV file from the test data directory.
    
    Args:
        file_path: Path to the CSV file, relative to the test data directory
        
    Returns:
        DataFrame loaded from the CSV
    """
    full_path = os.path.join(TEST_DATA_DIR, file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Test file not found: {full_path}")
    return pd.read_csv(full_path)


def generate_date_range(start_date='2023-01-01', periods=30, freq='D'):
    """
    Generate a date range for testing.
    
    Args:
        start_date: Starting date as string (YYYY-MM-DD)
        periods: Number of dates to generate
        freq: Frequency ('D' for daily, 'M' for monthly, etc.)
        
    Returns:
        List of datetime objects
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def check_date_columns(df, date_columns):
    """
    Check that specified columns are datetime types.
    
    Args:
        df: DataFrame to check
        date_columns: List of column names expected to be dates
        
    Returns:
        True if all columns are datetime types, False otherwise
    """
    for col in date_columns:
        if col not in df.columns:
            return False
        if not pd.api.types.is_datetime64_dtype(df[col]):
            return False
    return True


def check_numeric_columns(df, numeric_columns):
    """
    Check that specified columns are numeric types.
    
    Args:
        df: DataFrame to check
        numeric_columns: List of column names expected to be numeric
        
    Returns:
        True if all columns are numeric types, False otherwise
    """
    for col in numeric_columns:
        if col not in df.columns:
            return False
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False
    return True


def validate_analysis_result(result):
    """
    Validate that an analysis result has the expected structure.
    
    Args:
        result: Analysis result dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check success flag
    if 'success' not in result:
        errors.append("Missing 'success' flag")
    
    # Check timestamp
    if 'timestamp' not in result:
        errors.append("Missing 'timestamp'")
    
    # Check for required sections when success is True
    if result.get('success', False):
        required_sections = [
            'metadata', 'warnings', 'data_quality',
            'sales_trends', 'profit_analysis', 
            'yoy_comparison', 'mom_comparison'
        ]
        for section in required_sections:
            if section not in result:
                errors.append(f"Missing required section: {section}")
    
    # Check for error when success is False
    elif 'error' not in result:
        errors.append("Missing 'error' message for failed analysis")
    
    return errors

"""
Global pytest configuration and fixtures.

This module provides shared fixtures and configurations for use across all test files.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Generator, Union, ContextManager
from datetime import datetime, timedelta
import asyncio
import logging
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import sqlalchemy as sa
import io
import random
from pathlib import Path
from faker import Faker

# Ensure src is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import validator components - these imports will be available to all tests
from src.validators.base_validator import BaseValidator
from src.validators.validator_registry import get_validator_classes, get_validators, get_validator_by_name
from src.validators.validator_service import ValidatorService

# Configure logging to prevent log noise during tests
logging.basicConfig(level=logging.INFO)
logging.getLogger("watchdog_ai").setLevel(logging.WARNING)

# Initialize faker instance for data generation
faker = Faker()

# Export all fixtures and utilities
__all__ = [
    # Existing fixtures
    'event_loop', 'sample_dataframe', 'mock_file', 'mock_validator', 'validator_service',
    'mock_llm_response', 'cleanup_test_files', 'test_data_dir', 'performance_thresholds',
    'mock_feedback_data', 'assets_dir', 'clean_csv_data', 'messy_csv_data',
    'create_sample_csv', 'sample_clean_csv_path', 'sample_messy_csv_path',
    'sample_excel_path', 'empty_dataframe', 'sample_metadata',
    # New fixtures
    'mock_db_connection', 'capture_detailed_logs', 'perf_monitor', 'data_factory',
    'mock_redis', 'mock_mongodb', 'mock_requests', 'temp_directory', 'timing',
]


# --- Asyncio Fixtures ---

@pytest.fixture(scope="function")
def event_loop():
    """
    Fixture that creates and yields an event loop for each test function.
    This is needed for pytest-asyncio to work correctly.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# --- Database Fixtures ---

@pytest.fixture
def mock_db_connection():
    """
    Create an in-memory SQLite database connection for testing.
    
    Returns:
        SQLAlchemy engine connected to an in-memory SQLite database.
        
    Example:
        def test_data_persistence(mock_db_connection):
            # Create tables
            Base.metadata.create_all(mock_db_connection)
            # Use connection for testing
            with mock_db_connection.connect() as conn:
                result = conn.execute(sa.text("SELECT 1"))
                assert result.scalar() == 1
    """
    engine = sa.create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture
def mock_redis():
    """
    Create a mock Redis client for testing.
    
    Returns:
        A mock Redis client with common methods mocked.
        
    Example:
        def test_cache(mock_redis):
            mock_redis.get.return_value = b'{"data": 42}'
            assert your_cache.fetch('key') == {'data': 42}
    """
    redis_mock = MagicMock()
    # Setup common Redis methods
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.exists.return_value = 0
    redis_mock.expire.return_value = True
    redis_mock.pipeline.return_value = redis_mock
    redis_mock.execute.return_value = []
    
    yield redis_mock


@pytest.fixture
def mock_mongodb():
    """
    Create a mock MongoDB client for testing.
    
    Returns:
        A dictionary containing mock MongoDB database and collection objects.
        
    Example:
        def test_mongo_operations(mock_mongodb):
            mock_mongodb['collection'].find_one.return_value = {'_id': 1, 'name': 'Test'}
            result = your_service.get_document(1)
            assert result['name'] == 'Test'
    """
    # Create mock MongoDB objects
    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_client = MagicMock()
    
    # Configure collection methods
    mock_collection.find_one.return_value = None
    mock_collection.find.return_value = []
    mock_collection.insert_one.return_value = MagicMock(inserted_id=1)
    mock_collection.update_one.return_value = MagicMock(modified_count=1)
    mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
    
    # Configure DB to return the mock collection
    mock_db.__getitem__.return_value = mock_collection
    mock_client.__getitem__.return_value = mock_db
    
    yield {
        'client': mock_client,
        'db': mock_db,
        'collection': mock_collection
    }


# --- Mock Data Fixtures ---

@pytest.fixture
def sample_dataframe():
    """
    Create a sample DataFrame with test data for validator testing.
    """
    return pd.DataFrame({
        "Gross_Profit": [1000, -100, 500, 0, 2000], 
        "Lead_Source": ["Website", None, "Google", "", "Facebook"],
        "Salesperson": ["John", "Jane", "Bob", None, "Alice"],
        "SaleDate": ["2023-01-01", "2023-02-15", "invalid-date", None, "2023-05-30"],
        "Vehicle_Make": ["Honda", "Toyota", "Ford", "Chevrolet", "BMW"],
        "Vehicle_Model": ["Civic", "Camry", "F-150", "Malibu", "X5"],
        "VIN": ["1HGCM82633A123456", "5TDZA23C13S012345", None, "", "WBAFG01070L123456"]
    })


@pytest.fixture
def mock_file():
    """
    Mock file object for testing file uploads.
    """
    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
            self.position = 0
        
        def read(self):
            return self.content
        
        def seek(self, position, whence=0):
            if whence == 0:
                self.position = position
            elif whence == 1:
                self.position += position
            elif whence == 2:
                self.position = len(self.content) + position
            return self.position
        
        def tell(self):
            return self.position
    
    # Create a simple CSV content
    content = b"Lead_Source,Gross_Profit,Salesperson,SaleDate,Vehicle_Make,Vehicle_Model,VIN\n" \
              b"Website,1000,John,2023-01-01,Honda,Civic,1HGCM82633A123456\n" \
              b"Google,500,Bob,2023-03-15,Ford,F-150,1FTFW1ET2DFA12345\n" \
              b"Facebook,2000,Alice,2023-05-30,BMW,X5,WBAFG01070L123456\n"
    
    return MockFile("test_data.csv", content)


# --- Validator Fixtures ---

@pytest.fixture
def mock_validator():
    """
    Create a simple mock validator for testing.
    """
    class MockValidator(BaseValidator):
        def __init__(self, data=None):
            super().__init__(data)
            self.validation_called = False
        
        def validate(self):
            self.validation_called = True
            return {"status": "success", "issues": []}
        
        def get_name(self) -> str:
            return "Mock Validator"
        
        def get_description(self) -> str:
            return "A mock validator for testing"
    
    return MockValidator()


@pytest.fixture
def validator_service():
    """
    Create a ValidatorService instance for testing.
    """
    # Use a test profiles directory
    test_profiles_dir = os.path.join(os.path.dirname(__file__), "test_profiles")
    
    # Ensure test profiles directory exists
    os.makedirs(test_profiles_dir, exist_ok=True)
    
    # Create service with test profiles directory
    service = ValidatorService(profiles_dir=test_profiles_dir)
    
    yield service
    
    # Cleanup (if needed)
    # This runs after the test completes


@pytest.fixture
def mock_llm_response():
    """
    Mock LLM response for testing column mapping.
    """
    return {
        "mapping": {
            "financial": {
                "gross_profit": {"column": "Gross_Profit", "confidence": 0.98},
            },
            "customer": {
                "lead_source": {"column": "Lead_Source", "confidence": 0.95},
                "salesperson": {"column": "Salesperson", "confidence": 0.97},
            },
            "vehicle": {
                "make": {"column": "Vehicle_Make", "confidence": 0.99},
                "model": {"column": "Vehicle_Model", "confidence": 0.98},
                "vin": {"column": "VIN", "confidence": 0.99},
            },
            "date": {
                "sale_date": {"column": "SaleDate", "confidence": 0.96},
            }
        },
        "clarifications": [],
        "unmapped_columns": []
    }


# --- Helper Functions ---

@pytest.fixture
def cleanup_test_files():
    """
    Fixture to clean up test files after tests run.
    """
    # Setup - nothing to do before the test
    yield
    
    # Teardown - clean up any test files
    test_files = [
        "test_results.json",
        "test_report.md"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove {file}: {e}")


@pytest.fixture
def temp_directory():
    """
    Create a temporary directory for test file operations.
    
    Returns:
        Path to a temporary directory that will be automatically cleaned up.
        
    Example:
        def test_file_processing(temp_directory):
            file_path = os.path.join(temp_directory, "test.txt")
            with open(file_path, "w") as f:
                f.write("test data")
            assert os.path.exists(file_path)
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_requests():
    """
    Mock requests library for testing HTTP interactions.
    
    Returns:
        A dictionary with mock response factory functions for different HTTP methods.
        
    Example:
        def test_api_call(mock_requests):
            mock_requests['get']('https://api.example.com/users', 
                               json={'users': [{'id': 1, 'name': 'Test User'}]})
            result = your_service.get_users()
            assert len(result) == 1
            assert result[0]['name'] == 'Test User'
    """
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('requests.put') as mock_put, \
         patch('requests.delete') as mock_delete:
        
        # Create response factory function
        def create_response(status_code=200, json_data=None, text=None, content=None):
            mock_resp = Mock()
            mock_resp.status_code = status_code
            mock_resp.ok = 200 <= status_code < 300
            mock_resp.json.return_value = json_data
            mock_resp.text = text
            mock_resp.content = content
            return mock_resp
        
        # Create method-specific factories
        def config_get(url, status_code=200, json=None, text=None, content=None, **kwargs):
            mock_get.return_value = create_response(status_code, json, text, content)
            return mock_get
            
        def config_post(url, status_code=200, json=None, text=None, content=None, **kwargs):
            mock_post.return_value = create_response(status_code, json, text, content)
            return mock_post
            
        def config_put(url, status_code=200, json=None, text=None, content=None, **kwargs):
            mock_put.return_value = create_response(status_code, json, text, content)
            return mock_put
            
        def config_delete(url, status_code=200, json=None, text=None, content=None, **kwargs):
            mock_delete.return_value = create_response(status_code, json, text, content)
            return mock_delete
        
        yield {
            'get': config_get,
            'post': config_post,
            'put': config_put,
            'delete': config_delete
        }


def create_test_profile(profile_dir, profile_id="test_profile", is_default=True):
    """
    Helper function to create a test validation profile.
    
    Args:
        profile_dir: Directory to create the profile in
        profile_id: ID for the profile
        is_default: Whether this is the default profile
        
    Returns:
        Path to the created profile file
    """
    import json
    
    profile_data = {
        "id": profile_id,
        "name": "Test Profile",
        "description": "Profile for testing",
        "is_default": is_default,
        "rules": [
            {
                "column": "Gross_Profit",
                "rule_type": "range",
                "min_value": 0,
                "flag_name": "negative_gross"
            },
            {
                "column": "Lead_Source",
                "rule_type": "not_empty",
                "flag_name": "missing_lead_source"
            },
            {
                "column": "VIN",
                "rule_type": "unique",
                "flag_name": "duplicate_vin"
            }
        ]
    }
    
    # Ensure directory exists
    os.makedirs(profile_dir, exist_ok=True)
    
    # Write profile to file
    profile_path = os.path.join(profile_dir, f"{profile_id}.json")
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    
    return profile_path

"""
Pytest configuration file for Watchdog AI testing.

This file contains shared fixtures and configuration for all test modules.
"""

import os
import pytest
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Get project root directory
PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())

# Add project root to Python path
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths for test assets
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)

# Test Categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "dashboard: marks tests related to dashboard functionality"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers",
        "performance: marks performance tests"
    )
    config.addinivalue_line(
        "markers",
        "error_handling: marks error handling tests"
    )

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    return os.path.join(PROJECT_ROOT, "tests", "data")

@pytest.fixture(scope="session")
def performance_thresholds():
    """Fixture providing performance thresholds for tests."""
    return {
        "metrics_calculation": 3.0,  # seconds
        "heatmap_creation": 3.0,    # seconds
        "trends_creation": 3.0,     # seconds
        "memory_limit": 500,        # MB
        "response_time": 1.0,       # seconds
    }

@pytest.fixture(scope="session")
def mock_feedback_data():
    """Fixture providing mock feedback data for testing."""
    return {
        "insights": {
            "insight-001": {
                "feedback": [
                    {"type": "thumbs", "value": "up", "timestamp": "2024-03-15T10:30:00"},
                    {"type": "rating", "value": 5, "timestamp": "2024-03-15T10:31:00"}
                ]
            },
            "insight-002": {
                "feedback": [
                    {"type": "thumbs", "value": "down", "timestamp": "2024-03-15T11:00:00"},
                    {"type": "comment", "value": "Not helpful", "timestamp": "2024-03-15T11:01:00"}
                ]
            }
        },
        "rejected_queries": [
            {
                "trace_id": "trace-001",
                "query": "Show me sales data",
                "error": "Invalid date range",
                "timestamp": "2024-03-15T10:30:00"
            }
        ]
    }

@pytest.fixture
def assets_dir():
    """Fixture that provides the path to the assets directory."""
    return ASSETS_DIR

@pytest.fixture
def clean_csv_data():
    """Fixture that provides a clean pandas DataFrame for testing."""
    # Sample automotive dealer data with clean formatting
    data = {
        'VIN': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'WBAGH83576D123458'],
        'Make': ['Honda', 'Toyota', 'BMW'],
        'Model': ['Accord', 'Tundra', '7 Series'],
        'Year': [2019, 2020, 2018],
        'Sale_Date': ['2023-01-15', '2023-02-20', '2023-03-05'],
        'Sale_Price': [28500.00, 45750.00, 62000.00],
        'Cost': [25000.00, 40000.00, 55000.00],
        'Gross_Profit': [3500.00, 5750.00, 7000.00],
        'Lead_Source': ['Website', 'Walk-in', 'CarGurus'],
        'Salesperson': ['John Smith', 'Jane Doe', 'Bob Johnson']
    }
    return pd.DataFrame(data)

@pytest.fixture
def messy_csv_data():
    """Fixture that provides a messy pandas DataFrame for testing."""
    # Sample messy automotive dealer data with inconsistent formatting and issues
    data = {
        'VIN #': ['1HGCM82633A123456', '5TFBW5F13AX123457', 'INVALID VIN'],
        'Vehicle Make': ['Honda', 'Toyota', 'BMW'],
        ' Model ': ['Accord', 'Tundra', '7 Series'],
        'Year  ': [2019, 'Twenty-Twenty', 2018],
        'Date of Sale': ['01/15/2023', '02/20/2023', None],
        'Price $': ['$28,500.00', '$45,750.00', '$62,000.00'],
        'Dealer Cost': ['$25,000.00', '$40,000.00', None],
        'Gross': ['$3,500.00', '$5,750.00', '$7,000.00'],
        'Source of Lead': ['Website', 'Walk-in', None],
        'Sales Person': ['John Smith', 'Jane Doe', 'Bob Johnson']
    }
    return pd.DataFrame(data)

@pytest.fixture
def create_sample_csv(assets_dir):
    """Fixture that creates a sample CSV file and returns its path."""
    def _create_csv(filename, data):
        filepath = os.path.join(assets_dir, filename)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath
    return _create_csv

@pytest.fixture
def sample_clean_csv_path(assets_dir, clean_csv_data):
    """Fixture that provides a path to a clean CSV file."""
    filepath = os.path.join(assets_dir, 'clean_file.csv')
    clean_csv_data.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def sample_messy_csv_path(assets_dir, messy_csv_data):
    """Fixture that provides a path to a messy CSV file."""
    filepath = os.path.join(assets_dir, 'messy_file.csv')
    messy_csv_data.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def sample_excel_path(assets_dir, clean_csv_data, messy_csv_data):
    """Fixture that provides a path to a sample Excel file with multiple sheets."""
    filepath = os.path.join(assets_dir, 'sample_excel.xlsx')
    
    with pd.ExcelWriter(filepath) as writer:
        clean_csv_data.to_excel(writer, sheet_name='Clean Data', index=False)
        messy_csv_data.to_excel(writer, sheet_name='Messy Data', index=False)
        
        # Create an empty sheet
        pd.DataFrame().to_excel(writer, sheet_name='Empty Sheet', index=False)
        
        # Create a sheet with header issues
        header_issue_df = clean_csv_data.copy()
        header_issue_df.columns = [f"Column{i}" for i in range(len(header_issue_df.columns))]
        header_issue_df.to_excel(writer, sheet_name='Header Issues', index=False)
    
    return filepath

@pytest.fixture
def empty_dataframe():
    """Fixture that provides an empty DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def sample_metadata():
    """Fixture that provides a sample metadata dictionary."""
    return {
        "source_file_name": "test_file.csv",
        "upload_timestamp": "2023-04-15T10:30:45",
        "rows": 100,
        "columns": 10,
        "column_names": ["VIN", "Make", "Model", "Year", "Sale_Date", 
                        "Sale_Price", "Cost", "Gross_Profit", "Lead_Source", "Salesperson"],
        "file_size_bytes": 10240,
        "missing_values": {"VIN": 0, "Make": 2, "Model": 1},
        "numeric_stats": {
            "Year": {"min": 2018, "max": 2023, "mean": 2020.5},
            "Sale_Price": {"min": 15000, "max": 85000, "mean": 42500}
        }
    }
