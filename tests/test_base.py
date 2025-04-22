"""Base test utilities and helper functions.

This module provides a comprehensive set of utilities for testing, including:
- DataFrame comparison and manipulation
- Test data generation
- Common assertion helpers
- Mocking utilities for external services
- Performance testing helpers
- File handling utilities
"""

import os
import sys
import json
import time
import shutil
import tempfile
import logging
import unittest
import random
import string
import warnings
import contextlib
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterator, Set, TypeVar, cast
from typing import Generator, ContextManager, Protocol, Type
from datetime import datetime, timedelta, date
from unittest import mock
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from faker import Faker

# Initialize faker for data generation
faker = Faker()

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')

class TestBase:
    """Base class for test cases with comprehensive testing utilities.
    
    This class provides various methods for:
    - DataFrame comparison and manipulation
    - Test data generation
    - Common assertion helpers
    - Mocking utilities for external services
    - Performance testing helpers
    - File handling utilities
    """
    
    # Class constants
    TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
    TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    
    @classmethod
    def setUp(cls) -> None:
        """Set up test environment before test execution."""
        # Create temp directory if it doesn't exist
        if not os.path.exists(cls.TEMP_DIR):
                else:
                    # For non-numeric columns, compare values directly
                    assert (col1 == col2).all(), f"Non-numeric values differ in column {col}"
        else:
            # Regular equality check without ignoring order
            if ignore_index:
                df1_reset = df1.reset_index(drop=True)
                df2_reset = df2.reset_index(drop=True)
                
                for col in df1_reset.columns:
                    col1 = df1_reset[col]
                    col2 = df2_reset[col]
                    
                    # Check data types if required
                    if check_dtype:
                        assert col1.dtype == col2.dtype, f"Data types differ for column {col}: {col1.dtype} vs {col2.dtype}"
                    
                    # Compare values with tolerance for numeric data
                    if pd.api.types.is_numeric_dtype(col1):
                        # Handle NaN values
                        mask = ~(pd.isna(col1) & pd.isna(col2))
                        numeric_mask = mask & ~pd.isna(col1) & ~pd.isna(col2)
                        
                        if numeric_mask.any():
                            numeric_diff = abs(col1[numeric_mask] - col2[numeric_mask])
                            assert (numeric_diff <= tol).all(), f"Numeric values differ in column {col}"
                        
                        # Check that NaN patterns match
                        assert (pd.isna(col1) == pd.isna(col2)).all(), f"NaN patterns differ in column {col}"
                    else:
                        # For non-numeric columns, compare values directly
                        assert (col1 == col2).all(), f"Non-numeric values differ in column {col}"
            else:
                # Use pandas built-in testing function with appropriate options
                pd.testing.assert_frame_equal(
                    df1, df2,
                    check_dtype=check_dtype,
                    check_index_type=check_index_type,
                    check_column_type=check_dtype,
                    check_frame_type=False,
                    check_names=True,
                    check_exact=False,
                    atol=tol,
                    rtol=tol,
                )
    
    # ----- Advanced Assertion Helpers -----
    
    @staticmethod
    def assertDictEqual(
        dict1: Dict[Any, Any], 
        dict2: Dict[Any, Any], 
        msg: Optional[str] = None
    ) -> None:
        """Assert that two dictionaries are equal.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            msg: Optional message to display on failure
            
        Raises:
            AssertionError: If dictionaries are not equal
        """
        assert dict1 == dict2, msg or f"Dictionaries are not equal: {dict1} != {dict2}"
    
    @staticmethod
    def assertDictContainsSubset(
        subset: Dict[Any, Any], 
        dictionary: Dict[Any, Any], 
        msg: Optional[str] = None
    ) -> None:
        """Assert that a dictionary contains all key-value pairs from a subset.
        
        Args:
            subset: Dictionary with key-value pairs expected to be in dictionary
            dictionary: Dictionary to check
            msg: Optional message to display on failure
            
        Raises:
            AssertionError: If dictionary does not contain all key-value pairs from subset
        """
        for key, value in subset.items():
            assert key in dictionary, msg or f"Key '{key}' not found in dictionary"
            assert dictionary[key] == value, msg or f"Value for key '{key}' is {dictionary[key]}, expected {value}"
    
    @staticmethod
    def assertAlmostEqualDict(
        dict1: Dict[str, Union[float, int]], 
        dict2: Dict[str, Union[float, int]], 
        places: int = 7,
        msg: Optional[str] = None
    ) -> None:
        """Assert that two dictionaries with numeric values are almost equal.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            places: Number of decimal places for comparison
            msg: Optional message to display on failure
            
        Raises:
            AssertionError: If dictionaries are not almost equal
        """
        assert set(dict1.keys()) == set(dict2.keys()), \
            msg or f"Dictionaries have different keys: {set(dict1.keys())} != {set(dict2.keys())}"
        
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                assert round(abs(val1 - val2), places) == 0, \
                    msg or f"Values for key '{key}' differ: {val1} != {val2}"
            else:
                assert val1 == val2, \
                    msg or f"Values for key '{key}' differ: {val1} != {val2}"
    
    @staticmethod
    def assertListEqual(
        list1: List[Any], 
        list2: List[Any], 
        ignore_order: bool = False,
        msg: Optional[str] = None
    ) -> None:
        """Assert that two lists are equal.
        
        Args:
            list1: First list
            list2: Second list
            ignore_order: Whether to ignore order of elements
            msg: Optional message to display on failure
            
        Raises:
            AssertionError: If lists are not equal based on specified criteria
        """
        if ignore_order:
            assert sorted(list1) == sorted(list2), \
                msg or f"Lists are not equal (ignoring order): {list1} != {list2}"
        else:
            assert list1 == list2, \
                msg or f"Lists are not equal: {list1} != {list2}"
    
    # ----- Mocking Utilities -----
    
    @staticmethod
    @contextmanager
    def mock_service(
        url_pattern: str, 
        response_data: Any, 
        status_code: int = 200,
        content_type: str = 'application/json',
        method: str = 'GET'
    ) -> Generator[mock.MagicMock, None, None]:
        """Mock an external service response.
        
        Args:
            url_pattern: URL pattern to match (supports regex with re.match)
            response_data: Data to return in response
            status_code: HTTP status code
            content_type: Content type header
            method: HTTP method to mock
            
        Yields:
            Mock object that can be used to validate calls
            
        Example:
            with TestBase.mock_service('https://api.example.com/users', {'users': []}, method='GET'):
                result = your_code_that_calls_api()
                assert result == expected_result
        """
        import requests
        import re
        import json
        
        # Define the side effect function to check the URL pattern
        def side_effect(url, *args, **kwargs):
            if re.match(url_pattern, url):
                mock_response = mock.MagicMock()
                mock_response.status_code = status_code
                mock_response.ok = 200 <= status_code < 300
                mock_response.headers = {'Content-Type': content_type}
                
                # Handle different response types
                if isinstance(response_data, dict) or isinstance(response_data, list):
                    mock_response.json.return_value = response_data
                    mock_response.text = json.dumps(response_data)
                    mock_response.content = json.dumps(response_data).encode('utf-8')
                elif isinstance(response_data, str):
                    mock_response.text = response_data
                    mock_response.content = response_data.encode('utf-8')
                    
                    if content_type == 'application/json':
                        try:
                            mock_response.json.return_value = json.loads(response_data)
                        except:
                            mock_response.json.side_effect = ValueError("Invalid JSON")
                else:
                    mock_response.text = str(response_data)
                    mock_response.content = str(response_data).encode('utf-8')
                
                return mock_response
            else:
                # Pass through to real requests for URLs that don't match
                return requests_method_real(url, *args, **kwargs)
        
        # Patch the appropriate HTTP method
        method = method.upper()
        if method == 'GET':
            target = 'requests.get'
            requests_method_real = requests.get
        elif method == 'POST':
            target = 'requests.post'
            requests_method_real = requests.post
        elif method == 'PUT':
            target = 'requests.put'
            requests_method_real = requests.put
        elif method == 'DELETE':
            target = 'requests.delete'
            requests_method_real = requests.delete
        elif method == 'PATCH':
            target = 'requests.patch'
            requests_method_real = requests.patch
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # Apply the patch
        with mock.patch(target) as mock_method:
            mock_method.side_effect = side_effect
            yield mock_method
    
    @staticmethod
    @contextmanager
    def mock_environment_variables(
        env_vars: Dict[str, str]
    ) -> Generator[None, None, None]:
        """Temporarily mock environment variables.
        
        Args:
            env_vars: Dictionary of environment variables to set temporarily
            
        Yields:
            None
            
        Example:
            with TestBase.mock_environment_variables({'API_KEY': 'test-key', 'ENV': 'test'}):
                result = your_code_that_uses_env_vars()
                assert result == expected_result
        """
        original_environ = dict(os.environ)
        os.environ.update(env_vars)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(original_environ)
    
    @staticmethod
    @contextmanager
    def mock_stdout() -> Generator[StringIO, None, None]:
        """Capture stdout for testing.
        
        Yields:
            StringIO object containing captured stdout
            
        Example:
            with TestBase.mock_stdout() as stdout:
                print("Hello, world!")
                assert stdout.getvalue() == "Hello, world!\n"
        """
        original_stdout = sys.stdout
        stdout_buffer = StringIO()
        sys.stdout = stdout_buffer
        try:
            yield stdout_buffer
        finally:
            sys.stdout = original_stdout
    
    @staticmethod
    @contextmanager
    def mock_stderr() -> Generator[StringIO, None, None]:
        """Capture stderr for testing.
        
        Yields:
            StringIO object containing captured stderr
            
        Example:
            with TestBase.mock_stderr() as stderr:
                print("Error message", file=sys.stderr)
                assert "Error message" in stderr.getvalue()
        """
        original_stderr = sys.stderr
        stderr_buffer = StringIO()
        sys.stderr = stderr_buffer
        try:
            yield stderr_buffer
        finally:
            sys.stderr = original_stderr
    
    # ----- Performance Testing Helpers -----
    
    @staticmethod
    @contextmanager
    def time_block(
        label: str = "Operation", 
        threshold: Optional[float] = None, 
        logger: Optional[logging.Logger] = None
    ) -> Generator[None, None, None]:
        """Time a block of code and optionally verify it completes within threshold.
        
        Args:
            label: Label for the timed operation
            threshold: Maximum allowed execution time in seconds
            logger: Logger to use for output, if None it will print to stdout
            
        Yields:
            None
            
        Raises:
            AssertionError: If threshold is provided and execution time exceeds it
            
        Example:
            with TestBase.time_block("Data processing", threshold=1.0):
                # Code that should complete within 1 second
                process_data(large_dataset)
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            message = f"{label} took {elapsed_time:.4f} seconds"
            
            if logger:
                logger.info(message)
            else:
                print(message)
            
            if threshold is not None:
                assert elapsed_time <= threshold, \
                    f"{label} took {elapsed_time:.4f} seconds, which exceeds threshold of {threshold:.4f} seconds"
    
    @staticmethod
    def measure_execution_time(
        func: Callable[..., T], 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[T, float]:
        """Measure execution time of a function.
        
        Args:
            func: Function to time
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Tuple of (function result, execution time in seconds)
            
        Example:
            result, exec_time = TestBase.measure_execution_time(calculate_metrics, large_dataset)
            assert exec_time < 5.0  # Should complete in under 5 seconds
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    
    @staticmethod
    def profile_memory_usage(
        func: Callable[..., T], 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[T, float]:
        """Profile memory usage of a function.
        
        Args:
            func: Function to profile
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Tuple of (function result, peak memory usage in MB)
            
        Example:
            result, memory_used = TestBase.profile_memory_usage(process_large_file, 'huge_file.csv')
            assert memory_used < 100  # Should use less than 100MB of memory
        """
        try:
            import psutil
        except ImportError:
            warnings.warn("psutil not installed, memory profiling disabled")
            result = func(*args, **kwargs)
            return result, 0.0
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        result = func(*args, **kwargs)
        
    def tearDown(cls) -> None:
        """Clean up test environment after test execution."""
        # Clean up temp directory
        if os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR)
            os.makedirs(cls.TEMP_DIR)
    
    # ----- DataFrame Creation and Comparison Methods -----
    
    @staticmethod
    def create_sample_df(rows: int = 100) -> pd.DataFrame:
        """Create a sample DataFrame for testing.
        
        Args:
            rows: Number of rows to create
            
        Returns:
            A DataFrame with date and random value columns
        """
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=rows),
            'value': np.random.randn(rows)
        })
    
    @staticmethod
    def generate_dataframe(
        rows: int = 100, 
        columns: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
        include_nulls: bool = False,
        null_probability: float = 0.1
    ) -> pd.DataFrame:
        """Generate a DataFrame with synthetic data for testing.
        
        Args:
            rows: Number of rows to generate
            columns: List of column names (defaults to standard sales columns if None)
            dtypes: Dictionary mapping column names to data types ('int', 'float', 'str', 'date', 'bool')
            include_nulls: Whether to include NULL values
            null_probability: Probability of a cell being NULL (if include_nulls is True)
            
        Returns:
            A DataFrame with generated test data
        """
        if columns is None:
            columns = ['id', 'name', 'value', 'date', 'category', 'is_active']
            
        if dtypes is None:
            dtypes = {
                'id': 'int',
                'name': 'str',
                'value': 'float',
                'date': 'date',
                'category': 'str',
                'is_active': 'bool'
            }
        
        data: Dict[str, List[Any]] = {}
        
        for col in columns:
            col_type = dtypes.get(col, 'str')
            column_data: List[Any] = []
            
            for i in range(rows):
                # Randomly generate NULL values
                if include_nulls and random.random() < null_probability:
                    column_data.append(None)
                    continue
                
                # Generate appropriate data based on column type
                if col_type == 'int':
                    column_data.append(random.randint(1, 10000))
                elif col_type == 'float':
                    column_data.append(random.uniform(0, 1000))
                elif col_type == 'date':
                    column_data.append(faker.date_between(start_date='-2y', end_date='today'))
                elif col_type == 'bool':
                    column_data.append(random.choice([True, False]))
                else:  # Default to string
                    if col == 'name':
                        column_data.append(faker.name())
                    elif col == 'category':
                        column_data.append(faker.random_element(elements=('A', 'B', 'C', 'D', 'E')))
                    elif 'email' in col.lower():
                        column_data.append(faker.email())
                    elif 'phone' in col.lower():
                        column_data.append(faker.phone_number())
                    elif 'address' in col.lower():
                        column_data.append(faker.address())
                    else:
                        column_data.append(faker.word())
            
            data[col] = column_data
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_automotive_df(rows: int = 100, include_nulls: bool = False) -> pd.DataFrame:
        """Generate a DataFrame with automotive sales data.
        
        Args:
            rows: Number of rows to generate
            include_nulls: Whether to include NULL values
            
        Returns:
            DataFrame with automotive sales data
        """
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Audi', 'Nissan']
        models = {
            'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Tacoma'],
            'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey'],
            'Ford': ['F-150', 'Escape', 'Explorer', 'Mustang', 'Edge'],
            'Chevrolet': ['Silverado', 'Equinox', 'Malibu', 'Traverse', 'Tahoe'],
            'BMW': ['3 Series', '5 Series', 'X3', 'X5', '7 Series'],
            'Mercedes': ['C-Class', 'E-Class', 'S-Class', 'GLC', 'GLE'],
            'Audi': ['A4', 'A6', 'Q5', 'Q7', 'A8'],
            'Nissan': ['Altima', 'Rogue', 'Sentra', 'Pathfinder', 'Murano']
        }
        lead_sources = ['Website', 'Walk-in', 'Referral', 'Phone', 'Email', 'Social Media']
        sales_people = ['John Smith', 'Jane Doe', 'Robert Johnson', 'Emily Wilson', 'Michael Brown']
        
        data = {
            'Sale_Date': [],
            'VIN': [],
            'Make': [],
            'Model': [],
            'Year': [],
            'Sale_Price': [],
            'Cost': [],
            'Gross_Profit': [],
            'Lead_Source': [],
            'Salesperson': []
        }
        
        for _ in range(rows):
            # Generate sale date within last 2 years
            sale_date = faker.date_between(start_date='-2y', end_date='today')
            
            # Generate Make and Model
            make = random.choice(makes)
            model = random.choice(models[make])
            
            # Generate other fields
            year = random.randint(2018, 2025)
            cost = random.uniform(15000, 50000)
            sale_price = cost * random.uniform(1.05, 1.2)  # 5-20% markup
            gross_profit = sale_price - cost
            
            # Generate VIN (simplified version)
            vin = ''.join(random.choices(string.ascii_uppercase + string.digits, k=17))
            
            # Randomly include nulls if specified
            if include_nulls and random.random() < 0.1:
                if random.random() < 0.3:
                    lead_source = None
                else:
                    lead_source = random.choice(lead_sources)
                
                if random.random() < 0.2:
                    salesperson = None
                else:
                    salesperson = random.choice(sales_people)
            else:
                lead_source = random.choice(lead_sources)
                salesperson = random.choice(sales_people)
            
            # Add to data dictionary
            data['Sale_Date'].append(sale_date)
            data['VIN'].append(vin)
            data['Make'].append(make)
            data['Model'].append(model)
            data['Year'].append(year)
            data['Sale_Price'].append(sale_price)
            data['Cost'].append(cost)
            data['Gross_Profit'].append(gross_profit)
            data['Lead_Source'].append(lead_source)
            data['Salesperson'].append(salesperson)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """Compare two DataFrames for equality.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            True if DataFrames are equal, False otherwise
        """
        if df1.shape != df2.shape:
            return False
        return df1.equals(df2)
    
    @staticmethod
    def assert_frame_equal_extended(
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        ignore_order: bool = False,
        ignore_index: bool = False,
        check_dtype: bool = True,
        check_column_order: bool = True,
        check_index_type: bool = False,
        tol: float = 1e-6,
        msg: Optional[str] = None
    ) -> None:
        """Assert that two DataFrames are equal with extended options.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            ignore_order: Whether to ignore row order
            ignore_index: Whether to ignore index values
            check_dtype: Whether to check data types
            check_column_order: Whether to check column order
            check_index_type: Whether to check index type
            tol: Tolerance for floating point comparisons
            msg: Optional message to display on failure
            
        Raises:
            AssertionError: If DataFrames are not equal based on specified criteria
        """
        # Check shape first
        assert df1.shape[1] == df2.shape[1], f"DataFrames have different number of columns: {df1.shape[1]} vs {df2.shape[1]}"
        
        if not ignore_order:
            assert df1.shape[0] == df2.shape[0], f"DataFrames have different number of rows: {df1.shape[0]} vs {df2.shape[0]}"
        
        # Check columns, regardless of order if specified
        if not check_column_order:
            assert set(df1.columns) == set(df2.columns), f"DataFrames have different column names: {set(df1.columns)} vs {set(df2.columns)}"
        else:
            assert list(df1.columns) == list(df2.columns), f"DataFrames have different column names or order: {list(df1.columns)} vs {list(df2.columns)}"
        
        # Handle both ignore_order and normal comparison
        if ignore_order:
            # Sort both dataframes
            df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
            df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
            
            # Now compare cell by cell with tolerance for float values
            for col in df1_sorted.columns:
                col1 = df1_sorted[col]
                col2 = df2_sorted[col]
                
                # Check data types if required
                if check_dtype:
                    assert col1.dtype == col2.dtype, f"Data types differ for column {col}: {col1.dtype} vs {col2.dtype}"
                
                # Compare values with tolerance for numeric data
                if pd.api.types.is_numeric_dtype(col1):
                    # Handle NaN values
                    mask = ~(pd.isna(col1) & pd.isna(col2))
                    numeric_mask = mask & ~pd.isna(col1) & ~pd.isna(col2)
                    
                    if numeric_mask.any():
                        numeric_diff = abs(col1[numeric_mask] - col2[numeric_mask])
                        assert (numeric_diff <= tol).all(), f"Numeric values differ in column {col}"
                    
                    # Check that NaN patterns match
                    assert (pd.isna(col1) == pd.isna(col2)).all(), f"NaN patterns differ in column {col}"
                else:
                    # For

