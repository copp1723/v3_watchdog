#!/usr/bin/env python
"""
Tests for guardrail functionality in the Insight Generation System.

These tests verify that the token limits, retry mechanisms, and cost logging
function correctly under various conditions.
"""

import os
import json
import pytest
import time
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import csv
from io import StringIO

import tiktoken

from watchdog_ai.insights.insight_generator import (
    InsightGenerator,
    count_tokens,
    truncate_or_summarize,
    TokenLimitError,
    APIError,
    MAX_INPUT_TOKENS,
    OPENAI_COST_PER_1K_TOKENS,
    retry_with_exponential_backoff
)

# Test fixtures and constants
SAMPLE_ANALYTICS_DATA = {
    "sales": [
        {"rep": "Alice", "amount": 120000, "units": 12},
        {"rep": "Bob", "amount": 85000, "units": 9},
        {"rep": "Charlie", "amount": 150000, "units": 15}
    ],
    "inventory": {
        "total_units": 120,
        "average_days_in_stock": 45,
        "by_make": {
            "Toyota": 40,
            "Honda": 25,
            "Ford": 30,
            "Others": 25
        }
    },
    "finance": {
        "gross_profit": 450000,
        "expenses": 300000,
        "net_profit": 150000
    }
}

SAMPLE_QUERY = "Which sales representative has the highest performance this month?"

SAMPLE_RESPONSE = {
    "summary": "Charlie leads all sales representatives with 15 units sold and $150,000 in total sales, 25% higher than the next best performer.",
    "chart_data": {
        "type": "bar",
        "data": {
            "x": ["Charlie", "Alice", "Bob"],
            "y": [150000, 120000, 85000]
        },
        "title": "Sales by Representative",
        "x_axis_label": "Sales Representative",
        "y_axis_label": "Sales Amount ($)"
    },
    "recommendation": "Consider having Charlie share best practices with the team, especially with Bob who has the lowest performance.",
    "risk_flag": False,
    "confidence_score": 0.95
}

# Mock API response
MOCK_API_RESPONSE = MagicMock()
MOCK_API_RESPONSE.choices = [MagicMock()]
MOCK_API_RESPONSE.choices[0].message = MagicMock()
MOCK_API_RESPONSE.choices[0].message.content = json.dumps(SAMPLE_RESPONSE)
MOCK_API_RESPONSE.usage = MagicMock()
MOCK_API_RESPONSE.usage.completion_tokens = 350
MOCK_API_RESPONSE.usage.total_tokens = 1000
MOCK_API_RESPONSE.id = "mock-request-id"


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("openai.OpenAI") as mock_client:
        instance = mock_client.return_value
        instance.chat.completions.create.return_value = MOCK_API_RESPONSE
        yield instance


@pytest.fixture
def temp_log_file():
    """Create a temporary file for testing cost logging."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Clean up after test
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestTokenLimitEnforcement:
    """Tests for token limit enforcement guardrails."""
    
    def test_count_tokens(self):
        """Test token counting function."""
        text = "This is a test sentence."
        token_count = count_tokens(text)
        
        # Assert realistic token count (may vary based on tokenizer)
        assert token_count > 0
        assert token_count <= len(text)
    
    def test_truncate_or_summarize(self):
        """Test truncation function."""
        # Create a text that exceeds the specified max tokens
        long_text = "word " * 1000  # A text with 1000 "word" tokens
        max_tokens = 100
        
        truncated = truncate_or_summarize(long_text, max_tokens)
        
        # Assert truncation happened
        assert len(truncated) < len(long_text)
        
        # Assert token count is within limit (with some margin for the truncation note)
        assert count_tokens(truncated) <= max_tokens * 1.1
        
        # Assert truncation note is added
        assert "[Note: This data has been truncated" in truncated
    
    def test_token_limit_exception(self, mock_openai_client):
        """Test that TokenLimitError is raised when input is too large."""
        generator = InsightGenerator()
        
        # Create an extremely large analytics summary
        large_analytics = {"data": ["large" * 1000000]}
        
        # Mock count_tokens to simulate exceeding token limit
        with patch("watchdog_ai.insights.insight_generator.count_tokens") as mock_count:
            # First call is checking custom params, return small number
            # Second call is checking the final prompt, return over limit
            mock_count.side_effect = [100, MAX_INPUT_TOKENS + 1000]
            
            # Assert TokenLimitError is raised
            with pytest.raises(TokenLimitError):
                generator.generate_insight(large_analytics, SAMPLE_QUERY)
    
    def test_token_limit_truncation(self, mock_openai_client):
        """Test that analytics data is truncated to fit token limits."""
        generator = InsightGenerator()
        
        # Create a moderately large analytics summary
        moderately_large = {"data": ["moderate" * 10000]}
        
        # Mock truncate_or_summarize to verify it's called
        with patch("watchdog_ai.insights.insight_generator.truncate_or_summarize") as mock_truncate:
            mock_truncate.return_value = json.dumps({"data": ["truncated"]})
            
            # Generate insight
            generator.generate_insight(moderately_large, SAMPLE_QUERY)
            
            # Assert truncate_or_summarize was called
            mock_truncate.assert_called_once()


class TestRetryMechanism:
    """Tests for retry mechanism guardrails."""
    
    def test_retry_decorator(self):
        """Test the retry decorator with a failing function."""
        
        # Create a function that fails twice then succeeds
        mock_function = MagicMock()
        mock_function.side_effect = [APIError("Test error"), APIError("Test error"), "Success"]
        
        # Apply the retry decorator
        decorated_function = retry_with_exponential_backoff(max_retries=3, initial_backoff=0.01)(mock_function)
        
        # Execute the function
        with patch("time.sleep") as mock_sleep:  # Avoid actual sleep in tests
            result = decorated_function()
        
        # Assert function was called 3 times
        assert mock_function.call_count == 3
        
        # Assert sleep was called twice (after each failure)
        assert mock_sleep.call_count == 2
        
        # Assert final result is correct
        assert result == "Success"
    
    def test_retry_max_attempts(self):
        """Test that retry gives up after max attempts."""
        
        # Create a function that always fails
        mock_function = MagicMock()
        mock_function.side_effect = APIError("Always fails")
        
        # Apply the retry decorator
        decorated_function = retry_with_exponential_backoff(max_retries=3, initial_backoff=0.01)(mock_function)
        
        # Execute the function
        with patch("time.sleep") as mock_sleep:  # Avoid actual sleep in tests
            with pytest.raises(APIError):
                decorated_function()
        
        # Assert function was called max_retries times
        assert mock_function.call_count == 3
        
        # Assert sleep was called max_retries-1 times
        assert mock_sleep.call_count == 2
    
    def test_openai_api_retry(self, mock_openai_client):
        """Test retry mechanism in _call_openai method."""
        generator = InsightGenerator()
        
        # Make API call fail twice then succeed
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            MOCK_API_RESPONSE
        ]
        
        # Mock time.sleep to avoid actual delays
        with patch("time.sleep"):
            # Call the method
            response, usage = generator._call_openai("Test prompt")
        
        # Assert API was called 3 times
        assert mock_openai_client.chat.completions.create.call_count == 3
        
        # Assert final response is correct
        assert json.loads(response) == SAMPLE_RESPONSE


class TestCostLogging:
    """Tests for cost logging guardrails."""
    
    def test_console_cost_logging(self, mock_openai_client, caplog):
        """Test that costs are logged to console."""
        generator = InsightGenerator()
        
        # Generate an insight
        generator.generate_insight(SAMPLE_ANALYTICS_DATA, SAMPLE_QUERY)
        
        # Assert cost logging appears in logs
        assert any("OpenAI API call cost" in record.message for record in caplog.records)
        assert any("Input:" in record.message for record in caplog.records)
        assert any("Output:" in record.message for record in caplog.records)
        assert any("Total:" in record.message for record in caplog.records)
    
    def test_file_cost_logging(self, mock_openai_client, temp_log_file):
        """Test that costs are logged to file when enabled."""
        # Create generator with file logging enabled
        generator = InsightGenerator(log_to_file=True, log_file_path=temp_log_file)
        
        # Generate an insight
        generator.generate_insight(SAMPLE_ANALYTICS_DATA, SAMPLE_QUERY)
        
        # Assert log file exists and contains expected content
        assert os.path.exists(temp_log_file)
        
        with open(temp_log_file, 'r') as f:
            content = f.read()
            assert "input_tokens" in content
            assert "output_tokens" in content
            assert "total_cost" in content
    
    def test_cost_calculation(self):
        """Test that cost calculation is accurate."""
        generator = InsightGenerator()
        
        # Calculate expected costs
        input_tokens = 650
        output_tokens = 350
        model = "gpt-4o"
        
        model_costs = OPENAI_COST_PER_1K_TOKENS[model]
        expected_input_cost = (input_tokens / 1000) * model_costs["input"]
        expected_output_cost = (output_tokens / 1000) * model_costs["output"]
        expected_total_cost = expected_input_cost + expected_output_cost
        
        # Create a StringIO to capture CSV output
        csv_output = StringIO()
        csv_writer = MagicMock()
        
        # Mock open and csv.DictWriter
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("csv.DictWriter") as mock_csv:
            mock_csv.return_value = csv_writer
            
            # Call log_usage_cost
            generator._log_usage_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                request_id="test-request-id"
            )
            
            # If log_to_file is False, file should not be opened
            mock_file.assert_not_called()
        
        # Now test with file logging enabled
        generator.log_to_file = True
        
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("csv.DictWriter") as mock_csv:
            mock_csv.return_value = csv_writer
            
            # Call log_usage_cost
            generator._log_usage_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                request_id="test-request-id"
            )
            
            # File should be opened
            mock_file.assert_called_once()
            
            # CSV writer should be initialized with correct field names
            mock_csv_call = mock_csv.call_args
            assert "timestamp" in mock_csv_call[1]["fieldnames"]
            assert "input_tokens" in mock_csv_call[1]["fieldnames"]
            assert "output_tokens" in mock_csv_call[1]["fieldnames"]
            assert "total_cost" in mock_csv_call[1]["fieldnames"]
            
            # Row should be written with correct values
            csv_writer.writerow.assert_called_once()
            row_data = csv_writer.writerow.call_args[0][0]
            assert row_data["input_tokens"] == input_tokens
            assert row_data["output_tokens"] == output_tokens
            assert f"${expected_input_cost:.4f}" in row_data["input_cost"]
            assert f"${expected_output_cost:.4f}" in row_data["output_cost"]
            assert f"${expected_total_cost:.4f}" in row_data["total_cost"]


if __name__ == "__main__":
    pytest.main()

