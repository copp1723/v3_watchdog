"""
Unit tests for the insight engine pipeline.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta
from src.insights.engine import InsightEngine
from src.utils.errors import InsightGenerationError

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate.return_value = """
    The dealership showed strong performance in Q1 2025 with total gross profit of $125,000.
    Deal volume remained steady at 25 units with average gross per deal increasing from $4,500 to $5,000.
    
    Key areas of strength:
    - Consistent deal flow above team average
    - Higher gross profit per deal
    - Low rate of negative gross deals (2%)
    
    Recommendations:
    1. Share best practices with team members
    2. Consider focusing on high-margin inventory segments
    3. Monitor recent pricing strategy for replication
    """
    return client

@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing."""
    return pd.DataFrame({
        'gross': [1000, -500, 2000, 1500, 3000, 2500, 1000, 1500],
        'sales_rep': ['John', 'John', 'Jane', 'Jane', 'Bob', 'Bob', 'Alice', 'Alice'],
        'date': [datetime.now() - timedelta(days=x) for x in range(8)],
        'LeadSource': ['Web'] * 4 + ['Phone'] * 4,
        'TotalGross': [1000, -500, 2000, 1500, 3000, 2500, 1000, 1500],
        'VIN': ['1HGCM82633A123456'] * 8,
        'SaleDate': [datetime.now() - timedelta(days=x) for x in range(8)],
        'SalePrice': [20000] * 8
    })

@pytest.fixture
def mock_uploaded_file(sample_sales_data):
    """Create a mock uploaded file."""
    class MockFile:
        def __init__(self, df):
            self.df = df
            self.name = "test.csv"
        
        def read(self):
            return self.df.to_csv().encode()
        
        def seek(self, pos):
            pass
    
    return MockFile(sample_sales_data)

def test_engine_initialization(mock_llm_client):
    """Test insight engine initialization."""
    engine = InsightEngine(mock_llm_client)
    
    assert engine.llm_client == mock_llm_client
    assert isinstance(engine.insights, dict)
    assert len(engine.insights) > 0
    assert engine.rules_version is not None

@patch('src.insights.engine.load_data')
@patch('src.insights.engine.validate_data')
def test_run_pipeline_success(mock_validate_data, mock_load_data, mock_llm_client, sample_sales_data, mock_uploaded_file):
    """Test successful pipeline execution."""
    # Setup mocks
    mock_load_data.return_value = sample_sales_data
    mock_validate_data.return_value = (sample_sales_data, {
        "quality_score": 95,
        "missing_values": {},
        "invalid_values": {}
    })
    
    # Run pipeline
    engine = InsightEngine(mock_llm_client)
    result = engine.run(mock_uploaded_file)
    
    # Verify structure
    assert "data" in result
    assert "insights" in result
    assert "metadata" in result
    
    # Check metadata
    metadata = result["metadata"]
    assert "session_id" in metadata
    assert "run_timestamp" in metadata
    assert "rules_version" in metadata
    assert "validation_summary" in metadata
    
    # Check insights
    assert isinstance(result["insights"], list)
    assert len(result["insights"]) > 0
    
    # Verify first insight
    insight = result["insights"][0]
    assert "insight_type" in insight
    assert "summary" in insight
    assert insight["summary"] is not None

@patch('src.insights.engine.load_data')
def test_run_pipeline_load_error(mock_load_data, mock_llm_client, mock_uploaded_file):
    """Test pipeline handling of load error."""
    mock_load_data.side_effect = ValueError("Invalid file format")
    
    engine = InsightEngine(mock_llm_client)
    
    with pytest.raises(InsightGenerationError):
        engine.run(mock_uploaded_file)

def test_format_metrics_table(mock_llm_client):
    """Test metrics table formatting."""
    engine = InsightEngine(mock_llm_client)
    
    insight = {
        "overall_stats": {
            "total_gross": 125000.50,
            "deal_count": 25,
            "avg_gross": 5000.02
        },
        "benchmarks": {
            "top_quartile_gross": 7500.00,
            "deals_per_rep_mean": 6.25
        }
    }
    
    table = engine._format_metrics_table(insight)
    
    # Verify table structure
    assert "| Metric | Value |" in table
    assert "|--------|--------|" in table
    assert "| Total Gross | $125,000.50 |" in table
    assert "| Deal Count | 25 |" in table

def test_get_date_range(mock_llm_client):
    """Test date range extraction."""
    engine = InsightEngine(mock_llm_client)
    
    # Create test DataFrame with dates
    dates = [datetime(2025, 1, 1) + timedelta(days=x) for x in range(30)]
    df = pd.DataFrame({
        'date': dates,
        'value': range(30)
    })
    
    date_range = engine._get_date_range(df)
    assert "Jan 2025" in date_range
    assert "Jan 2025 - Feb 2025" == date_range

@patch('sentry_sdk.capture_exception')
def test_error_reporting(mock_capture_exception, mock_llm_client, mock_uploaded_file):
    """Test error reporting to Sentry."""
    engine = InsightEngine(mock_llm_client)
    
    # Make LLM client raise an error
    engine.llm_client.generate.side_effect = Exception("LLM API error")
    
    with pytest.raises(InsightGenerationError):
        engine.run(mock_uploaded_file)
    
    # Verify Sentry was called
    mock_capture_exception.assert_called()