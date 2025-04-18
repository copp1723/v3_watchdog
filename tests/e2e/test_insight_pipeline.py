"""
End-to-end tests for the Watchdog AI insight pipeline.
"""

import pytest
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.insights.engine import InsightEngine
from src.utils.data_io import load_data
from src.insights.feedback import feedback_manager

class MockUploadedFile:
    """Mock Streamlit's UploadedFile."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.name = "test.csv"
    
    def read(self):
        return self.df.to_csv().encode()
    
    def seek(self, pos):
        pass
    
    def getvalue(self):
        return self.df.to_csv().encode()

@pytest.fixture
def sample_data():
    """Create sample dealership data."""
    return pd.DataFrame({
        'LeadSource': ['Web'] * 4 + ['Phone'] * 4,
        'TotalGross': [1000, -500, 2000, 1500, 3000, 2500, 1000, 1500],
        'VIN': ['1HGCM82633A123456'] * 8,
        'SaleDate': [datetime.now() - timedelta(days=x) for x in range(8)],
        'SalePrice': [20000] * 8,
        'sales_rep': ['John', 'John', 'Jane', 'Jane', 'Bob', 'Bob', 'Alice', 'Alice']
    })

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate.return_value = """
    The dealership showed strong performance with total gross profit of $12,000.
    Deal volume remained steady at 8 units with average gross per deal of $1,500.
    
    Key areas of strength:
    - Consistent deal flow
    - Higher gross profit per deal
    - Low rate of negative gross deals
    
    Recommendations:
    1. Share best practices with team members
    2. Consider focusing on high-margin inventory segments
    3. Monitor recent pricing strategy for replication
    """
    return client

@pytest.fixture
def engine(mock_llm_client):
    """Create an InsightEngine instance."""
    return InsightEngine(mock_llm_client)

@patch('streamlit.file_uploader')
@patch('streamlit.write')
@patch('streamlit.dataframe')
def test_end_to_end_flow(mock_dataframe, mock_write, mock_uploader, engine, sample_data, mock_llm_client):
    """Test the complete insight generation pipeline."""
    # Setup mock file upload
    uploaded_file = MockUploadedFile(sample_data)
    mock_uploader.return_value = uploaded_file
    
    # Step 1: Load and validate data
    df = load_data(uploaded_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_data)
    
    # Step 2: Generate insights
    result = engine.run(uploaded_file)
    assert "data" in result
    assert "insights" in result
    assert "metadata" in result
    
    # Verify insights were generated
    insights = result["insights"]
    assert len(insights) > 0
    assert all("summary" in insight for insight in insights)
    
    # Step 3: Test feedback collection
    feedback_result = feedback_manager.record_feedback(
        insight_id=insights[0].get("insight_type", "unknown"),
        feedback_type="helpful",
        user_id="test_user",
        session_id=result["metadata"]["session_id"],
        comment="Great insight!"
    )
    assert feedback_result is True
    
    # Verify feedback was recorded
    feedback = feedback_manager.get_feedback(
        session_id=result["metadata"]["session_id"]
    )
    assert len(feedback) > 0
    assert feedback[0]["feedback_type"] == "helpful"

@patch('streamlit.error')
def test_error_handling(mock_error, engine, sample_data):
    """Test error handling in the pipeline."""
    # Create corrupted data
    bad_data = sample_data.copy()
    bad_data['TotalGross'] = 'not_a_number'
    uploaded_file = MockUploadedFile(bad_data)
    
    # Run pipeline and verify error handling
    result = engine.run(uploaded_file)
    assert "error" in result.get("metadata", {})
    mock_error.assert_called()

@patch('sentry_sdk.capture_exception')
def test_sentry_integration(mock_capture_exception, engine, sample_data):
    """Test Sentry error reporting."""
    # Force an error by making the LLM client fail
    engine.llm_client.generate.side_effect = Exception("LLM API error")
    
    uploaded_file = MockUploadedFile(sample_data)
    engine.run(uploaded_file)
    
    # Verify Sentry was called
    mock_capture_exception.assert_called()

def test_performance_metrics(engine, sample_data):
    """Test performance instrumentation."""
    uploaded_file = MockUploadedFile(sample_data)
    
    # Run pipeline and check timing metadata
    result = engine.run(uploaded_file)
    metadata = result["metadata"]
    
    assert "run_timestamp" in metadata
    # Verify we're tracking execution time
    assert isinstance(datetime.fromisoformat(metadata["run_timestamp"]), datetime)