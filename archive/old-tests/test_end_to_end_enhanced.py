"""
End-to-end integration tests for the enhanced Watchdog AI application.
"""

import pytest
import streamlit as st
import pandas as pd
import io
import os
from unittest.mock import patch, MagicMock, call
from datetime import datetime

# Import components to test
from src.app_enhanced import main, render_analyst_insight
from src.insight_conversation_enhanced import ConversationManager
from src.validators.validator_service import process_uploaded_file

@pytest.fixture
def setup_environment():
    """Setup environment variables for testing."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test environment
    os.environ['USE_MOCK'] = 'true'
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['WATCHDOG_PROFILES_DIR'] = 'profiles'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_session_state():
    """Fixture to setup session state for testing."""
    # Store original session state
    original_session_state = st.session_state
    
    # Create a mock session state dictionary
    mock_state = {
        'conversation_history': [],
        'prompt_input': '',
        'validated_data': None,
        'validation_summary': None,
        'validation_report': None,
        'last_uploaded_file': None,
        'set_prompt_input_value_flag': '',
        'clear_prompt_input_flag': False,
        'error_details': None,
        'conversation_manager': None
    }
    
    # Replace session state with mock
    st.session_state = mock_state
    
    yield mock_state
    
    # Restore original session state
    st.session_state = original_session_state

@pytest.fixture
def mock_uploaded_file():
    """Fixture providing a mock UploadedFile with Car Gurus data."""
    # Create a sample CSV with Car Gurus data
    csv_content = """LeadSource,LeadSource Category,DealNumber,SellingPrice,FrontGross,BackGross,Total Gross,SalesRepName,SplitSalesRep,VehicleYear,VehicleMake,VehicleModel,VehicleStockNumber,VehicleVIN
Dealer Website,Online,24278,$44929,$870,$2170,$3039.72,Vaughn Lockett,,2023,Acura,RDX,LA36537,5J8TC2H61PL019014
Car Gurus,Online,24285,$35800,$900,$1200,$2100.50,Lisa Garcia,,2020,Toyota,Highlander,TH67890,5TDZA23C65S987654
Car Gurus,Online,24288,$42500,$1100,$1600,$2700.00,Tom Wilson,,2021,Toyota,4Runner,TR12345,JTEBU5JR7M5123456
Internet Lead,Online,24286,$31500,$500,$700,$1200,Tom Wilson,,,,,INT22987,
Dealer Website,Online,24287,$19750,$-175,$100,$-75.25,Emily Davis,,2023,Chevrolet,Equinox,CE43210,3GNAXUEV2NL134567"""
    
    csv_bytes = csv_content.encode('utf-8')
    
    # Create a mock UploadedFile
    class MockUploadedFile:
        def __init__(self, content, name):
            self.content = content
            self.name = name
            self._io = io.BytesIO(content)
        
        def read(self):
            return self.content
        
        def getvalue(self):
            return self.content
        
        def seek(self, pos):
            self._io.seek(pos)
        
        def __iter__(self):
            return iter(self._io)
    
    return MockUploadedFile(csv_bytes, "test_car_gurus_data.csv")

@pytest.fixture
def mock_validation_context(mock_uploaded_file):
    """Fixture providing a mock validation context after file upload."""
    # Process the uploaded file to get a real context
    try:
        df, summary, report = process_uploaded_file(mock_uploaded_file)
        
        return {
            'validated_data': df,
            'validation_summary': summary,
            'validation_report': report
        }
    except Exception as e:
        # Fallback to a mock context if processing fails
        return {
            'validated_data': pd.read_csv(io.BytesIO(mock_uploaded_file.content)),
            'validation_summary': {
                'status': 'success',
                'rows': 5,
                'columns': 14,
                'flags': {'missing_values': 0}
            },
            'validation_report': {
                'file_name': mock_uploaded_file.name,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        }

def test_full_insight_flow_with_car_gurus_query(setup_environment, mock_session_state, mock_uploaded_file, mock_validation_context):
    """Test the full insight generation flow with a Car Gurus query."""
    # Setup session state with uploaded data
    mock_session_state['validated_data'] = mock_validation_context['validated_data']
    mock_session_state['validation_summary'] = mock_validation_context['validation_summary']
    mock_session_state['validation_report'] = mock_validation_context['validation_report']
    mock_session_state['last_uploaded_file'] = mock_uploaded_file.name
    
    # Initialize ConversationManager
    mock_session_state['conversation_manager'] = ConversationManager(use_mock=True)
    
    # Set the query
    query = "how many totals sales did lead source car gurus produce?"
    mock_session_state['prompt_input'] = query
    
    # Mock successful insight generation
    cm = mock_session_state['conversation_manager']
    
    # Generate the insight
    response = cm.generate_insight(
        query,
        validation_context=mock_validation_context,
        add_to_history=True
    )
    
    # Verify response
    assert response is not None
    assert 'summary' in response
    assert 'value_insights' in response
    assert 'actionable_flags' in response
    assert 'confidence' in response
    assert 'timestamp' in response
    assert 'is_mock' in response
    
    # Verify Car Gurus is mentioned in the response
    response_text = response['summary'].lower()
    assert 'car gurus' in response_text or 'gurus' in response_text
    
    # Verify history was updated
    assert len(mock_session_state['conversation_history']) == 1
    assert mock_session_state['conversation_history'][0]['prompt'] == query
    assert mock_session_state['conversation_history'][0]['response'] == response

def test_error_handling_with_invalid_query(setup_environment, mock_session_state, mock_validation_context):
    """Test error handling when an invalid query is provided."""
    # Setup session state with data
    mock_session_state['validated_data'] = mock_validation_context['validated_data']
    mock_session_state['validation_summary'] = mock_validation_context['validation_summary']
    mock_session_state['validation_report'] = mock_validation_context['validation_report']
    
    # Initialize ConversationManager
    mock_session_state['conversation_manager'] = ConversationManager(use_mock=True)
    
    # Generate insight with empty query (should handle gracefully)
    cm = mock_session_state['conversation_manager']
    response = cm.generate_insight(
        "",
        validation_context=mock_validation_context,
        add_to_history=False
    )
    
    # Verify error handling
    assert response is not None
    assert 'summary' in response
    assert 'error' in response or 'Empty prompt' in response['summary']
    assert response['confidence'] == 'low'

def test_handle_missing_validation_context(setup_environment, mock_session_state):
    """Test handling of missing validation context."""
    # Initialize ConversationManager
    mock_session_state['conversation_manager'] = ConversationManager(use_mock=True)
    
    # Set a query
    query = "how many sales did we make last month?"
    
    # Generate insight without validation context
    cm = mock_session_state['conversation_manager']
    response = cm.generate_insight(
        query,
        validation_context=None,
        add_to_history=False
    )
    
    # Verify response (should still work with empty context)
    assert response is not None
    assert 'summary' in response
    assert 'value_insights' in response
    assert 'is_mock' in response
    assert response['is_mock'] is True

def test_simulated_exception_handling(setup_environment, mock_session_state, mock_validation_context):
    """Test handling of exceptions during insight generation."""
    # Setup session state with data
    mock_session_state['validated_data'] = mock_validation_context['validated_data']
    mock_session_state['validation_summary'] = mock_validation_context['validation_summary']
    mock_session_state['validation_report'] = mock_validation_context['validation_report']
    
    # Initialize ConversationManager
    mock_session_state['conversation_manager'] = ConversationManager(use_mock=True)
    
    # Replace _generate_mock_response with a function that raises an exception
    def raise_exception(*args, **kwargs):
        raise Exception("Simulated error during insight generation")
    
    cm = mock_session_state['conversation_manager']
    original_method = cm._generate_mock_response
    cm._generate_mock_response = raise_exception
    
    try:
        # Generate insight (should handle the exception)
        response = cm.generate_insight(
            "test query",
            validation_context=mock_validation_context,
            add_to_history=False
        )
        
        # Verify error handling
        assert response is not None
        assert 'summary' in response
        assert 'error' in response
        assert 'Simulated error' in response['error']
        assert response['confidence'] == 'low'
    finally:
        # Restore original method
        cm._generate_mock_response = original_method

def test_json_response_parsing(setup_environment, mock_session_state):
    """Test parsing of JSON responses."""
    # Initialize ConversationManager
    mock_session_state['conversation_manager'] = ConversationManager(use_mock=True)
    cm = mock_session_state['conversation_manager']
    
    # Test with valid JSON
    valid_json = '{"summary": "Test summary", "value_insights": ["Insight 1"], "actionable_flags": [], "confidence": "high"}'
    formatted = cm.formatter.format_response(valid_json)
    
    assert formatted['summary'] == "Test summary"
    assert formatted['value_insights'] == ["Insight 1"]
    assert formatted['confidence'] == "high"
    
    # Test with malformed JSON
    invalid_json = '{"summary": "Test summary", "value_insights": ["Incomplete'
    formatted = cm.formatter.format_response(invalid_json)
    
    assert 'error' in formatted
    assert 'summary' in formatted
    assert formatted['confidence'] == 'low'

def test_render_insight_integration(setup_environment):
    """Test rendering insights works properly with different formats."""
    # Test with complete insight
    complete_insight = {
        "summary": "Car Gurus produced 2 sales with a total of $4,800.50 in gross profit.",
        "value_insights": [
            "Car Gurus represents 40% of the deals in the dataset.",
            "Both Car Gurus deals were for Toyota vehicles."
        ],
        "actionable_flags": [
            "Consider increasing marketing spend with Car Gurus."
        ],
        "confidence": "high",
        "timestamp": datetime.now().isoformat(),
        "is_mock": True
    }
    
    # Just verify this doesn't raise an exception
    with patch('streamlit.markdown'):
        with patch('streamlit.caption'):
            with patch('streamlit.columns'):
                render_analyst_insight(complete_insight, 0)
    
    # Test with error insight
    error_insight = {
        "summary": "Error generating insight",
        "error": "Failed to process query",
        "value_insights": [],
        "actionable_flags": [],
        "confidence": "low",
        "timestamp": datetime.now().isoformat(),
        "is_mock": False
    }
    
    with patch('streamlit.error') as mock_error:
        render_analyst_insight(error_insight, 0)
        mock_error.assert_called_once()
