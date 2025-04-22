"""
Unit tests for the insight card rendering module.
"""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
from src.insight_card import (
    extract_metadata,
    format_markdown_with_highlights,
    InsightMetadata,
    InsightOutputFormatter,
    render_insight_card,
    format_insight_for_display
)

def test_extract_metadata_comparison():
    """Test metadata extraction for comparison text."""
    summary = "Let's compare the sales performance between Q1 and Q2."
    metadata = extract_metadata(summary)
    assert metadata.has_comparison is True
    assert metadata.has_trend is False

def test_extract_metadata_trend():
    """Test metadata extraction for trend text."""
    summary = "Sales showed an increasing trend over the last 3 months."
    metadata = extract_metadata(summary)
    assert metadata.has_comparison is False
    assert metadata.has_trend is True

def test_extract_metadata_metrics():
    """Test metadata extraction for text containing metrics."""
    summary = "Revenue increased by $1,234,567 (15.5%) from last year."
    metadata = extract_metadata(summary)
    assert metadata.has_metrics is True
    assert "$1,234,567" in metadata.highlight_phrases
    assert "15.5%" in metadata.highlight_phrases

def test_format_markdown_with_highlights():
    """Test markdown formatting with highlights."""
    text = "Revenue: $1,000,000 and growth: 10%"
    phrases = ["$1,000,000", "10%"]
    formatted = format_markdown_with_highlights(text, phrases)
    assert "**$1,000,000**" in formatted
    assert "**10%**" in formatted

def test_format_markdown_with_special_chars():
    """Test markdown formatting with special regex characters."""
    text = "Price: $1.00 + tax"
    phrases = ["$1.00"]
    formatted = format_markdown_with_highlights(text, phrases)
    assert "**$1.00**" in formatted

@pytest.fixture
def sample_response():
    """Fixture providing a sample response dictionary."""
    return {
        'summary': "Sales increased by $1,000,000 (10%) compared to last year.",
        'table': pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar'],
            'Sales': [100, 200, 300]
        }),
        'chart_data': {
            'type': 'line',
            'data': pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar'],
                'Sales': [100, 200, 300]
            })
        }
    }

def test_empty_response():
    """Test handling of empty response."""
    render_insight_card({})  # Should not raise an exception

def test_missing_summary():
    """Test handling of response without summary."""
    render_insight_card({'table': pd.DataFrame()})  # Should not raise an exception

def test_malformed_table():
    """Test handling of malformed table data."""
    render_insight_card({
        'summary': "Test summary",
        'table': "not a dataframe"  # Should be ignored
    })  # Should not raise an exception

def test_malformed_chart_data():
    """Test handling of malformed chart data."""
    render_insight_card({
        'summary': "Test summary",
        'chart_data': {'type': 'invalid'}  # Should be ignored
    })  # Should not raise an exception 

@pytest.fixture
def sample_insight():
    """Fixture providing a sample insight dictionary."""
    return {
        'summary': "Sales increased by $1,000,000 (10%) compared to last year.",
        'chart_data': {
            'type': 'line',
            'data': pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar'],
                'Sales': [100, 200, 300]
            }),
            'title': 'Sales Trend'
        },
        'recommendation': "Consider increasing marketing budget to sustain growth.",
        'risk_flag': False
    }

@pytest.fixture
def mock_st():
    """Fixture providing mocked Streamlit functions."""
    with patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.bar_chart') as mock_bar_chart, \
         patch('streamlit.line_chart') as mock_line_chart, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.container') as mock_container, \
         patch('streamlit.columns') as mock_columns:
        
        # Setup mock container context
        mock_container.return_value.__enter__.return_value = MagicMock()
        mock_container.return_value.__exit__.return_value = None
        
        # Setup mock columns
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        yield {
            'markdown': mock_markdown,
            'bar_chart': mock_bar_chart,
            'line_chart': mock_line_chart,
            'button': mock_button,
            'container': mock_container,
            'columns': mock_columns
        }

class TestInsightOutputFormatter:
    """Test suite for InsightOutputFormatter class."""
    
    def test_format_output_valid_data(self):
        """Test formatting with valid data."""
        formatter = InsightOutputFormatter()
        data = {
            'summary': 'Test summary',
            'chart_data': {'type': 'line', 'data': {}},
            'recommendation': 'Test recommendation',
            'risk_flag': True
        }
        result = formatter.format_output(data)
        assert result == data
    
    def test_format_output_missing_fields(self):
        """Test formatting with missing fields."""
        formatter = InsightOutputFormatter()
        data = {'summary': 'Test summary'}
        result = formatter.format_output(data)
        assert result['summary'] == 'Test summary'
        assert result['chart_data'] == {}
        assert result['recommendation'] == 'No data available'
        assert result['risk_flag'] is False
    
    def test_format_output_invalid_types(self):
        """Test formatting with invalid field types."""
        formatter = InsightOutputFormatter()
        data = {
            'summary': 123,  # Should be string
            'chart_data': 'not a dict',  # Should be dict
            'recommendation': True,  # Should be string
            'risk_flag': 'true'  # Should be bool
        }
        result = formatter.format_output(data)
        assert result['summary'] == 'No data available'
        assert result['chart_data'] == {}
        assert result['recommendation'] == 'No data available'
        assert result['risk_flag'] is False

class TestInsightCardRendering:
    """Test suite for insight card rendering."""
    
    def test_render_insight_card_basic(self, sample_insight, mock_st):
        """Test basic insight card rendering checks for highlighted summary."""
        render_insight_card(sample_insight)
        # Manually format the expected highlighted summary
        metadata = extract_metadata(sample_insight['summary'])
        expected_summary = format_markdown_with_highlights(sample_insight['summary'], metadata.highlight_phrases)
        # Check that summary markdown was called with the highlighted text
        mock_st['markdown'].assert_any_call(expected_summary)
        # Check that line chart was called
        mock_st['line_chart'].assert_called_once_with(sample_insight['chart_data']['data'])
    
    def test_render_insight_card_no_title(self, sample_insight, mock_st):
        """Test that the card title markdown is no longer called."""
        render_insight_card(sample_insight)
        # Verify that the title markdown is NOT called
        for call_args, call_kwargs in mock_st['markdown'].call_args_list:
            assert "###" not in call_args[0]
    
    def test_render_insight_card_bar_chart(self, sample_insight, mock_st):
        """Test rendering with bar chart."""
        sample_insight['chart_data']['type'] = 'bar'
        render_insight_card(sample_insight)
        mock_st['bar_chart'].assert_called_once_with(sample_insight['chart_data']['data'])
    
    def test_render_insight_card_risk_flag(self, sample_insight, mock_st):
        """Test rendering with risk flag triggers warning."""
        sample_insight['risk_flag'] = True
        with patch('streamlit.warning') as mock_warning:
             render_insight_card(sample_insight)
             mock_warning.assert_called_once_with('⚠️ Risk flag: This insight requires attention')
    
    def test_render_insight_card_missing_data(self, mock_st):
        """Test rendering with missing summary uses default text."""
        render_insight_card({})
        # Check that the default summary text is displayed
        mock_st['markdown'].assert_any_call("No summary available.")
    
    def test_render_insight_card_invalid_chart(self, sample_insight, mock_st):
        """Test rendering with invalid chart data shows warning."""
        sample_insight['chart_data'] = {'type': 'invalid', 'data': 'not a dataframe'}
        with patch('streamlit.warning') as mock_warning:
             render_insight_card(sample_insight)
             # Expect a warning about chart rendering
             mock_warning.assert_called()
    
    def test_render_insight_card_session_state(self, sample_insight, mock_st):
        """Test session state tracking during rendering and button clicks."""
        # Reset session state before test
        if 'insight_card_renders' in st.session_state: del st.session_state['insight_card_renders']
        if 'insight_button_clicks' in st.session_state: del st.session_state['insight_button_clicks']

        # Initial render
        render_insight_card(sample_insight)
        assert st.session_state.get('insight_card_renders', 0) == 1
        # Capture initial click counts after first render
        initial_follow_up_clicks = st.session_state.get('insight_button_clicks', {}).get('follow_up', 0)
        initial_regenerate_clicks = st.session_state.get('insight_button_clicks', {}).get('regenerate', 0)

        # Simulate button clicks - requires mocking the specific button calls
        # Mock st.button to return True only for specific keys
        def mock_button_logic(label, key):
            if key == "insight_1_follow_up": # Assume second render uses key_prefix insight_1
                return True # Simulate follow-up click
            if key == "insight_1_regenerate":
                return True # Simulate regenerate click
            return False
        
        mock_st['button'].side_effect = mock_button_logic

        # Second render where buttons are clicked
        render_insight_card(sample_insight, show_buttons=True)
        assert st.session_state['insight_card_renders'] == 2
        # Check that the correct keys were used in button calls
        mock_st['button'].assert_any_call('🔍 Follow-up', key='insight_1_follow_up')
        mock_st['button'].assert_any_call('🔁 Regenerate', key='insight_1_regenerate')
        # Verify button click tracking increments by 1
        assert st.session_state['insight_button_clicks']['follow_up'] == initial_follow_up_clicks + 1
        assert st.session_state['insight_button_clicks']['regenerate'] == initial_regenerate_clicks + 1

def test_format_insight_for_display(sample_insight):
    """Test the format_insight_for_display wrapper function."""
    result = format_insight_for_display(sample_insight)
    assert result == sample_insight
    
    # Test with missing fields
    partial_insight = {'summary': 'Test summary'}
    result = format_insight_for_display(partial_insight)
    assert result['summary'] == 'Test summary'
    assert result['chart_data'] == {}
    assert result['recommendation'] == 'No data available'
    assert result['risk_flag'] is False 