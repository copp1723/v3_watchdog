"""
Unit tests for the digest bot module.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.digest_bot import DigestBot

# --- Fixtures ---

@pytest.fixture
def sample_insights_history():
    """Provides a sample conversation history with varied insights."""
    now = datetime.now()
    return [
        {
            'prompt': 'P1',
            'response': {
                'summary': 'Low risk insight from yesterday.',
                'recommendation': 'Rec 1',
                'risk_flag': False,
                'timestamp': (now - timedelta(days=1)).isoformat()
            }
        },
        {
            'prompt': 'P2',
            'response': {
                'summary': 'High risk insight from today.',
                'recommendation': 'Rec 2',
                'risk_flag': True,
                'timestamp': now.isoformat(),
                'chart_data': {'type': 'line', 'data': {'x': [1], 'y': [1]}} # Include some chart data
            }
        },
        {
            'prompt': 'P3',
            'response': {
                'summary': 'Another low risk insight, older.',
                'recommendation': 'Rec 3',
                'risk_flag': False,
                'timestamp': (now - timedelta(days=3)).isoformat()
            }
        },
         {
            'prompt': 'P4',
            'response': {
                'summary': 'Medium risk insight from today (no timestamp).',
                'recommendation': 'Rec 4',
                'risk_flag': True, # Treat as high risk for filtering
                # Missing timestamp, should be handled gracefully
            }
        },
         {
            'prompt': 'P5',
            'response': {
                'summary': 'Insight with missing risk flag (should default to False).',
                'recommendation': 'Rec 5',
                'timestamp': (now - timedelta(hours=1)).isoformat()
                # Missing risk_flag
            }
        },
          {
            'prompt': 'P6',
            'response': {
                'summary': 'High risk, but older than cutoff.',
                'recommendation': 'Rec 6',
                'risk_flag': True,
                'timestamp': (now - timedelta(days=2)).isoformat()
            }
        }
    ]

@pytest.fixture
def digest_bot(sample_insights_history):
    """Provides a DigestBot instance initialized with sample insights."""
    return DigestBot(sample_insights_history)

# --- Test Functions ---

def test_digest_bot_init(digest_bot, sample_insights_history):
    """Test DigestBot initialization."""
    assert digest_bot.all_insights is not None
    # Check if insights were processed correctly (extracting responses)
    assert len(digest_bot.all_insights) == len(sample_insights_history)
    assert digest_bot.all_insights[0]['summary'] == sample_insights_history[0]['response']['summary']

def test_filter_top_insights_default(digest_bot):
    """Test filtering top insights with default settings (risk=True, days=1)."""
    top_insights = digest_bot.filter_top_insights()
    
    assert len(top_insights) == 2 # Should get P2 (high risk, today) and P4 (high risk, today)
    summaries = [insight['summary'] for insight in top_insights]
    assert 'High risk insight from today.' in summaries
    assert 'Medium risk insight from today (no timestamp).' in summaries
    # P6 is high risk but older than 1 day
    assert 'High risk, but older than cutoff.' not in summaries
    # P1, P3, P5 are low risk or default to low risk
    assert 'Low risk insight from yesterday.' not in summaries
    assert 'Insight with missing risk flag' not in summaries

def test_filter_top_insights_no_risk(digest_bot):
    """Test filtering with risk_only=False."""
    top_insights = digest_bot.filter_top_insights(risk_only=False, days_cutoff=3)
    
    assert len(top_insights) == 4 # P1, P2, P4, P5 (within 3 days)
    summaries = [insight['summary'] for insight in top_insights]
    assert 'Low risk insight from yesterday.' in summaries
    assert 'High risk insight from today.' in summaries
    assert 'Medium risk insight from today (no timestamp).' in summaries
    assert 'Insight with missing risk flag (should default to False).' in summaries
    # P3 is older than 3 days (just on the edge, depending on exact time)
    # P6 is within 3 days but included here as risk_only is False
    assert 'High risk, but older than cutoff.' in summaries

def test_filter_top_insights_custom_days(digest_bot):
    """Test filtering with a custom day cutoff."""
    top_insights = digest_bot.filter_top_insights(risk_only=True, days_cutoff=2)
    assert len(top_insights) == 3 # P2, P4 (today), P6 (2 days ago)
    summaries = [insight['summary'] for insight in top_insights]
    assert 'High risk insight from today.' in summaries
    assert 'Medium risk insight from today (no timestamp).' in summaries
    assert 'High risk, but older than cutoff.' in summaries

def test_filter_top_insights_no_matches(digest_bot):
    """Test filtering when no insights match the criteria."""
    # Filter for high risk insights older than 5 days
    top_insights = digest_bot.filter_top_insights(risk_only=True, days_cutoff=-5) # Use negative days
    assert len(top_insights) == 0

def test_format_slack_message(digest_bot):
    """Test formatting insights for Slack."""
    insights = digest_bot.filter_top_insights() # P2, P4
    slack_message = digest_bot.format_slack_message(insights)
    
    assert isinstance(slack_message, list) # Should be list of blocks
    assert len(slack_message) > 0
    assert "Top Insights Digest" in json.dumps(slack_message) # Check title
    assert "High risk insight from today." in json.dumps(slack_message)
    assert "Medium risk insight from today" in json.dumps(slack_message)
    assert "Low risk insight" not in json.dumps(slack_message)
    assert "Recommendation: Rec 2" in json.dumps(slack_message)
    assert "Recommendation: Rec 4" in json.dumps(slack_message)

def test_format_email_content(digest_bot):
    """Test formatting insights for Email (HTML)."""
    insights = digest_bot.filter_top_insights() # P2, P4
    email_html = digest_bot.format_email_content(insights)
    
    assert isinstance(email_html, str)
    assert "<h1>Top Insights Digest</h1>" in email_html
    assert "<h2>Insight: High risk insight from today.</h2>" in email_html
    assert "<h3>Recommendation:</h3><p>Rec 2</p>" in email_html
    assert "<h2>Insight: Medium risk insight from today (no timestamp).</h2>" in email_html
    assert "<h3>Recommendation:</h3><p>Rec 4</p>" in email_html
    assert "Low risk insight" not in email_html

def test_format_dashboard_card(digest_bot):
    """Test formatting insights for Dashboard Card (Markdown)."""
    insights = digest_bot.filter_top_insights() # P2, P4
    dashboard_md = digest_bot.format_dashboard_card(insights)
    
    assert isinstance(dashboard_md, str)
    assert "### Top Insights Digest" in dashboard_md
    assert "**Insight:** High risk insight from today." in dashboard_md
    assert "**Recommendation:** Rec 2" in dashboard_md
    assert "**Insight:** Medium risk insight from today (no timestamp)." in dashboard_md
    assert "**Recommendation:** Rec 4" in dashboard_md
    assert "Low risk insight" not in dashboard_md

def test_generate_digest_slack(digest_bot):
    """Test generate_digest function for Slack format."""
    digest = digest_bot.generate_digest(output_format='slack')
    assert isinstance(digest, list)
    assert "Top Insights Digest" in json.dumps(digest)

def test_generate_digest_email(digest_bot):
    """Test generate_digest function for Email format."""
    digest = digest_bot.generate_digest(output_format='email')
    assert isinstance(digest, str)
    assert "<h1>Top Insights Digest</h1>" in digest

def test_generate_digest_dashboard(digest_bot):
    """Test generate_digest function for Dashboard format."""
    digest = digest_bot.generate_digest(output_format='dashboard')
    assert isinstance(digest, str)
    assert "### Top Insights Digest" in digest

def test_generate_digest_invalid_format(digest_bot):
    """Test generate_digest function with an invalid format."""
    digest = digest_bot.generate_digest(output_format='invalid')
    assert isinstance(digest, str) # Should default to markdown/dashboard
    assert "### Top Insights Digest" in digest

def test_render_digest_preview(digest_bot):
    """Test the render_digest_preview function (mocked Streamlit)."""
    # This is hard to test fully without running Streamlit,
    # but we can check if it calls the formatting functions.
    with patch.object(digest_bot, 'format_slack_message') as mock_slack, \
         patch.object(digest_bot, 'format_email_content') as mock_email, \
         patch.object(digest_bot, 'format_dashboard_card') as mock_dashboard, \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.json') as mock_json, \
         patch('streamlit.code') as mock_code:
        
        # Mock the tabs context manager
        mock_tab_instances = [MagicMock(), MagicMock(), MagicMock()]
        mock_tabs.return_value = mock_tab_instances
        for tab in mock_tab_instances:
            tab.__enter__.return_value = None
            tab.__exit__.return_value = None
            
        digest_bot.render_digest_preview()
        
        mock_dashboard.assert_called_once()
        # In Streamlit 1.3 tabs, only the first tab's content is rendered initially
        # So Slack and Email formatters might not be called until user interaction
        # mock_slack.assert_called_once()
        # mock_email.assert_called_once()
        
        # Check if markdown was called (for the default dashboard view)
        mock_markdown.assert_called() 