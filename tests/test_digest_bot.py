"""
Unit tests for the DigestBot module.
"""

import pytest
import streamlit as st
from src.digest_bot import DigestBot, run_nightly_digest

@pytest.fixture
def sample_insights():
    """Fixture providing a list of sample insights."""
    return [
        {
            "summary": "Sales increased by 15% compared to last month.",
            "recommendation": "Increase marketing budget to sustain growth.",
            "risk_flag": False,
            "timestamp": "2023-05-01T08:00:00Z"
        },
        {
            "summary": "Inventory turnover decreased by 10%, indicating potential overstock.",
            "recommendation": "Review inventory levels and consider promotions.",
            "risk_flag": True,
            "timestamp": "2023-05-01T09:00:00Z"
        },
        {
            "summary": "Customer satisfaction improved by 8% this quarter.",
            "recommendation": "Continue focusing on customer service training.",
            "risk_flag": False,
            "timestamp": "2023-05-01T07:00:00Z"
        },
        {
            "summary": "Revenue growth is steady at 5% month-over-month.",
            "recommendation": "Maintain current strategies for consistent growth.",
            "risk_flag": False,
            "timestamp": "2023-05-01T06:00:00Z"
        },
        {
            "summary": "Significant drop in website traffic by 20% this week.",
            "recommendation": "Investigate SEO and marketing channels immediately.",
            "risk_flag": True,
            "timestamp": "2023-05-01T10:00:00Z"
        },
        {
            "summary": "Older insight that should not be included due to max limit.",
            "recommendation": "This should not appear.",
            "risk_flag": False,
            "timestamp": "2023-05-01T05:00:00Z"
        }
    ]

@pytest.fixture
def empty_insights():
    """Fixture providing an empty list of insights."""
    return []

@pytest.fixture
def digest_bot(sample_insights):
    """Fixture providing a DigestBot instance with sample insights."""
    return DigestBot(sample_insights, max_insights=5)

@pytest.fixture
def digest_bot_empty(empty_insights):
    """Fixture providing a DigestBot instance with no insights."""
    return DigestBot(empty_insights, max_insights=5)

@pytest.fixture
def sample_history(sample_insights):
    """Fixture providing a sample conversation history."""
    return [
        {"prompt": "Prompt 1", "response": sample_insights[0]},
        {"prompt": "Prompt 2", "response": sample_insights[1]},
        {"prompt": "Prompt 3", "response": sample_insights[2]},
        {"prompt": "Prompt 4", "response": sample_insights[3]},
        {"prompt": "Prompt 5", "response": sample_insights[4]},
        {"prompt": "Prompt 6", "response": sample_insights[5]}
    ]

def test_digest_bot_init(digest_bot, sample_insights):
    """Test initialization of DigestBot."""
    assert len(digest_bot.insights) == len(sample_insights)
    assert digest_bot.max_insights == 5
    assert 'Morning Briefing' in digest_bot.digest_title

def test_filter_top_insights(digest_bot, sample_insights):
    """Test filtering of top insights based on risk_flag and timestamp."""
    top_insights = digest_bot.filter_top_insights()
    
    assert len(top_insights) == 5  # Should be limited to max_insights
    assert top_insights[0]['risk_flag'] is True  # Highest priority: risk_flag=True, newest timestamp
    assert 'website traffic' in top_insights[0]['summary'].lower()
    assert top_insights[1]['risk_flag'] is True  # Next: risk_flag=True, older timestamp
    assert 'inventory turnover' in top_insights[1]['summary'].lower()

def test_filter_top_insights_empty(digest_bot_empty):
    """Test filtering top insights with empty list."""
    top_insights = digest_bot_empty.filter_top_insights()
    assert len(top_insights) == 0

def test_format_slack_message(digest_bot):
    """Test formatting top insights as a Slack message."""
    top_insights = digest_bot.filter_top_insights()
    message = digest_bot.format_slack_message(top_insights)
    
    assert 'Morning Briefing' in message
    assert 'top 5 insights' in message.lower()
    assert '🚨' in message  # Risk flag emoji should appear
    assert 'website traffic' in message.lower()  # Top insight should be included
    assert 'Recommendation' in message

def test_format_slack_message_empty(digest_bot_empty):
    """Test formatting Slack message with no insights."""
    message = digest_bot_empty.format_slack_message([])
    assert 'Morning Briefing' in message
    assert 'No insights available' in message

def test_format_email_content(digest_bot):
    """Test formatting top insights as email content (HTML)."""
    top_insights = digest_bot.filter_top_insights()
    content = digest_bot.format_email_content(top_insights)
    
    assert '<html>' in content
    assert 'Morning Briefing' in content
    assert 'top 5 insights' in content.lower()
    assert '🚨' in content  # Risk flag emoji should appear
    assert 'website traffic' in content.lower()
    assert 'Recommendation' in content

def test_format_email_content_empty(digest_bot_empty):
    """Test formatting email content with no insights."""
    content = digest_bot_empty.format_email_content([])
    assert '<html>' in content
    assert 'Morning Briefing' in content
    assert 'No insights available' in content

def test_format_dashboard_card(digest_bot):
    """Test formatting top insights as a dashboard card."""
    top_insights = digest_bot.filter_top_insights()
    card = digest_bot.format_dashboard_card(top_insights)
    
    assert 'Morning Briefing' in card
    assert 'Top 5 Insights' in card
    assert '🚨' in card  # Risk flag emoji should appear
    assert 'website traffic' in card.lower()
    assert 'Recommendation' in card

def test_format_dashboard_card_empty(digest_bot_empty):
    """Test formatting dashboard card with no insights."""
    card = digest_bot_empty.format_dashboard_card([])
    assert 'Morning Briefing' in card
    assert 'No insights available' in card

def test_generate_digest_slack(digest_bot):
    """Test generating a digest in Slack format."""
    digest = digest_bot.generate_digest('slack')
    assert 'Morning Briefing' in digest
    assert 'insights for today' in digest.lower()

def test_generate_digest_email(digest_bot):
    """Test generating a digest in Email format."""
    digest = digest_bot.generate_digest('email')
    assert '<html>' in digest
    assert 'Morning Briefing' in digest

def test_generate_digest_dashboard(digest_bot):
    """Test generating a digest in Dashboard format."""
    digest = digest_bot.generate_digest('dashboard')
    assert 'Morning Briefing' in digest
    assert 'Top' in digest

def test_generate_digest_invalid_format(digest_bot):
    """Test generating a digest with an invalid format."""
    with pytest.raises(ValueError):
        digest_bot.generate_digest('invalid')

def test_run_nightly_digest(sample_history):
    """Test running a nightly digest from conversation history."""
    digest = run_nightly_digest(sample_history, output_format='slack', max_insights=3)
    assert 'Morning Briefing' in digest
    assert 'top 3 insights' in digest.lower()  # Should respect max_insights
    assert '🚨' in digest  # Should include risk-flagged insights

def test_run_nightly_digest_empty():
    """Test running a nightly digest with empty history."""
    digest = run_nightly_digest([], output_format='slack')
    assert 'Morning Briefing' in digest
    assert 'No insights available' in digest 