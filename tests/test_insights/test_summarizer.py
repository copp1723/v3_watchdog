"""
Unit tests for the LLM summarization system.
"""

import pytest
from unittest.mock import Mock, patch
from src.insights.summarizer import Summarizer, SummarizationError
from src.utils.errors import ValidationError

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate.return_value = """
    Sales rep John Doe showed strong performance in Q1 2025 with total gross profit of $125,000,
    representing a 15% increase over the previous quarter. Deal volume remained steady at 25 units,
    though the average gross per deal increased from $4,500 to $5,000.
    
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
def sample_context():
    """Create sample context for testing."""
    return {
        "entity_name": "John Doe",
        "date_range": "Q1 2025",
        "metrics_table": """
        | Metric | Value |
        |--------|-------|
        | Total Gross | $125,000 |
        | Deal Count | 25 |
        | Avg Gross/Deal | $5,000 |
        | Negative Deals | 2% |
        """
    }

def test_load_template(mock_llm_client, tmp_path):
    """Test template loading."""
    # Create a temporary template file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "test_prompt.tpl"
    template_path.write_text("Test template for {{ entity_name }}")
    
    summarizer = Summarizer(mock_llm_client, str(prompts_dir))
    template = summarizer.load_template("test_prompt.tpl")
    
    assert template is not None
    assert "Test template for" in template.render(entity_name="Test")

def test_format_prompt(mock_llm_client, sample_context, tmp_path):
    """Test prompt formatting with context."""
    # Create a temporary template file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "test_prompt.tpl"
    template_path.write_text("Summary for {{ entity_name }} over {{ date_range }}:\n{{ metrics_table }}")
    
    summarizer = Summarizer(mock_llm_client, str(prompts_dir))
    prompt = summarizer.format_prompt("test_prompt.tpl", **sample_context)
    
    assert sample_context["entity_name"] in prompt
    assert sample_context["date_range"] in prompt
    assert sample_context["metrics_table"] in prompt

def test_format_prompt_missing_context(mock_llm_client, sample_context, tmp_path):
    """Test prompt formatting with missing context variables."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "test_prompt.tpl"
    template_path.write_text("Test template")
    
    summarizer = Summarizer(mock_llm_client, str(prompts_dir))
    
    # Remove required context variable
    invalid_context = sample_context.copy()
    del invalid_context["entity_name"]
    
    with pytest.raises(ValidationError):
        summarizer.format_prompt("test_prompt.tpl", **invalid_context)

def test_summarize_success(mock_llm_client, sample_context, tmp_path):
    """Test successful summary generation."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "test_prompt.tpl"
    template_path.write_text("Test template")
    
    summarizer = Summarizer(mock_llm_client, str(prompts_dir))
    summary = summarizer.summarize("test_prompt.tpl", **sample_context)
    
    assert summary is not None
    assert sample_context["entity_name"] in summary
    assert "Recommendations" in summary
    assert "$" in summary  # Contains monetary values
    assert "%" in summary  # Contains percentages

def test_validate_summary_missing_entity(mock_llm_client, sample_context):
    """Test summary validation with missing entity name."""
    summarizer = Summarizer(mock_llm_client)
    
    # Mock a summary without entity name
    invalid_summary = "Generic performance summary without specific details"
    
    with pytest.raises(SummarizationError):
        summarizer.validate_summary(invalid_summary, sample_context)

def test_validate_summary_missing_metrics(mock_llm_client, sample_context):
    """Test summary validation with missing metrics."""
    summarizer = Summarizer(mock_llm_client)
    
    # Mock a summary without enough metrics
    invalid_summary = f"Summary for {sample_context['entity_name']} without any metrics or recommendations"
    
    with pytest.raises(SummarizationError):
        summarizer.validate_summary(invalid_summary, sample_context)

def test_validate_summary_missing_recommendations(mock_llm_client, sample_context):
    """Test summary validation with missing recommendations."""
    summarizer = Summarizer(mock_llm_client)
    
    # Mock a summary without recommendations
    invalid_summary = f"""
    {sample_context['entity_name']} achieved $125,000 in gross profit with 25 deals.
    Average gross per deal was $5,000 with a 2% negative rate.
    """
    
    with pytest.raises(SummarizationError):
        summarizer.validate_summary(invalid_summary, sample_context)

def test_llm_client_error(mock_llm_client, sample_context, tmp_path):
    """Test handling of LLM client errors."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_path = prompts_dir / "test_prompt.tpl"
    template_path.write_text("Test template")
    
    # Make the LLM client raise an error
    mock_llm_client.generate.side_effect = Exception("LLM API error")
    
    summarizer = Summarizer(mock_llm_client, str(prompts_dir))
    
    with pytest.raises(SummarizationError):
        summarizer.summarize("test_prompt.tpl", **sample_context)