"""
Formatter for structuring AI insight outputs in Watchdog AI.

Provides schema validation, mode-based formatting, and fallback logic.
"""

import json
from typing import Dict, Any, Optional, List
import re
import traceback
import streamlit as st
import pandas as pd
import logging
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InsightMetadata:
    """Metadata associated with an insight."""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "text_analysis"
    category: str = "general"
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    has_comparison: bool = False
    has_metrics: bool = False
    has_trend: bool = False
    highlight_phrases: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightMetadata':
        """Create InsightMetadata from a dictionary."""
        return cls(
            created_at=data.get('created_at', datetime.now().isoformat()),
            source=data.get('source', 'text_analysis'),
            category=data.get('category', 'general'),
            confidence=data.get('confidence', 1.0),
            tags=data.get('tags', []),
            has_comparison=data.get('has_comparison', False),
            has_metrics=data.get('has_metrics', False),
            has_trend=data.get('has_trend', False),
            highlight_phrases=data.get('highlight_phrases', []),
            entities=data.get('entities', []),
            timeframes=data.get('timeframes', [])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert InsightMetadata to a dictionary."""
        return {
            'created_at': self.created_at,
            'source': self.source,
            'category': self.category,
            'confidence': self.confidence,
            'tags': self.tags,
            'has_comparison': self.has_comparison,
            'has_metrics': self.has_metrics,
            'has_trend': self.has_trend,
            'highlight_phrases': self.highlight_phrases,
            'entities': self.entities,
            'timeframes': self.timeframes
        }

def extract_metadata(text: str) -> InsightMetadata:
    """
    Extract metadata from insight summary text.

    Args:
        text: The insight summary text

    Returns:
        InsightMetadata object with extracted information
    """
    metadata = InsightMetadata()

    # Extract metrics (numbers with currency, percentages)
    metric_pattern = r'\$[\d,]+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?%?'
    metrics = re.findall(metric_pattern, text)
    if metrics:
        metadata.has_metrics = True
        metadata.highlight_phrases.extend(metrics)

    # Extract comparisons
    # Use word boundaries to avoid matching parts of words
    comparison_words = ['compare', 'versus', 'vs', 'higher', 'lower', 'more', 'less', 'increased', 'decreased']
    text_lower = text.lower()
    if any(re.search(rf'\b{word}\b', text_lower) for word in comparison_words):
        metadata.has_comparison = True

    # Extract trends
    trend_words = ['trend', 'over time', 'growth', 'decline', 'pattern']
    if any(re.search(rf'\b{word}\b', text_lower) for word in trend_words):
        metadata.has_trend = True

    return metadata

def format_markdown_with_highlights(text: str, phrases: Optional[List[str]] = None) -> str:
    """
    Format text with markdown bold for specified phrases.
    Handles potential regex conflicts in phrases and preserves original casing.

    Args:
        text: The text to format
        phrases: List of phrases to highlight

    Returns:
        Formatted text with markdown highlights
    """
    if not phrases:
        return text

    # Create a pattern that matches any of the phrases using word boundaries
    # Escape each phrase and join with | (OR)
    # Sort by length descending to ensure longer phrases are matched first
    escaped_phrases = [re.escape(p) for p in sorted(phrases, key=len, reverse=True)]
    # Remove \b word boundaries as they might interfere with symbols like $ and %
    pattern = rf"(?:{'|'.join(escaped_phrases)})"

    try:
        # Use re.sub with a function to wrap matches in ** **
        # The function receives the match object and returns the replacement string
        formatted_text = re.sub(pattern, lambda match: f"**{match.group(0)}**", text, flags=re.IGNORECASE)
    except re.error as e:
        logger.warning(f"Regex error during highlighting: {e}")
        formatted_text = text # Return original text on error

    return formatted_text

class InsightOutputFormatter:
    """Formats LLM output to match the expected insight schema."""

    def __init__(self):
        self.schema = {
            'summary': str,
            'chart_data': dict,
            'recommendation': str,
            'risk_flag': bool
        }
        self.defaults = {
            'summary': 'No data available',
            'chart_data': {},
            'recommendation': 'No data available',
            'risk_flag': False
        }

    def format_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and format the data according to the schema.

        Args:
            data: Raw data dictionary from LLM

        Returns:
            Formatted and validated dictionary
        """
        formatted = {}
        for key, expected_type in self.schema.items():
            value = data.get(key)
            if value is None or not isinstance(value, expected_type):
                logger.warning(f"Field '{key}' missing or invalid type. Using default.")
                formatted[key] = self.defaults[key]
            else:
                formatted[key] = value
        return formatted

    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """
        Attempt to parse JSON output, with fallback for errors.

        Args:
            output: String output from LLM, potentially JSON.

        Returns:
            Parsed dictionary or default structure on error.
        """
        try:
            # Attempt to find JSON block within ```json ... ``` markers
            match = re.search(r'```json\s*({.*?})\s*```', output, re.DOTALL)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
            else:
                # Fallback: try parsing the whole string as JSON
                data = json.loads(output)
            
            # Basic validation: ensure it's a dictionary
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not a dictionary.")
                
            return data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON. Using fallback.")
            return {'summary': output} # Use the raw output as summary
        except Exception as e:
            logger.error(f"Error processing LLM output: {e}\nTraceback: {traceback.format_exc()}")
            return self.defaults

    def process_llm_output(self, output: str) -> Dict[str, Any]:
        """
        Process raw LLM string output, parse if JSON, and format.

        Args:
            output: Raw string output from the LLM.

        Returns:
            Formatted insight dictionary.
        """
        parsed_data = self._parse_json_output(output)
        return self.format_output(parsed_data)

def format_insight_for_display(insight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function to format insight data using InsightOutputFormatter.
    Ensures consistent formatting before rendering.

    Args:
        insight: Potentially raw insight data dictionary.

    Returns:
        Formatted insight data dictionary.
    """
    formatter = InsightOutputFormatter()
    return formatter.format_output(insight)

def render_insight_card(insight_data: Dict[str, Any], show_buttons: bool = True) -> None:
    """Render an insight card with the provided data."""
    # Initialize session state if not exists
    if 'insight_card_renders' not in st.session_state:
        st.session_state['insight_card_renders'] = 0
    if 'insight_button_clicks' not in st.session_state:
        st.session_state['insight_button_clicks'] = {
            'follow_up': 0,
            'regenerate': 0
        }

    # Unique key prefix for widgets in this card instance
    render_count = st.session_state['insight_card_renders']
    key_prefix = f"insight_{render_count}"
    st.session_state['insight_card_renders'] += 1

    # Create container for the card
    with st.container():
        # Extract metadata from summary
        metadata = extract_metadata(insight_data.get('summary', ''))

        # Display summary with highlights
        summary_text = insight_data.get('summary')
        if summary_text:
            formatted_summary = format_markdown_with_highlights(summary_text, metadata.highlight_phrases)
            st.markdown(formatted_summary)
        else:
            st.markdown("No summary available.")

        # Display chart if present
        chart_info = insight_data.get('chart_data')
        if isinstance(chart_info, dict):
            chart_data = chart_info.get('data')
            chart_type = chart_info.get('type')
            
            # Check if data is suitable for charting before trying
            if isinstance(chart_data, (pd.DataFrame, dict)) and chart_type in ['bar', 'line']:
                try:
                    if chart_type == 'bar':
                        st.bar_chart(chart_data)
                    elif chart_type == 'line':
                        st.line_chart(chart_data)
                except Exception as e:
                    logger.error(f"Error rendering chart ({chart_type}): {e}")
                    st.warning("Could not render chart data.")
            elif chart_info: # If chart_info exists but data/type is invalid
                logger.warning(f"Invalid chart data or type provided: {chart_info}")
                st.warning("Chart data is invalid or type is unsupported.")

        # Display recommendation if present
        recommendation = insight_data.get('recommendation')
        if recommendation:
            st.info(recommendation)

        # Display risk flag if present and True
        if insight_data.get('risk_flag', False):
            st.warning('⚠️ Risk flag: This insight requires attention')

        # Add buttons if requested
        if show_buttons:
            col1, col2 = st.columns(2)
            with col1:
                if st.button('🔍 Follow-up', key=f"{key_prefix}_follow_up"):
                    st.session_state['insight_button_clicks']['follow_up'] += 1
            with col2:
                if st.button('🔁 Regenerate', key=f"{key_prefix}_regenerate"):
                    st.session_state['insight_button_clicks']['regenerate'] += 1
