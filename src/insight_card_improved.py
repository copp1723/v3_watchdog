"""
Improved formatter for structuring AI insight outputs in Watchdog AI.

Provides schema validation, mode-based formatting, and fallback logic.
Includes Altair chart integration and mock indicators.
"""

import json
from typing import Dict, Any, Optional, List
import re
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
import altair as alt

# Import chart utilities
from src.chart_utils import extract_chart_data_from_llm_response, build_chart_data

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
                
        # Preserve the is_mock flag if it exists
        if 'is_mock' in data:
            formatted['is_mock'] = data['is_mock']
                
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

def create_altair_chart(chart_data: Dict[str, Any]) -> Optional[alt.Chart]:
    """
    Create an Altair chart based on chart data.
    
    Args:
        chart_data: Chart data dictionary with type, data and title
        
    Returns:
        Altair chart object or None if chart creation fails
    """
    try:
        chart_type = chart_data.get('type', 'bar')
        title = chart_data.get('title', 'Data Analysis')
        data = chart_data.get('data', {})
        x_label = chart_data.get('x_axis_label', 'Category') # Default X label
        y_label = chart_data.get('y_axis_label', 'Value')    # Default Y label
        
        # Create DataFrame based on chart type
        if chart_type in ['bar', 'line']:
            # For bar and line charts, expect x and y arrays
            x_values = data.get('x', [])
            y_values = data.get('y', [])
            
            # Ensure x and y are of equal length
            min_len = min(len(x_values), len(y_values))
            if min_len == 0:
                return None
                
            df = pd.DataFrame({
                'x': x_values[:min_len],
                'y': y_values[:min_len]
            })
            
            # Create chart
            if chart_type == 'bar':
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('x:N', title=x_label), # Use dynamic label
                    y=alt.Y('y:Q', title=y_label), # Use dynamic label
                    tooltip=['x', 'y']
                ).properties(
                    title=title
                )
            else:  # line chart
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X('x:N', title=x_label), # Use dynamic label
                    y=alt.Y('y:Q', title=y_label), # Use dynamic label
                    tooltip=['x', 'y']
                ).properties(
                    title=title
                )
        
        elif chart_type == 'pie':
            # For pie charts, expect labels and values
            labels = data.get('labels', [])
            values = data.get('values', [])
            
            # Ensure labels and values are of equal length
            min_len = min(len(labels), len(values))
            if min_len == 0:
                return None
                
            df = pd.DataFrame({
                'category': labels[:min_len],
                'value': values[:min_len]
            })
            
            # Create pie chart
            chart = alt.Chart(df).mark_arc().encode(
                theta=alt.Theta('value:Q', title=y_label), # Use Y label for theta
                color=alt.Color('category:N', title=x_label), # Use X label for color legend
                tooltip=['category', 'value']
            ).properties(
                title=title
            )
        
        else:
            # Unsupported chart type
            return None
        
        return chart
    
    except Exception as e:
        logger.error(f"Error creating Altair chart: {e}")
        return None

def render_insight_card(insight_data: Dict[str, Any], show_buttons: bool = False, card_index: int = 0) -> None:
    """Render an insight card with the provided data using the styled design."""
    # Unique key prefix for widgets in this card instance
    key_prefix = f"insight_{card_index}" # Use index for uniqueness

    # Format the insight data if needed
    if not isinstance(insight_data, dict):
        insight_data = {}

    # Create container for the card with styling
    with st.container():
        # Add a mock indicator badge if this is a mock response
        if insight_data.get('is_mock', False):
            st.caption("*Mock insight* - LLM integration not active")

        # Extract metadata from summary
        metadata = extract_metadata(insight_data.get('summary', ''))

        # We don't show the summary text here as it's already in the AI bubble
        
        # Display chart if available
        chart_data = insight_data.get("chart_data")
        if chart_data and isinstance(chart_data, dict) and chart_data.get("data"):
            try:
                # Add a little space before the chart
                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                
                # Create a visually appealing card for the chart
                st.markdown("""
                <div style="background-color: #28222225; border-radius: 8px; padding: 16px; margin-top: 12px; margin-bottom: 12px; border: 1px solid #38383840;">
                """, unsafe_allow_html=True)
                
                chart_title = chart_data.get("title", "Chart")
                # Style the chart title
                st.markdown(f"<h4 style='margin-top: 0; margin-bottom: 12px; color: #f4f4f4;'>{chart_title}</h4>", unsafe_allow_html=True)
                
                chart = create_altair_chart(chart_data)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Could not generate chart from data.")
                
                # Close the chart card div
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error rendering chart: {e}\n{traceback.format_exc()}")
                st.error("An error occurred while rendering the chart.")

        # Display recommendation if present in a styled box
        recommendation = insight_data.get('recommendation')
        if recommendation:
            st.markdown(f"""
            <div style="background-color: #2d405930; border-left: 4px solid #4d7c93; color: #e0e0e0; 
                        padding: 12px; margin-top: 16px; margin-bottom: 16px; border-radius: 0 4px 4px 0;">
                <strong>Recommendation:</strong><br>
                {recommendation}
            </div>
            """, unsafe_allow_html=True)

        # Display risk flag if present and True with a more modern style
        if insight_data.get('risk_flag', False):
            st.markdown(f"""
            <div style="background-color: #e4525835; border-left: 4px solid #e45258; color: #e0e0e0; 
                        padding: 12px; margin-top: 16px; margin-bottom: 16px; border-radius: 0 4px 4px 0;">
                <strong>Risk Alert:</strong><br>
                This insight requires immediate attention
            </div>
            """, unsafe_allow_html=True)

        # Removed buttons section (show_buttons=False by default now too)
        # if show_buttons:
        #    button_cols = st.columns(4)
        #    with button_cols[0]:
        #        if st.button('Follow-up', key=f"{key_prefix}_follow_up"):
        #             st.session_state['next_prompt'] = f"Tell me more about: {summary_text}"
        #             st.rerun()
