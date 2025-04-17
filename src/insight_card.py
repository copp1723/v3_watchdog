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
    """Formats and validates insight responses from LLM."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.required_fields = ["summary"]
        self.optional_fields = ["value_insights", "actionable_flags", "confidence", "is_mock"]
    
    def format_response(self, response_text: str) -> Dict[str, Any]:
        """
        Format the raw response text into a structured insight dict.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            A structured insight dictionary
        """
        try:
            # Try to parse as JSON first
            try:
                response = json.loads(response_text)
                print("[DEBUG] Response successfully parsed as JSON")
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from markdown
                print("[DEBUG] Not valid JSON, trying to extract from text")
                response = self._extract_json_from_text(response_text)
                
            # Validate and fill in missing fields
            return self._validate_and_complete(response)
            
        except Exception as e:
            print(f"[ERROR] Error formatting response: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            # Return a fallback response
            return {
                "summary": "Failed to format insight response",
                "value_insights": [
                    "The system received a response that could not be properly formatted.",
                    f"Error: {str(e)}"
                ],
                "actionable_flags": [],
                "confidence": "low",
                "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
            }
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from a text that may contain markdown or other formatting.
        
        Args:
            text: The text that might contain JSON
            
        Returns:
            The extracted JSON as a dictionary
        """
        # Look for JSON between ``` blocks (common in markdown)
        import re
        json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        
        if json_blocks:
            for block in json_blocks:
                try:
                    return json.loads(block.strip())
                except:
                    continue
        
        # Try to find JSON between { and } (the outermost curly braces)
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
        
        # If no JSON found, construct a basic response from the text
        return {
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "value_insights": [text],
            "actionable_flags": [],
            "confidence": "low"
        }
    
    def _validate_and_complete(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate response has required fields and fill in any missing optional fields.
        
        Args:
            response: The parsed response dictionary
            
        Returns:
            A validated and completed response dictionary
        """
        validated = {}
        
        # Check required fields
        for field in self.required_fields:
            if field not in response or not response[field]:
                print(f"[WARN] Required field '{field}' missing or empty in response")
                validated[field] = f"Missing {field}"
            else:
                validated[field] = response[field]
        
        # Fill in optional fields
        for field in self.optional_fields:
            if field not in response or response[field] is None:
                if field == "value_insights" or field == "actionable_flags":
                    validated[field] = []
                elif field == "confidence":
                    validated[field] = "medium"
                elif field == "is_mock":
                    validated[field] = False
                else:
                    validated[field] = None
            else:
                validated[field] = response[field]
        
        return validated

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
    return formatter.format_response(insight['summary'])

def render_insight_card(insight: Dict[str, Any], index: int = 0, show_buttons: bool = True) -> None:
    """
    Render a structured insight as a card in the UI.
    
    Args:
        insight: The insight dictionary to render
        index: The index of this insight in the conversation (for button keys)
        show_buttons: Whether to show interaction buttons
    """
    # Ensure insight is properly formatted
    if not isinstance(insight, dict) or "summary" not in insight:
        st.error("Invalid insight data format")
        return
    
    # Extract key fields with defaults
    summary = insight.get("summary", "No summary available")
    value_insights = insight.get("value_insights", [])
    actionable_flags = insight.get("actionable_flags", [])
    confidence = insight.get("confidence", "medium").lower()
    is_mock = insight.get("is_mock", False)
    
    # Render the insight card
    with st.container():
        # Header with confidence indicator
        if confidence == "high":
            st.markdown(f"#### 🟢 {summary}")
        elif confidence == "medium":
            st.markdown(f"#### 🟡 {summary}")
        else:
            st.markdown(f"#### 🔴 {summary}")
        
        # Value insights
        if value_insights:
            st.markdown("**Key insights:**")
            for insight_text in value_insights:
                st.markdown(f"- {insight_text}")
        
        # Actionable flags
        if actionable_flags:
            st.markdown("**Action items:**")
            for flag in actionable_flags:
                st.markdown(f"- {flag}")
        
        # Metadata
        if is_mock:
            st.caption("*This is a mock response for testing*")
        
        # Interaction buttons
        if show_buttons:
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button(f"🔄 Regenerate", key=f"regenerate_{index}"):
                    st.session_state.regenerate_insight = True
                    st.session_state.regenerate_index = index
                    st.rerun()
            with col2:
                if st.button(f"📋 Copy", key=f"copy_{index}"):
                    # Can't directly interact with clipboard, so show instructions
                    st.info("Copy functionality requires clipboard API integration")
        
        st.markdown("---")
