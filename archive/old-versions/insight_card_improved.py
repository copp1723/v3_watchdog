"""
Enhanced formatter for structuring AI insight outputs in Watchdog AI.

Provides improved schema validation, markdown formatting, and output parsing.
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

class EnhancedInsightOutputFormatter:
    """Enhanced formatter for insight responses with improved markdown support."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.required_fields = ["summary"]
        self.optional_fields = ["value_insights", "actionable_flags", "confidence", "is_mock"]
    
    def format_response(self, response_text: str) -> Dict[str, Any]:
        """
        Format the raw response text into a structured insight dict with improved formatting.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            A structured insight dictionary with markdown formatting
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
            validated = self._validate_and_complete(response)
            
            # Apply markdown formatting to the insight components
            formatted = self._apply_markdown_formatting(validated)
            
            return formatted
            
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
        
        # If no JSON found, try to parse as a structured text response
        try:
            return self._parse_structured_text(text)
        except:
            # If all parsing fails, construct a basic response from the text
            return {
                "summary": text[:200] + "..." if len(text) > 200 else text,
                "value_insights": [text],
                "actionable_flags": [],
                "confidence": "low"
            }
    
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """
        Parse non-JSON text that follows a structured format.
        
        Args:
            text: The structured text to parse
            
        Returns:
            A dictionary representation of the structured text
        """
        result = {
            "summary": "",
            "value_insights": [],
            "actionable_flags": [],
            "confidence": "medium"
        }
        
        # Split by double newlines to get sections
        sections = text.split('\n\n')
        
        # Process each section
        current_section = None
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Check for section headers
            if section.lower().startswith('summary:'):
                result['summary'] = section[8:].strip()
                current_section = 'summary'
            elif any(section.lower().startswith(header) for header in ['key insights:', 'insights:', 'value insights:']):
                current_section = 'value_insights'
            elif any(section.lower().startswith(header) for header in ['action items:', 'recommendations:', 'actionable flags:']):
                current_section = 'actionable_flags'
            elif section.lower().startswith('confidence:'):
                confidence_text = section[11:].strip().lower()
                if any(level in confidence_text for level in ['high', 'medium', 'low']):
                    for level in ['high', 'medium', 'low']:
                        if level in confidence_text:
                            result['confidence'] = level
                            break
                current_section = None
            # If we're in a list section, process bullet points
            elif current_section in ['value_insights', 'actionable_flags']:
                # Split by newlines and process each line
                for line in section.split('\n'):
                    line = line.strip()
                    # Check if it's a bullet point (-, *, •)
                    if line.startswith(('-', '*', '•')):
                        item = line[1:].strip()
                        result[current_section].append(item)
                    elif line:  # If not a bullet point but not empty, add it anyway
                        result[current_section].append(line)
        
        return result
    
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
    
    def _apply_markdown_formatting(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply markdown formatting to insight components.
        
        Args:
            insight: The validated insight dictionary
            
        Returns:
            Insight dictionary with markdown formatting applied
        """
        formatted = insight.copy()
        
        # Extract metrics and important terms for highlighting
        summary_metadata = extract_metadata(formatted['summary'])
        highlight_phrases = summary_metadata.highlight_phrases
        
        # Format summary with bold highlights
        formatted['summary'] = format_markdown_with_highlights(formatted['summary'], highlight_phrases)
        
        # Format value insights
        formatted_insights = []
        for item in formatted['value_insights']:
            # Add bold for metrics and key terms
            item_metadata = extract_metadata(item)
            formatted_item = format_markdown_with_highlights(item, item_metadata.highlight_phrases)
            
            # Add markdown bullet point if not already present
            if not formatted_item.startswith(('- ', '* ', '• ')):
                formatted_item = f"• {formatted_item}"
                
            formatted_insights.append(formatted_item)
        
        formatted['value_insights'] = formatted_insights
        
        # Format actionable flags
        formatted_flags = []
        for flag in formatted['actionable_flags']:
            # Add markdown bullet point if not already present
            if not flag.startswith(('- ', '* ', '• ')):
                flag = f"→ {flag}"
            formatted_flags.append(flag)
        
        formatted['actionable_flags'] = formatted_flags
        
        return formatted

def format_insight_for_display(insight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function to format insight data using EnhancedInsightOutputFormatter.
    Ensures consistent formatting before rendering with improved markdown.

    Args:
        insight: Potentially raw insight data dictionary.

    Returns:
        Formatted insight data dictionary with markdown enhancements.
    """
    formatter = EnhancedInsightOutputFormatter()
    
    # Handle different input formats
    if isinstance(insight, str):
        return formatter.format_response(insight)
    elif isinstance(insight, dict) and 'response' in insight:
        # For conversation history format
        if isinstance(insight['response'], str):
            return formatter.format_response(insight['response'])
        else:
            return insight['response']  # Already formatted
    elif isinstance(insight, dict) and 'summary' in insight:
        # Already in the expected format
        return insight
    else:
        # Try to format as a raw response
        return formatter.format_response(json.dumps(insight))

def render_enhanced_insight_card(insight: Dict[str, Any], index: int = 0, show_buttons: bool = True) -> None:
    """
    Render a structured insight as a card in the UI with enhanced formatting.
    
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
    
    # Render the insight card with enhanced styling
    with st.container():
        # Use a card-like container with padding and border
        with st.container():
            # Header with confidence indicator and improved styling
            if confidence == "high":
                st.markdown(f"<h4 style='color: #2e7d32; border-left: 4px solid #2e7d32; padding-left: 10px;'>🟢 {summary}</h4>", unsafe_allow_html=True)
            elif confidence == "medium":
                st.markdown(f"<h4 style='color: #f9a825; border-left: 4px solid #f9a825; padding-left: 10px;'>🟡 {summary}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4 style='color: #c62828; border-left: 4px solid #c62828; padding-left: 10px;'>🔴 {summary}</h4>", unsafe_allow_html=True)
            
            # Draw a light separator line
            st.markdown("<hr style='margin: 0.5em 0; opacity: 0.3;'>", unsafe_allow_html=True)
            
            # Value insights with improved formatting
            if value_insights:
                st.markdown("##### Key Insights:")
                for insight_text in value_insights:
                    st.markdown(insight_text)
            
            # Actionable flags with improved formatting
            if actionable_flags:
                st.markdown("##### Action Items:")
                for flag in actionable_flags:
                    st.markdown(flag)
            
            # Metadata with subtle styling
            if is_mock:
                st.markdown("<div style='color: #666; font-size: 0.8em; margin-top: 1em;'>*This is a mock response for testing*</div>", unsafe_allow_html=True)
            
            # Interaction buttons with improved styling
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
                with col3:
                    if st.button(f"📊 Visualize Data", key=f"visualize_{index}"):
                        st.session_state.visualize_insight = True
                        st.session_state.visualize_index = index
                        st.rerun()
            
            st.markdown("---")

# For backwards compatibility
render_insight_card = render_enhanced_insight_card
