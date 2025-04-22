"""
Insight metadata handling for Watchdog AI.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re

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
        # Return original text on error
        formatted_text = text

    return formatted_text

