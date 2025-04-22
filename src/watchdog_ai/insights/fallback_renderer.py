"""
Fallback renderer for handling query failures with user-friendly messages.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class FallbackReason(Enum):
    """Reasons for falling back to simpler analysis."""
    LOW_PRECISION = "low_precision"
    MISSING_COLUMNS = "missing_columns"
    DATA_QUALITY = "data_quality"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    COMPLEX_QUERY = "complex_query"
    SYSTEM_ERROR = "system_error"

@dataclass
class FallbackContext:
    """Context for fallback rendering."""
    reason: FallbackReason
    details: Dict[str, Any]
    original_query: str
    timestamp: str = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class FallbackRenderer:
    """Renders user-friendly fallback messages and suggestions."""
    
    def __init__(self):
        """Initialize the fallback renderer."""
        self.templates = {
            FallbackReason.LOW_PRECISION: {
                "message": "I'm not confident I can answer that question accurately.",
                "suggestion": "Try being more specific about what you want to analyze.",
                "examples": [
                    "Instead of 'Show me sales', try 'Show me total sales by region for last month'",
                    "Instead of 'What about profit?', try 'What is the average gross profit per sale?'"
                ]
            },
            FallbackReason.MISSING_COLUMNS: {
                "message": "I can't find some of the data needed to answer that question.",
                "suggestion": "Try asking about one of these available metrics:",
                "examples": []  # Will be filled from context
            },
            FallbackReason.DATA_QUALITY: {
                "message": "The data quality isn't good enough for reliable analysis.",
                "suggestion": "Try filtering the data or asking about a different time period.",
                "examples": [
                    "Try 'Show me sales where the data is complete'",
                    "Try looking at a more recent time period"
                ]
            },
            FallbackReason.AMBIGUOUS_INTENT: {
                "message": "I'm not sure exactly what you're asking for.",
                "suggestion": "Could you clarify which of these you mean?",
                "examples": []  # Will be filled from context
            },
            FallbackReason.COMPLEX_QUERY: {
                "message": "That's a complex question that might need to be broken down.",
                "suggestion": "Try asking one thing at a time:",
                "examples": [
                    "First ask about overall sales trends",
                    "Then ask about specific regions or time periods"
                ]
            },
            FallbackReason.SYSTEM_ERROR: {
                "message": "Sorry, I encountered a technical issue.",
                "suggestion": "Please try again or contact support if the issue persists.",
                "examples": []
            }
        }
    
    def render_fallback(self, context: FallbackContext) -> Dict[str, Any]:
        """
        Render a user-friendly fallback message with suggestions.
        
        Args:
            context: Fallback context
            
        Returns:
            Dictionary containing the rendered message and suggestions
        """
        template = self.templates[context.reason]
        
        # Get base message and suggestion
        message = template["message"]
        suggestion = template["suggestion"]
        
        # Get examples (either from template or context)
        examples = context.details.get("examples", template["examples"])
        
        # Add specific details based on reason
        if context.reason == FallbackReason.MISSING_COLUMNS:
            available_columns = context.details.get("available_columns", [])
            examples = [f"- {col}" for col in available_columns[:5]]
            
        elif context.reason == FallbackReason.AMBIGUOUS_INTENT:
            possible_intents = context.details.get("possible_intents", [])
            examples = [f"- {intent}" for intent in possible_intents]
        
        # Format the response
        response = {
            "type": "fallback",
            "message": message,
            "suggestion": suggestion,
            "examples": examples,
            "original_query": context.original_query,
            "timestamp": context.timestamp,
            "reason": context.reason.value
        }
        
        # Add any additional context-specific details
        if "precision_score" in context.details:
            response["precision_score"] = context.details["precision_score"]
        if "data_quality_metrics" in context.details:
            response["data_quality_metrics"] = context.details["data_quality_metrics"]
        
        # Log the fallback
        logger.info(
            f"Rendered fallback for reason: {context.reason.value}",
            extra={
                "reason": context.reason.value,
                "query": context.original_query,
                "details": context.details
            }
        )
        
        return response
    
    def render_did_you_mean(self, query: str, suggestions: List[str]) -> Dict[str, Any]:
        """
        Render 'Did you mean?' suggestions for ambiguous queries.
        
        Args:
            query: Original query
            suggestions: List of query suggestions
            
        Returns:
            Dictionary containing the suggestions
        """
        return {
            "type": "did_you_mean",
            "original_query": query,
            "message": "Did you mean one of these?",
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
    
    def render_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Render a user-friendly error message.
        
        Args:
            error: Exception that occurred
            query: Original query
            
        Returns:
            Dictionary containing the error message
        """
        return {
            "type": "error",
            "message": "Sorry, something went wrong while processing your request.",
            "suggestion": "Please try again or contact support if the issue persists.",
            "original_query": query,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }

def test_fallback_renderer():
    """Test the fallback renderer."""
    renderer = FallbackRenderer()
    
    # Test low precision fallback
    context = FallbackContext(
        reason=FallbackReason.LOW_PRECISION,
        details={"precision_score": 0.3},
        original_query="Show me sales"
    )
    result = renderer.render_fallback(context)
    print("\n=== LOW PRECISION FALLBACK ===")
    print(f"Message: {result['message']}")
    print(f"Suggestion: {result['suggestion']}")
    print("Examples:")
    for example in result['examples']:
        print(f"- {example}")
    
    # Test missing columns fallback
    context = FallbackContext(
        reason=FallbackReason.MISSING_COLUMNS,
        details={
            "available_columns": [
                "total_sales",
                "gross_profit",
                "region",
                "date"
            ]
        },
        original_query="Show me customer lifetime value"
    )
    result = renderer.render_fallback(context)
    print("\n=== MISSING COLUMNS FALLBACK ===")
    print(f"Message: {result['message']}")
    print(f"Suggestion: {result['suggestion']}")
    print("Available columns:")
    for example in result['examples']:
        print(example)
    
    # Test did you mean
    suggestions = [
        "Show me total sales by region",
        "Show me total sales by product",
        "Show me total sales trend"
    ]
    result = renderer.render_did_you_mean("Show me sales", suggestions)
    print("\n=== DID YOU MEAN ===")
    print(f"Message: {result['message']}")
    print("Suggestions:")
    for suggestion in result['suggestions']:
        print(f"- {suggestion}")

if __name__ == "__main__":
    test_fallback_renderer()