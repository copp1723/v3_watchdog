"""
Base insight classes for Watchdog AI.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import traceback

class InsightBase(ABC):
    """Abstract base class for all insights."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary representation."""
        pass
        
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightBase':
        """Create insight from dictionary representation."""
        pass
        
    @abstractmethod
    def get_summary(self) -> str:
        """Get the insight summary text."""
        pass
        
    @abstractmethod
    def get_confidence(self) -> str:
        """Get the confidence level (high, medium, low)."""
        pass
        
    @property
    @abstractmethod
    def is_error(self) -> bool:
        """Check if this insight represents an error."""
        pass

class InsightFormatter(ABC):
    """Abstract base class for insight formatters."""
    
    @abstractmethod
    def format_response(self, response_text: str) -> Dict[str, Any]:
        """
        Format the raw response text into a structured insight dict.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            A structured insight dictionary
        """
        pass

class DefaultInsightFormatter(InsightFormatter):
    """Default implementation of InsightFormatter."""
    
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
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from markdown
                response = self._extract_json_from_text(response_text)
                
            # Validate and fill in missing fields
            return self._validate_and_complete(response)
            
        except Exception as e:
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

