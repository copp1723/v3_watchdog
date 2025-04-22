"""
Response parsing module for LLM engine.
"""

import json
import re
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def parse_llm_response(content: str) -> Dict[str, Any]:
    """
    Parse and structure the raw LLM response.
    
    Args:
        content: Raw response from LLM
        
    Returns:
        Dictionary containing parsed response
    """
    try:
        # First try to find JSON content within markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
            
        # Try to parse as JSON
        try:
            response = json.loads(content)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract structured content
            response = _extract_structured_content(content)
            
        # Validate and clean the response
        return _clean_response(response)
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return _generate_error_response(str(e))

def _extract_structured_content(text: str) -> Dict[str, Any]:
    """Extract structured information from unstructured text."""
    # Initialize basic structure
    structured = {
        "summary": "",
        "value_insights": [],
        "actionable_flags": [],
        "confidence": "low"
    }
    
    # Try to identify sections
    sections = text.split('\n\n')
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Look for summary (usually first non-empty paragraph)
        if not structured["summary"]:
            structured["summary"] = section
            continue
            
        # Look for insights (often bullet points)
        if section.startswith('•') or section.startswith('-') or section.startswith('*'):
            insights = [line.strip('•- *').strip() for line in section.split('\n')]
            structured["value_insights"].extend(insights)
            continue
            
        # Look for action items
        if 'action' in section.lower() or 'recommend' in section.lower():
            actions = [line.strip('•- *').strip() for line in section.split('\n')]
            structured["actionable_flags"].extend(actions)
            continue
    
    return structured

def _clean_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and validate response structure."""
    cleaned = {
        "summary": str(response.get("summary", "No summary available")),
        "value_insights": [],
        "actionable_flags": [],
        "confidence": "medium",
        "timestamp": datetime.now().isoformat()
    }
    
    # Clean value insights
    insights = response.get("value_insights", [])
    if not isinstance(insights, list):
        insights = [str(insights)]
    cleaned["value_insights"] = [str(insight) for insight in insights if insight]
    
    # Clean actionable flags
    flags = response.get("actionable_flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
    cleaned["actionable_flags"] = [str(flag) for flag in flags if flag]
    
    # Clean confidence
    confidence = str(response.get("confidence", "medium")).lower()
    if confidence not in ["high", "medium", "low"]:
        confidence = "medium"
    cleaned["confidence"] = confidence
    
    # Add any additional fields
    for key, value in response.items():
        if key not in cleaned and value is not None:
            cleaned[key] = value
    
    return cleaned

def _generate_error_response(error: str) -> Dict[str, Any]:
    """Generate a structured error response."""
    return {
        "summary": "Error parsing LLM response",
        "value_insights": [
            "The system encountered an error while parsing the LLM response.",
            f"Error: {error}"
        ],
        "actionable_flags": [
            "Please try regenerating the response"
        ],
        "confidence": "low",
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

