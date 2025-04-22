"""
Utility functions for Watchdog AI.
"""
import json
import logging
import re
from ..models import InsightResponse
from .columns import find_metric_column, find_category_column

# Import StatusFormatter if available
try:
    from watchdog_ai.ui.utils.status_formatter import StatusType
    HAS_STATUS_FORMATTER = True
except ImportError:
    HAS_STATUS_FORMATTER = False

def parse_llm_response(raw: str) -> InsightResponse:
    """
    Try to coerce raw JSON from the LLM into our schema.
    On failure, log & return the mock fallback.
    """
    try:
        # First, try to extract JSON if it's embedded in markdown or other text
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw)
        if json_match:
            raw = json_match.group(1)
        
        # Try to find JSON object if it's not properly formatted
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            raw = json_match.group(0)
        
        # Try to parse the JSON
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}\nRaw response: {raw}")
            # Try to fix common JSON formatting issues
            fixed_raw = raw.replace("'", '"')  # Replace single quotes with double quotes
            try:
                payload = json.loads(fixed_raw)
            except json.JSONDecodeError:
                # If still failing, create a basic payload with the raw text as summary
                payload = {
                    "summary": f"Raw LLM response (parsing failed): {raw[:200]}...",
                    "confidence": "low",
                    "is_error": True,
                    "error": f"JSON parsing failed: {str(e)}"
                }
        
        # Validate required fields
        required_fields = ["summary", "confidence"]
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            logging.warning(f"Missing required fields in LLM response: {missing_fields}")
            # Add missing fields with default values
            for field in missing_fields:
                if field == "summary":
                    payload["summary"] = "No summary provided by LLM"
                elif field == "confidence":
                    payload["confidence"] = "low"
        
        # Ensure confidence is one of the allowed values
        if "confidence" in payload and payload["confidence"] not in ["low", "medium", "high"]:
            payload["confidence"] = "low"
            logging.warning(f"Invalid confidence value, defaulting to 'low'")
        
        # Create the InsightResponse object
        return InsightResponse(**payload)
    except Exception as e:
        logging.error(f"LLM response parse error: {e}\nRaw response: {raw}")
        # Return a more informative mock insight without emoji
        fallback_summary = "This insight was generated from fallback due to a formatting error from the LLM."
        
        # Create fallback response
        fallback_response = {
            "summary": fallback_summary,
            "metrics": {"Error": 1},
            "recommendations": [
                "The LLM response could not be parsed correctly",
                "Check the logs for more details about the parsing error"
            ],
            "confidence": "low",
            "is_error": True,
            "error": f"Parsing error: {str(e)}",
            "is_mock": True,
            "status_type": "WARNING"  # For use with StatusFormatter
        }
        
        # Convert dictionary to InsightResponse
        return InsightResponse(**fallback_response)

__all__ = ['find_metric_column', 'find_category_column', 'parse_llm_response']