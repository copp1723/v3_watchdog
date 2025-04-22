"""
Fallback insight renderer for handling error cases and providing user-friendly explanations.
"""

import enum
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from time import time
from collections import Counter
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

class ErrorCode(enum.Enum):
    """Unified error taxonomy for insight generation failures."""
    
    # Schema and data validation errors
    COLUMN_MISSING = "column_missing"
    INVALID_DATA_TYPE = "invalid_data_type"
    DATA_CONVERSION_ERROR = "data_conversion_error"
    NO_MATCHING_DATA = "no_matching_data"
    
    # LLM and code generation errors
    INVALID_LLM_CODE = "invalid_llm_code"
    CODE_EXECUTION_ERROR = "code_execution_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    
    # Business logic errors
    INVALID_BUSINESS_RULE = "invalid_business_rule"
    INSUFFICIENT_DATA = "insufficient_data"
    AMBIGUOUS_RESULT = "ambiguous_result"
    
    # System errors
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    INTEGRATION_ERROR = "integration_error"

@dataclass
class ErrorContext:
    """Context information for error rendering."""
    error_code: ErrorCode
    error_message: str
    details: Dict[str, Any]
    timestamp: str
    user_query: Optional[str] = None
    affected_columns: Optional[list] = None
    stack_trace: Optional[str] = None

class FallbackRenderer:
    """Renders user-friendly error messages and logs error information."""
    
    def __init__(self):
        """Initialize the fallback renderer."""
        self.error_templates = {
            ErrorCode.COLUMN_MISSING: {
                "title": "Required Data Not Found",
                "message": "We couldn't find the {column_name} column in your data. This information is needed to generate the insight.",
                "action": "Please check your data file and ensure it contains the {column_name} column.",
                "technical_details": "Missing column: {column_name}"
            },
            ErrorCode.INVALID_DATA_TYPE: {
                "title": "Data Type Mismatch",
                "message": "The {column_name} column contains data in an unexpected format.",
                "action": "Please ensure the {column_name} column contains {expected_type} values.",
                "technical_details": "Column {column_name} expected {expected_type}, got {actual_type}"
            },
            ErrorCode.DATA_CONVERSION_ERROR: {
                "title": "Data Conversion Error",
                "message": "We couldn't process some values in the {column_name} column.",
                "action": "Please check the {column_name} column for any unusual or invalid values.",
                "technical_details": "Failed to convert values in {column_name}: {error_details}"
            },
            ErrorCode.NO_MATCHING_DATA: {
                "title": "No Matching Data Found",
                "message": "We couldn't find any data matching your criteria.",
                "action": "Try adjusting your query or checking if the data exists for the specified time period.",
                "technical_details": "No data found matching criteria: {criteria}"
            },
            ErrorCode.INVALID_LLM_CODE: {
                "title": "Code Generation Error",
                "message": "We encountered an issue while generating the analysis code.",
                "action": "Please try rephrasing your question or contact support if the issue persists.",
                "technical_details": "Invalid code generation: {error_details}"
            },
            ErrorCode.CODE_EXECUTION_ERROR: {
                "title": "Analysis Execution Error",
                "message": "We encountered an error while running the analysis.",
                "action": "Please try again or contact support if the issue persists.",
                "technical_details": "Code execution error: {error_details}"
            },
            ErrorCode.MEMORY_ERROR: {
                "title": "Memory Limit Exceeded",
                "message": "The analysis requires more memory than available.",
                "action": "Try breaking down your question into smaller parts.",
                "technical_details": "Memory limit exceeded: {memory_usage}"
            },
            ErrorCode.TIMEOUT_ERROR: {
                "title": "Analysis Timeout",
                "message": "The analysis is taking longer than expected.",
                "action": "Try simplifying your question or breaking it into smaller parts.",
                "technical_details": "Analysis timed out after {timeout_seconds} seconds"
            },
            ErrorCode.INVALID_BUSINESS_RULE: {
                "title": "Invalid Business Rule",
                "message": "The analysis violates a business rule: {rule_name}",
                "action": "Please adjust your query to comply with business rules.",
                "technical_details": "Business rule violation: {rule_details}"
            },
            ErrorCode.INSUFFICIENT_DATA: {
                "title": "Insufficient Data",
                "message": "There isn't enough data to generate a meaningful insight.",
                "action": "Try analyzing a different time period or metric.",
                "technical_details": "Insufficient data points: {data_points}"
            },
            ErrorCode.AMBIGUOUS_RESULT: {
                "title": "Ambiguous Result",
                "message": "The analysis produced unclear or ambiguous results.",
                "action": "Try being more specific in your question.",
                "technical_details": "Ambiguous analysis result: {ambiguity_details}"
            },
            ErrorCode.SYSTEM_ERROR: {
                "title": "System Error",
                "message": "We encountered an unexpected system error.",
                "action": "Please try again later or contact support if the issue persists.",
                "technical_details": "System error: {error_details}"
            },
            ErrorCode.CONFIGURATION_ERROR: {
                "title": "Configuration Error",
                "message": "There's an issue with the system configuration.",
                "action": "Please contact support to resolve this issue.",
                "technical_details": "Configuration error: {config_details}"
            },
            ErrorCode.INTEGRATION_ERROR: {
                "title": "Integration Error",
                "message": "We're having trouble connecting to a required service.",
                "action": "Please try again later or contact support if the issue persists.",
                "technical_details": "Integration error: {integration_details}"
            }
        }
        self._error_counts = Counter()
        self._lock = Lock()
    
    def _track_error(self, error_code: ErrorCode):
        """Track error occurrence for monitoring."""
        with self._lock:
            self._error_counts[error_code] += 1
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get current error statistics."""
        with self._lock:
            return dict(self._error_counts)
    
    def render_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """
        Generate a user-friendly error message and log the error.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            Dictionary containing the rendered error message and metadata
        """
        start_time = time()
        
        # Track the error
        self._track_error(error_context.error_code)
        
        # Get the error template
        template = self.error_templates.get(error_context.error_code)
        if not template:
            template = self.error_templates[ErrorCode.SYSTEM_ERROR]
        
        # Format the error message
        try:
            message = template["message"].format(**error_context.details)
            action = template["action"].format(**error_context.details)
            technical_details = template["technical_details"].format(**error_context.details)
        except KeyError as e:
            logger.error(f"Error formatting message template: {e}")
            message = template["message"]
            action = template["action"]
            technical_details = template["technical_details"]
        
        # Log the error
        self._log_error(error_context, technical_details)
        
        # Calculate render time
        render_time = time() - start_time
        
        # Log performance metric
        logger.info(
            f"Fallback render time: {render_time:.3f}s",
            extra={
                "metric": "fallback_render_time",
                "value": render_time,
                "error_code": error_context.error_code.value
            }
        )
        
        # Return the rendered error
        return {
            "type": "error",
            "title": template["title"],
            "message": message,
            "action": action,
            "error_code": error_context.error_code.value,
            "timestamp": error_context.timestamp,
            "render_time": render_time
        }
    
    def _log_error(self, error_context: ErrorContext, technical_details: str) -> None:
        """
        Log error information for debugging and improvement.
        
        Args:
            error_context: Context information about the error
            technical_details: Formatted technical details about the error
        """
        log_data = {
            "error_code": error_context.error_code.value,
            "error_message": error_context.error_message,
            "technical_details": technical_details,
            "timestamp": error_context.timestamp,
            "user_query": error_context.user_query,
            "affected_columns": error_context.affected_columns,
            "stack_trace": error_context.stack_trace
        }
        
        logger.error(
            f"Insight generation error: {error_context.error_code.value}",
            extra={"error_data": json.dumps(log_data)}
        ) 