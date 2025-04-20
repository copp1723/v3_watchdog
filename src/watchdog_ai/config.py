"""
Configuration for Watchdog AI.
"""
import os
from typing import Dict, Any

# Make sure you've set OPENAI_API_KEY in your environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Feature flags
ENABLE_DATA_QUALITY_WARNINGS = os.getenv("ENABLE_DATA_QUALITY_WARNINGS", "true").lower() == "true"
ENABLE_SCHEMA_ADAPTATION = os.getenv("ENABLE_SCHEMA_ADAPTATION", "true").lower() == "true"

# Data quality thresholds
QUALITY_THRESHOLD_WARNING = float(os.getenv("QUALITY_THRESHOLD_WARNING", "10.0"))
QUALITY_THRESHOLD_ERROR = float(os.getenv("QUALITY_THRESHOLD_ERROR", "20.0"))
MIN_SAMPLE_SIZE = int(os.getenv("MIN_SAMPLE_SIZE", "30"))

class SessionKeys:
    """Session state keys."""
    CHAT_HISTORY = "chat_history"
    CURRENT_INSIGHT = "current_insight"
    SELECTED_EXAMPLE = "selected_example"
    LAST_QUERY = "last_query"
    QUERY_TEXT = "query_text"
    UPLOADED_DATA = "uploaded_data"
    LAST_TAB = "last_tab"
    LAST_RESULT = "last_result"
    LAST_INTENT = "last_intent"
    INTENT_CACHE = "intent_cache"
    RESULT_CACHE = "result_cache"
    CONVERSATION_STATE = "conversation_state"
    METRICS_HISTORY = "metrics_history"
    ANALYSIS_STATE = "analysis_state"

def get_quality_thresholds() -> Dict[str, Any]:
    """Get current data quality thresholds."""
    return {
        "missing_data": {
            "warning": QUALITY_THRESHOLD_WARNING,
            "error": QUALITY_THRESHOLD_ERROR
        },
        "sample_size": {
            "minimum": MIN_SAMPLE_SIZE
        }
    }

def get_feature_flags() -> Dict[str, bool]:
    """Get current feature flag settings."""
    return {
        "data_quality_warnings": ENABLE_DATA_QUALITY_WARNINGS,
        "schema_adaptation": ENABLE_SCHEMA_ADAPTATION
    }