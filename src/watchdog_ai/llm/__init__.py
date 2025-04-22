"""
Watchdog AI LLM Engine

This module provides an enhanced Language Model interface with comprehensive
analysis capabilities, including pattern detection, metric analysis, and 
structured response handling.

Features:
    - Advanced pattern detection and analysis
    - Comprehensive metric calculation
    - Structured response parsing and validation
    - Configurable engine settings
    - Robust error handling

Example:
    >>> from watchdog_ai.llm import LLMEngine
    >>> engine = LLMEngine()
    >>> response = engine.generate_insight(
    ...     "Analyze sales trends",
    ...     context={'data': sales_data}
    ... )
    >>> print(response['summary'])
    'Sales show strong upward trend with 15% growth'
"""

from .engine import LLMEngine
from .config import APIConfig, SystemPrompts, EngineSettings
from .analysis.patterns import (
    detect_trends,
    analyze_seasonality,
    detect_anomalies,
    analyze_correlations
)
from .analysis.metrics import (
    calculate_metrics,
    calculate_confidence_intervals,
    analyze_period_changes
)
from .parsing import (
    parse_llm_response,
    validate_response,
    format_response
)

__version__ = "1.0.0"

__all__ = [
    # Main engine
    'LLMEngine',
    
    # Configuration
    'APIConfig',
    'SystemPrompts',
    'EngineSettings',
    
    # Pattern analysis
    'detect_trends',
    'analyze_seasonality',
    'detect_anomalies',
    'analyze_correlations',
    
    # Metric analysis
    'calculate_metrics',
    'calculate_confidence_intervals',
    'analyze_period_changes',
    
    # Response parsing
    'parse_llm_response',
    'validate_response',
    'format_response'
]
