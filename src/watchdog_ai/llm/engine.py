"""
Core LLM engine implementation.
Coordinates analysis, parsing, and configuration components.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import pandas as pd

from .analysis.patterns import (
    detect_trends, analyze_seasonality, detect_anomalies, analyze_correlations
)
from .analysis.metrics import (
    calculate_metrics, calculate_confidence_intervals, analyze_period_changes
)
from .parsing import parse_llm_response, validate_response, format_response
from .config import APIConfig, SystemPrompts, EngineSettings

logger = logging.getLogger(__name__)

class LLMEngine:
    """Enhanced LLM engine with comprehensive analysis capabilities."""
    
    def __init__(self, settings: Optional[EngineSettings] = None):
        """
        Initialize the LLM engine.
        
        Args:
            settings: Optional engine settings
        """
        self.settings = settings or EngineSettings()
        self.api_config = self.settings.api_config
        self.system_prompts = self.settings.system_prompts
        
        # Initialize API client
        try:
            from openai import OpenAI
            self.client = OpenAI(**self.api_config.get_client_config())
        except ImportError:
            logger.error("OpenAI package not installed")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.client = None
    
    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an insight based on the prompt and context.
        
        Args:
            prompt: User's prompt
            context: Optional context dictionary containing DataFrame or other data
            
        Returns:
            Dictionary containing the generated insight
        """
        try:
            # Enhance prompt with data analysis if context provided
            if context and isinstance(context.get('data'), pd.DataFrame):
                enhanced_prompt = self._enhance_prompt_with_analysis(prompt, context['data'])
            else:
                enhanced_prompt = prompt
            
            # Call LLM
            response = self._call_llm(enhanced_prompt)
            if not response:
                return self._generate_error_response("Failed to get LLM response")
            
            # Parse and validate response
            parsed = parse_llm_response(response)
            is_valid, issues, validated = validate_response(parsed)
            
            if not is_valid:
                logger.warning(f"Response validation issues: {issues}")
                return self._generate_error_response(
                    "Response validation failed",
                    details=issues
                )
            
            # Format response
            return format_response(validated)
            
        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _enhance_prompt_with_analysis(self, prompt: str, df: pd.DataFrame) -> str:
        """Enhance prompt with data analysis results."""
        try:
            analysis_results = {
                "patterns": self._analyze_patterns(df),
                "metrics": self._analyze_metrics(df, prompt)
            }
            
            # Format analysis results for prompt
            analysis_text = (
                "Based on data analysis:\n"
                f"1. Patterns detected: {len(analysis_results['patterns'])}\n"
                f"2. Metrics analyzed: {len(analysis_results['metrics'])}\n\n"
                "Please consider these findings in your response:\n"
            )
            
            # Add pattern findings
            if analysis_results["patterns"]:
                analysis_text += "\nPatterns:\n"
                for pattern in analysis_results["patterns"]:
                    analysis_text += f"- {pattern['description']}\n"
            
            # Add metric findings
            if analysis_results["metrics"]:
                analysis_text += "\nMetrics:\n"
                for metric, details in analysis_results["metrics"].items():
                    analysis_text += f"- {metric}: {details['summary']}\n"
            
            return f"{prompt}\n\n{analysis_text}"
            
        except Exception as e:
            logger.warning(f"Error enhancing prompt with analysis: {str(e)}")
            return prompt
    
    def _analyze_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data patterns."""
        patterns = []
        
        try:
            # Find time-based columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if not date_cols.empty:
                date_col = date_cols[0]
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                for col in numeric_cols:
                    series = df[col]
                    
                    # Detect trends
                    trend = detect_trends(series)
                    if trend.get("has_trend"):
                        patterns.append({
                            "type": "trend",
                            "metric": col,
                            "description": (
                                f"{col} shows {trend['strength']} {trend['direction']} "
                                f"trend (confidence: {trend['confidence']:.2f})"
                            )
                        })
                    
                    # Analyze seasonality
                    seasonality = analyze_seasonality(series)
                    if seasonality.get("has_seasonality"):
                        patterns.append({
                            "type": "seasonality",
                            "metric": col,
                            "description": (
                                f"{col} shows {seasonality['strength']} seasonal pattern "
                                f"with period {seasonality['period']}"
                            )
                        })
                    
                    # Detect anomalies
                    anomalies = detect_anomalies(series)
                    if anomalies.get("has_anomalies"):
                        patterns.append({
                            "type": "anomaly",
                            "metric": col,
                            "description": (
                                f"{col} has {len(anomalies['indices'])} anomalies "
                                f"({anomalies['percentage']:.1f}% of data)"
                            )
                        })
            
            # Analyze correlations
            correlations = analyze_correlations(df)
            if correlations.get("has_correlations"):
                for corr in correlations["correlations"]:
                    if corr["is_significant"]:
                        patterns.append({
                            "type": "correlation",
                            "metrics": corr["variables"],
                            "description": (
                                f"{corr['variables'][0]} and {corr['variables'][1]} show "
                                f"{corr['strength']} correlation ({corr['correlation']:.2f})"
                            )
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return []
    
    def _analyze_metrics(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze metrics based on query."""
        try:
            # Extract query terms
            terms = query.lower().split()
            
            # Calculate relevant metrics
            metrics = calculate_metrics(df, terms)
            
            if not metrics.get("has_metrics"):
                return {}
            
            results = {}
            for metric_type, metric_data in metrics["metrics"].items():
                for column, values in metric_data.items():
                    # Calculate confidence intervals
                    confidence = calculate_confidence_intervals(df[column])
                    
                    # Analyze period changes
                    changes = analyze_period_changes(df, column)
                    
                    results[column] = {
                        "current_value": values["current"],
                        "summary": (
                            f"Currently at {values['current']:.2f} "
                            f"(±{confidence.get('margin_of_error', 0):.2f})"
                        ),
                        "confidence": confidence,
                        "changes": changes
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing metrics: {str(e)}")
            return {}
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with proper error handling."""
        if not self.client:
            logger.error("LLM client not initialized")
            return None
            
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompts.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                **self.api_config.get_completion_config()
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return None
    
    def _generate_error_response(self, 
                               error: str,
                               details: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a standardized error response."""
        return {
            "summary": f"Error: {error}",
            "value_insights": [
                "The system encountered an error while processing the request.",
                f"Error details: {error}"
            ] + (details or []),
            "actionable_flags": [
                "Please check the error details and try again",
                "If the error persists, contact support"
            ],
            "confidence": "low",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

