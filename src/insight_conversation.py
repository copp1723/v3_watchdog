"""
Conversation manager for insight generation and follow-ups.
Implements LLM-driven intent processing system.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pydantic import ValidationError

from .insights.models import IntentSchema, InsightResponse  # Fixed import path
from .llm_engine import LLMEngine
from .insight_card import render_insight_card
from .utils.columns import find_metric_column, find_category_column

# Import MetricType as Metric from any appropriate module that defines it
# We'll use exec_schema_profiles.py since it contains a MetricType definition
from .exec_schema_profiles import MetricType as Metric

# Configure logging
logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation flow and LLM interactions."""
    
    def __init__(self, use_mock: bool = None):
        """Initialize the conversation manager."""
        # Initialize LLM engine
        env_mock = os.getenv("USE_MOCK", "true").strip().lower() in ["true", "1", "yes"]
        self.use_mock = use_mock if use_mock is not None else env_mock
        
        # Initialize LLM engine with API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_engine = LLMEngine(use_mock=self.use_mock, api_key=self.api_key)
        
        # Initialize conversation state
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
    
    def _generate_prompt(self, query: str, columns: List[str]) -> str:
        """Generate the prompt for intent detection."""
        with open("src/insights/prompts/intent_detection.tpl", "r") as f:
            template = f.read()
        
        return template.format(
            available_columns="\n".join(f"- {col}" for col in columns),
            query=query
        )
    
    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a user query using LLM-driven intent detection.
        
        Args:
            query: User's question
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Generate prompt with available columns
            prompt = self._generate_prompt(query, df.columns.tolist())
            
            # Get LLM response
            response = self.llm_engine.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse and validate response
            try:
                intent_data = IntentSchema.parse_raw(response.choices[0].message.content)
                logger.debug(f"Parsed intent: {intent_data.dict()}")
                return self._process_intent(intent_data, df)
            except ValidationError as e:
                logger.error(f"Intent validation error: {str(e)}")
                return self._generate_error_response(
                    "I couldn't understand how to analyze that. Could you rephrase?"
                )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _process_intent(self, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
        """Process the validated intent schema."""
        try:
            if intent.intent == "groupby":
                return self._handle_groupby(intent, df)
            elif intent.intent == "metric":
                return self._handle_metric(intent, df)
            elif intent.intent == "trend":
                return self._handle_trend(intent, df)
            elif intent.intent == "comparison":
                return self._handle_comparison(intent, df)
            else:
                return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"Error processing intent: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _handle_groupby(self, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle groupby intent type."""
        try:
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.filters:
                filtered_df = filtered_df[
                    filtered_df[filter.column].apply(
                        lambda x: eval(f"x {filter.operator} {filter.value}")
                    )
                ]
            
            # Group and aggregate
            aggs = {metric.name: metric.aggregation for metric in intent.metrics}
            dimensions = [dim.name for dim in intent.dimensions]
            
            result = filtered_df.groupby(dimensions).agg(aggs).reset_index()
            
            # Sort if specified
            if intent.sort:
                result = result.sort_values(
                    intent.sort.column,
                    ascending=intent.sort.direction == "asc"
                )
            
            # Apply limit
            if intent.limit:
                result = result.head(intent.limit)
            
            # Get primary metric and dimension
            metric = intent.metrics[0]
            dimension = intent.dimensions[0]
            
            # Get top result
            top_row = result.iloc[0]
            
            return {
                "summary": f"{top_row[dimension.name]} leads with {self._format_value(top_row[metric.name], metric)} {metric.name}s",
                "metrics": {
                    "top_performer": top_row[dimension.name],
                    "value": self._format_value(top_row[metric.name], metric),
                    "context": f"Out of {self._format_value(result[metric.name].sum(), metric)} total"
                },
                "breakdown": result.to_dict("records"),
                "chart_data": self._format_chart_data(result, dimension.name, metric.name),
                "recommendations": self._generate_recommendations(result, intent),
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"Error in groupby handler: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _handle_metric(self, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle metric intent type."""
        try:
            # Get primary metric
            metric = intent.metrics[0]
            
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.filters:
                filtered_df = filtered_df[
                    filtered_df[filter.column].apply(
                        lambda x: eval(f"x {filter.operator} {filter.value}")
                    )
                ]
            
            # Calculate metric
            if metric.aggregation == "count":
                value = len(filtered_df)
            else:
                value = getattr(filtered_df[metric.name], metric.aggregation)()
            
            # Format summary based on metric type
            if "gross" in metric.name.lower() or "profit" in metric.name.lower():
                summary = f"Total {metric.name} is ${value:,.2f}"
            else:
                summary = f"Total {metric.name} is {value:,}"
            
            return {
                "summary": summary,
                "metrics": {
                    "value": self._format_value(value, metric),
                    "context": f"Calculated from {len(filtered_df):,} records"
                },
                "breakdown": [],
                "recommendations": self._generate_recommendations_for_metric(value, metric),
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"Error in metric handler: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _handle_trend(self, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle trend intent type."""
        try:
            # Get metric and time dimension
            metric = intent.metrics[0]
            time_dim = next(dim for dim in intent.dimensions if dim.type == "time")
            
            # Convert time column to datetime
            df[time_dim.name] = pd.to_datetime(df[time_dim.name])
            
            # Group by time and aggregate
            result = df.groupby(pd.Grouper(key=time_dim.name, freq='M'))[metric.name].agg(metric.aggregation).reset_index()
            
            # Calculate trend
            values = result[metric.name].values
            if len(values) >= 2:
                change = ((values[-1] - values[0]) / values[0]) * 100
                direction = "increased" if change > 0 else "decreased"
                
                summary = (
                    f"{metric.name} has {direction} by {abs(change):.1f}% "
                    f"from {self._format_value(values[0], metric)} to {self._format_value(values[-1], metric)}"
                )
            else:
                summary = f"Not enough data to calculate trend for {metric.name}"
            
            return {
                "summary": summary,
                "metrics": {
                    "start_value": self._format_value(values[0], metric),
                    "end_value": self._format_value(values[-1], metric),
                    "change_percentage": f"{change:.1f}%" if len(values) >= 2 else "N/A"
                },
                "breakdown": result.to_dict("records"),
                "chart_data": {
                    "data": result.to_dict("records"),
                    "encoding": {
                        "x": {"field": time_dim.name, "type": "temporal"},
                        "y": {"field": metric.name, "type": "quantitative"},
                        "tooltip": [time_dim.name, metric.name]
                    }
                },
                "recommendations": self._generate_trend_recommendations(change if len(values) >= 2 else 0),
                "confidence": "high" if len(values) >= 2 else "medium"
            }
            
        except Exception as e:
            logger.error(f"Error in trend handler: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _format_value(self, value: Any, metric: Metric) -> str:
        """Format a value based on the metric type."""
        if "gross" in metric.name.lower() or "profit" in metric.name.lower():
            return f"${value:,.2f}"
        if isinstance(value, (int, float)):
            return f"{value:,}"
        return str(value)
    
    def _format_chart_data(self, df: pd.DataFrame, dimension: str, metric: str) -> Dict[str, Any]:
        """Format data for visualization."""
        return {
            "data": df.to_dict("records"),
            "encoding": {
                "x": {"field": dimension, "type": "nominal"},
                "y": {"field": metric, "type": "quantitative"},
                "tooltip": [dimension, metric]
            }
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, intent: IntentSchema) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Get primary metric
        metric = intent.metrics[0]
        
        # Add metric-specific recommendations
        if "gross" in metric.name.lower():
            recommendations.append({
                "action": "Review pricing strategy for top performers",
                "priority": "High",
                "impact_estimate": "Could increase gross profit by 10%"
            })
        elif metric.aggregation == "count":
            recommendations.append({
                "action": "Analyze successful patterns from top performers",
                "priority": "Medium",
                "impact_estimate": "Could improve conversion rates"
            })
        
        return recommendations
    
    def _generate_recommendations_for_metric(self, value: float, metric: Metric) -> List[Dict[str, Any]]:
        """Generate recommendations based on metric value."""
        recommendations = []
        
        if "gross" in metric.name.lower() or "profit" in metric.name.lower():
            if value < 0:
                recommendations.append({
                    "action": "Review pricing strategy and costs",
                    "priority": "High",
                    "impact_estimate": "Could turn losses into profits"
                })
            else:
                recommendations.append({
                    "action": "Analyze top performing deals for insights",
                    "priority": "Medium",
                    "impact_estimate": "Could increase profits by 10%"
                })
        elif metric.aggregation == "count":
            recommendations.append({
                "action": "Set target to increase volume by 20%",
                "priority": "Medium",
                "impact_estimate": "Could improve overall performance"
            })
        
        return recommendations
    
    def _generate_trend_recommendations(self, change_percentage: float) -> List[Dict[str, Any]]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if change_percentage < -10:
            recommendations.append({
                "action": "Investigate cause of significant decline",
                "priority": "High",
                "impact_estimate": "Stop negative trend"
            })
        elif change_percentage < 0:
            recommendations.append({
                "action": "Monitor trend and develop improvement plan",
                "priority": "Medium",
                "impact_estimate": "Reverse negative trend"
            })
        elif change_percentage > 10:
            recommendations.append({
                "action": "Analyze success factors driving growth",
                "priority": "Medium",
                "impact_estimate": "Maintain positive momentum"
            })
        else:
            recommendations.append({
                "action": "Set growth targets and action plan",
                "priority": "Medium",
                "impact_estimate": "Accelerate growth"
            })
        
        return recommendations
    
    def _generate_error_response(self, error: str) -> Dict[str, Any]:
        """Generate an error response."""
        return {
            "summary": f"⚠️ {error}",
            "metrics": {},
            "breakdown": [],
            "recommendations": [],
            "confidence": "low",
            "error": error
        }
    
    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate a fallback response."""
        return {
            "summary": "I'm not sure how to analyze that. Could you rephrase your question?",
            "metrics": {},
            "breakdown": [],
            "recommendations": [
                {
                    "action": "Try asking about specific metrics like 'gross profit' or 'sales count'",
                    "priority": "Low",
                    "impact_estimate": "N/A"
                }
            ],
            "confidence": "low"
        }

    def regenerate_insight(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """Regenerate an insight at a specific index."""
        history = st.session_state.get('conversation_history', [])
        if not history or abs(index) > len(history):
            return None
        
        # Get the original prompt
        entry = history[index]
        prompt = entry['prompt']
        
        # Remove the entry we're regenerating
        history.pop(index)
        
        # Generate new insight
        return self.generate_insight(prompt)

def render_conversation_history(history: List[Dict[str, Any]], show_buttons: bool = True) -> None:
    """Render conversation history."""
    if not history:
        st.info("No conversation history yet. Start by entering a prompt!")
        return
    
    # Render insights in reverse chronological order
    for idx, entry in enumerate(reversed(history)):
        with st.container():
            # Add timestamp if available
            if 'timestamp' in entry:
                st.caption(f"Generated at {entry['timestamp']}")
            
            # Show the prompt
            st.markdown(f"**🤔 Question:** {entry['prompt']}")
            
            # Render the insight card
            render_insight_card(
                entry['response'],
                index=len(history) - idx - 1,
                show_buttons=show_buttons
            )
            
            # Add separator
            st.markdown("---")