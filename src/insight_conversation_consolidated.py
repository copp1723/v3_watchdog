"""
Module for managing conversation flow with LLM interactions.
Handles conversation history, prompt management, and insight generation.

This is a consolidated version that combines features from all previous implementations.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from scipy import stats
import sentry_sdk

# Import local modules
from src.insight_card_consolidated import InsightOutputFormatter, render_insight_card
from src.insight_flow import PromptGenerator, generate_llm_prompt
# Import template support - comment out for testing
# from src.insight_templates import TemplateManager, InsightTemplate
from src.llm_engine import LLMEngine
from src.insights.intent_manager import intent_manager

# Configure logging
logger = logging.getLogger(__name__)

def _load_system_prompt(filepath="automotive_analyst_prompt.md") -> str:
    """Loads the system prompt from a specified file."""
    try:
        # Try relative path first
        script_dir = os.path.dirname(__file__)
        rel_path = os.path.join(script_dir, '..', filepath)
        if os.path.exists(rel_path):
            filepath = rel_path
        elif not os.path.exists(filepath):
            # Try project root path as fallback
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            root_path = os.path.join(project_root, filepath)
            if os.path.exists(root_path):
                filepath = root_path
            else:
                # If still not found, check one level up from src for prompt_templates
                prompts_dir = os.path.join(project_root, 'prompt_templates')
                template_path = os.path.join(prompts_dir, os.path.basename(filepath))
                if os.path.exists(template_path):
                    filepath = template_path
                else:
                    # Fallback to default if not found anywhere sensible
                    raise FileNotFoundError(f"System prompt not found near src or in project root: {os.path.basename(filepath)}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as e:
        logger.error(f"{e}. Using default fallback system prompt.")
        # Fallback prompt if file is missing
        return (
            "You are a helpful data analyst. Analyze the provided context and user question. "
            "Respond ONLY in valid JSON format: "
            '{"summary": "<summary>", "value_insights": [], "actionable_flags": [], "confidence": "medium"}. '
        )
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return (
            "You are a helpful data analyst. Analyze the provided context and user question. "
            "Respond ONLY in valid JSON format: "
            '{"summary": "<summary>", "value_insights": [], "actionable_flags": [], "confidence": "medium"}. '
        )

def _analyze_data_context(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data analysis for context enrichment.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary of analysis results
    """
    context = {}
    
    try:
        # Basic dataset info
        context["data_shape"] = df.shape
        context["columns"] = df.columns.tolist()
        context["data_types"] = df.dtypes.astype(str).to_dict()
        
        # Find key column types
        date_cols = [col for col in df.columns 
                    if any(term in col.lower() 
                          for term in ['date', 'time', 'month', 'year'])]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze numeric columns
        numeric_analysis = {}
        for col in numeric_cols:
            col_stats = df[col].describe()
            numeric_analysis[col] = {
                "mean": float(col_stats['mean']),
                "std": float(col_stats['std']),
                "min": float(col_stats['min']),
                "max": float(col_stats['max']),
                "quartiles": {
                    "25": float(col_stats['25%']),
                    "50": float(col_stats['50%']),
                    "75": float(col_stats['75%'])
                }
            }
            
            # Check for outliers
            Q1 = col_stats['25%']
            Q3 = col_stats['75%']
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | 
                        (df[col] > (Q3 + 1.5 * IQR))][col]
            numeric_analysis[col]["outliers"] = {
                "count": len(outliers),
                "percentage": len(outliers) / len(df) * 100
            }
            
            # Check distribution
            if len(df[col].dropna()) > 0:
                skewness = stats.skew(df[col].dropna())
                numeric_analysis[col]["distribution"] = {
                    "skewness": float(skewness),
                    "type": "right_skewed" if skewness > 1 else 
                           "left_skewed" if skewness < -1 else "normal"
                }
        
        context["numeric_analysis"] = numeric_analysis
        
        # Analyze categorical columns
        categorical_analysis = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            top_values = value_counts.head(5).to_dict()
            
            categorical_analysis[col] = {
                "unique_values": unique_count,
                "top_values": {str(k): int(v) for k, v in top_values.items()},
                "null_count": int(df[col].isna().sum()),
                "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
            }
        
        context["categorical_analysis"] = categorical_analysis
        
        # Time series analysis if date column exists
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            time_analysis = {}
            for col in numeric_cols:
                if col == date_col:
                    continue
                    
                # Sort by date
                series = df.sort_values(date_col)[[date_col, col]].dropna()
                
                if len(series) >= 2:
                    # Trend analysis
                    x = np.arange(len(series))
                    y = series[col].values
                    slope, _, r_value, p_value, _ = stats.linregress(x, y)
                    
                    time_analysis[col] = {
                        "trend": {
                            "direction": "increasing" if slope > 0 else "decreasing",
                            "strength": abs(r_value),
                            "significant": p_value < 0.05,
                            "p_value": float(p_value)
                        }
                    }
            
            context["time_analysis"] = time_analysis
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            strong_correlations = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.6:
                        strong_correlations.append({
                            "variables": [numeric_cols[i], numeric_cols[j]],
                            "correlation": float(corr),
                            "strength": "strong" if abs(corr) > 0.8 else "moderate"
                        })
            
            context["correlations"] = strong_correlations
        
    except Exception as e:
        logger.error(f"Error in data analysis: {str(e)}")
        context["error"] = str(e)
    
    return context

def _enhance_prompt_with_history(prompt: str, history: List[Dict[str, Any]]) -> str:
    """
    Enhance prompt with conversation history context.
    
    Args:
        prompt: Original prompt
        history: Conversation history
        
    Returns:
        Enhanced prompt
    """
    if not history:
        return prompt
    
    # Extract relevant insights from history
    relevant_insights = []
    for entry in history[-3:]:  # Last 3 exchanges
        if 'response' in entry and 'summary' in entry['response']:
            relevant_insights.append({
                "summary": entry['response']['summary'],
                "metrics": entry['response'].get('metrics', {}),
                "patterns": entry['response'].get('patterns', {})
            })
    
    # Create context section
    context = "\n\nPrevious Analysis Context:"
    for i, insight in enumerate(relevant_insights, 1):
        context += f"\n{i}. {insight['summary']}"
        
        # Add metrics if available
        if insight.get('metrics'):
            context += "\n   Metrics found:"
            for metric, value in insight['metrics'].items():
                if isinstance(value, (int, float)):
                    context += f"\n   - {metric}: {value}"
    
    return prompt + context

class ConversationManager:
    """Manages conversation flow and LLM interactions with enhanced analytics."""
    
    def __init__(self, schema: Optional[Dict[str, str]] = None, use_mock: bool = None):
        """Initialize the conversation manager."""
        # FINAL robust use_mock logic
        env_mock = os.getenv("USE_MOCK", "true").strip().lower() in ["true", "1", "yes"]
        self.use_mock = use_mock if use_mock is not None else env_mock
        self.schema = schema or {}  # Use empty dict if schema is None
        self.formatter = InsightOutputFormatter()
        
        # LLM client settings
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI
        
        logger.debug(f"Initializing ConversationManager with provider: {self.llm_provider}")
        logger.debug(f"Mock mode: {self.use_mock}")
        
        # Initialize LLM engine with API key
        self.llm_engine = LLMEngine(use_mock=self.use_mock, api_key=self.api_key)
        
        # Initialize template manager (commented out for now)
        # self.template_manager = TemplateManager()
        self.template_manager = None
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'regenerate_insight' not in st.session_state:
            st.session_state['regenerate_insight'] = False
        if 'analysis_context' not in st.session_state:
            st.session_state['analysis_context'] = {}
    
    def generate_insight(self, prompt: str, validation_context: Dict[str, Any] = None, add_to_history: bool = True) -> Dict[str, Any]:
        """
        Generate a new insight based on the user prompt and validation context.
        
        Args:
            prompt: The user's prompt
            validation_context: Context including DataFrame and validation info
            add_to_history: Whether to add this interaction to history
            
        Returns:
            The insight response
        """
        try:
            # Input validation
            if not prompt or not isinstance(prompt, str):
                error_msg = "Invalid prompt: Prompt must be a non-empty string"
                logger.error(error_msg)
                return {
                    "summary": "Failed to generate insight",
                    "error": error_msg,
                    "error_type": "input_validation",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True
                }

            logger.info(f"Generating insight for prompt: {prompt[:50]}...")
            
            # Track with Sentry if available
            try:
                sentry_sdk.set_tag("event_type", "insight_generation")
                sentry_sdk.set_tag("prompt_length", len(prompt))
                if validation_context and 'df' in validation_context:
                    sentry_sdk.set_tag("data_rows", len(validation_context['df']))
            except:
                pass  # Sentry not configured, skip
            
            # Get DataFrame from validation context
            df = None
            if validation_context:
                # Check both standard key and enhanced key
                if 'df' in validation_context:
                    df = validation_context['df']
                elif 'validated_data' in validation_context:
                    df = validation_context['validated_data']
            
            if df is None:
                return {
                    "summary": "No data available for analysis",
                    "error": "Missing DataFrame in validation context",
                    "error_type": "missing_data",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True
                }
            
            # Analyze data if available
            if df is not None:
                analysis_context = _analyze_data_context(df)
                if validation_context is None:
                    validation_context = {}
                validation_context.update(analysis_context)
                st.session_state['analysis_context'] = analysis_context
            
            # Try direct calculation first
            response = intent_manager.generate_insight(prompt, df)
            
            # If direct calculation failed or returned fallback, try LLM
            if response.get("is_error") or (not response.get("is_direct_calculation") and not self.use_mock):
                logger.info("Direct calculation failed or returned fallback, trying LLM")
                
                # Enhance prompt with history and templates
                enhanced_prompt = _enhance_prompt_with_history(
                    prompt, 
                    st.session_state.get('conversation_history', [])
                )
                
                # Find applicable templates (commented out for now)
                if self.template_manager:
                    templates = self.template_manager.get_applicable_templates(enhanced_prompt)
                    if templates:
                        template = templates[0]  # Use best matching template
                        enhanced_prompt = self.template_manager.apply_template(
                            template, 
                            enhanced_prompt, 
                            validation_context
                        )
                
                # Generate insight using LLM engine
                llm_response = self.llm_engine.generate_insight(enhanced_prompt, validation_context)
                
                if not llm_response.get("is_error"):
                    response = llm_response
            
            # Add timestamp and format consistently
            response["timestamp"] = datetime.now().isoformat()
            
            # Add to history if requested
            if add_to_history:
                st.session_state.conversation_history.append({
                    "prompt": prompt,
                    "response": response,
                    "timestamp": response["timestamp"]
                })
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating insight: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Report to Sentry if available
            try:
                sentry_sdk.capture_exception(e)
            except:
                pass
            
            return {
                "summary": "Failed to generate insight",
                "error": str(e),
                "error_type": "unknown",
                "timestamp": datetime.now().isoformat(),
                "is_error": True
            }
    
    def regenerate_insight(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Regenerate an insight at a specific index.
        
        Args:
            index: Index in conversation history (-1 for last)
            
        Returns:
            Regenerated insight or None if invalid index
        """
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
    """
    Render conversation history in a consistent format.
    
    Args:
        history: List of conversation entries
        show_buttons: Whether to show interaction buttons
    """
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