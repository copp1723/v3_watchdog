"""
Enhanced conversation manager for more intelligent insight generation and follow-ups.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

from src.insight_card_improved import EnhancedInsightOutputFormatter, render_enhanced_insight_card
from src.insight_flow import PromptGenerator, generate_llm_prompt
from src.insight_flow_enhanced import EnhancedPromptGenerator, enhanced_generate_llm_prompt
from src.insight_templates import TemplateManager, InsightTemplate
from src.llm_engine import LLMEngine

class ConversationManager:
    """Manages conversation flow and LLM interactions with enhanced analytics."""
    
    def __init__(self, schema: Optional[Dict[str, str]] = None, use_mock: bool = None):
        """Initialize the conversation manager."""
        # Initialize LLM engine
        env_mock = os.getenv("USE_MOCK", "true").strip().lower() in ["true", "1", "yes"]
        self.use_mock = use_mock if use_mock is not None else env_mock
        self.schema = schema or {}
        self.formatter = EnhancedInsightOutputFormatter()
        
        # Initialize LLM engine with API key
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.llm_engine = LLMEngine(use_mock=self.use_mock, api_key=self.api_key)
        
        # Initialize template manager
        self.template_manager = TemplateManager()
        
        # Initialize conversation state
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'regenerate_insight' not in st.session_state:
            st.session_state['regenerate_insight'] = False
        if 'analysis_context' not in st.session_state:
            st.session_state['analysis_context'] = {}
    
    def _analyze_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
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
                stats = df[col].describe()
                numeric_analysis[col] = {
                    "mean": float(stats['mean']),
                    "std": float(stats['std']),
                    "min": float(stats['min']),
                    "max": float(stats['max']),
                    "quartiles": {
                        "25": float(stats['25%']),
                        "50": float(stats['50%']),
                        "75": float(stats['75%'])
                    }
                }
                
                # Check for outliers
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | 
                            (df[col] > (Q3 + 1.5 * IQR))][col]
                numeric_analysis[col]["outliers"] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100
                }
                
                # Check distribution
                if len(df[col].dropna()) > 0:
                    skewness = stats.skew()
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
                    df[date_col] = pd.to_datetime(df[date_col])
                
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
                        
                        # Seasonality check if enough data
                        if len(series) >= 12:
                            autocorr = pd.Series(y).autocorr(lag=12)
                            time_analysis[col]["seasonality"] = {
                                "autocorrelation": float(autocorr),
                                "has_seasonality": abs(autocorr) > 0.6
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
            
            # Store in session state
            st.session_state['analysis_context'] = context
            
        except Exception as e:
            print(f"Error in data analysis: {str(e)}")
            context["error"] = str(e)
        
        return context
    
    def _enhance_prompt_with_history(self, prompt: str, history: List[Dict[str, Any]]) -> str:
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
            
            # Add patterns if available
            if insight.get('patterns'):
                context += "\n   Patterns detected:"
                pattern = insight['patterns']
                if isinstance(pattern, dict):
                    context += f"\n   - {pattern.get('type', 'unknown')} pattern in {pattern.get('description', '')}"
        
        return prompt + context
    
    def generate_insight(self, prompt: str, validation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a new insight with enhanced context and analysis.
        
        Args:
            prompt: User's prompt
            validation_context: Optional validation context
            
        Returns:
            Generated insight
        """
        try:
            print(f"[DEBUG] Generating insight for prompt: {prompt}")
            
            # Analyze data if available
            if validation_context and 'validated_data' in validation_context:
                df = validation_context['validated_data']
                analysis_context = self._analyze_data_context(df)
                validation_context.update(analysis_context)
            
            # Enhance prompt with history
            enhanced_prompt = self._enhance_prompt_with_history(
                prompt, 
                st.session_state.get('conversation_history', [])
            )
            
            # Find applicable templates
            templates = self.template_manager.get_applicable_templates(enhanced_prompt)
            if templates:
                template = templates[0]  # Use best matching template
                enhanced_prompt = self.template_manager.apply_template(
                    template, 
                    enhanced_prompt, 
                    validation_context
                )
            
            # Generate insight using LLM engine
            response = self.llm_engine.generate_insight(enhanced_prompt, validation_context)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating insight: {str(e)}"
            print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
            return {
                "summary": "Failed to generate insight",
                "error": str(e),
                "value_insights": [],
                "actionable_flags": ["Please check the logs for details and try again."],
                "confidence": "low",
                "timestamp": datetime.now().isoformat()
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
    """Render conversation history with enhanced formatting."""
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
            render_enhanced_insight_card(
                entry['response'],
                index=len(history) - idx - 1,
                show_buttons=show_buttons
            )
            
            # Add separator
            st.markdown("---")
