"""
Conversation management system for insight generation and follow-ups.

This module provides a unified conversation management interface for:
- Processing natural language queries
- Managing conversation history
- Generating structured insights
- Handling follow-up questions
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from watchdog_ai.core.config import get_logger

# Import internal modules
from .metadata import InsightMetadata, extract_metadata
from .card import InsightCard
from ..config.secure import config

logger = get_logger(__name__)

class ConversationManager:
    """
    Manages conversation flow and LLM interactions for insights.
    
    This class provides a unified interface for:
    - Processing natural language queries
    - Generating insights based on data context
    - Managing conversation history
    - Handling follow-up questions
    """
    
    def __init__(self, use_mock: bool = None):
        """
        Initialize the conversation manager.
        
        Args:
            use_mock: Whether to use mock data (default: use environment setting)
        """
        # Initialize configuration from environment
        env_mock = config.get_bool('USE_MOCK', True)
        self.use_mock = use_mock if use_mock is not None else env_mock
        
        # Get API key from config
        self.api_key = config.get_secret('OPENAI_API_KEY')
        
        # Initialize LLM engine
        self._initialize_llm_engine()
        
        # Initialize conversation state
        self._initialize_state()
        
        logger.info(
            "Initialized ConversationManager",
            extra={
                "component": "conversation_manager",
                "use_mock": self.use_mock
            }
        )
    
    def _initialize_llm_engine(self):
        """Initialize the LLM engine for conversation processing."""
        try:
            from watchdog_ai.llm.llm_engine import LLMEngine
            self.llm_engine = LLMEngine(use_mock=self.use_mock, api_key=self.api_key)
        except ImportError:
            logger.warning("Could not import LLMEngine, using mock implementation")
            # Create a mock LLM engine class
            class MockLLMEngine:
                def __init__(self, *args, **kwargs):
                    self.chat = type('obj', (object,), {
                        'completions': type('obj', (object,), {
                            'create': lambda **kwargs: type('obj', (object,), {
                                'choices': [type('obj', (object,), {
                                    'message': type('obj', (object,), {
                                        'content': '{"intent": "metric", "metrics": [{"name": "sales", "aggregation": "sum"}], "dimensions": [], "filters": []}'
                                    })
                                })]
                            })
                        })
                    })
            self.llm_engine = MockLLMEngine()
    
    def _initialize_state(self):
        """Initialize conversation state in Streamlit session."""
        # Initialize conversation history in session state if not exists
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'validation_context' not in st.session_state:
            st.session_state['validation_context'] = {}
    
    def generate_insight(self, query: str, validation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an insight based on the user query and data context.
        
        Args:
            query: User's question or prompt
            validation_context: Optional context with data validation information
            
        Returns:
            Structured insight response
        """
        try:
            logger.info(f"Generating insight for query: {query}")
            
            # Save the prompt in session state
            st.session_state['current_prompt'] = query
            
            # Get data from validation context
            df = None
            if validation_context and 'validated_data' in validation_context:
                df = validation_context.get('validated_data')
            
            # Process the query and get response
            if df is not None:
                # Use data-aware processing if DataFrame is available
                response = self.process_query(query, df)
            else:
                # Fallback to simple processing without data context
                response = self._generate_basic_insight(query)
            
            # Add timestamp
            timestamp = datetime.now().isoformat()
            
            # Add to conversation history
            st.session_state['conversation_history'].append({
                'prompt': query,
                'response': response,
                'timestamp': timestamp
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def regenerate_insight(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Regenerate an insight at a specific index.
        
        Args:
            index: Index of the insight to regenerate (default: last one)
            
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
    
    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a data-aware query using intent detection.
        
        Args:
            query: User's question
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Generate prompt with available columns
            prompt = self._generate_prompt(query, df.columns.tolist())
            
            # Get LLM response
            response = self.llm_engine.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse and validate response
            try:
                # Try to parse as JSON first
                response_text = response.choices[0].message.content
                intent_data = json.loads(response_text)
                return self._process_intent(intent_data, df)
            except Exception as e:
                logger.error(f"Intent parsing error: {str(e)}")
                return self._generate_error_response(
                    "I couldn't understand how to analyze that. Could you rephrase?"
                )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _generate_prompt(self, query: str, columns: List[str]) -> str:
        """
        Generate the prompt for intent detection.
        
        Args:
            query: User's question
            columns: List of available data columns
            
        Returns:
            Formatted prompt for LLM
        """
        # Look for the template file
        template_paths = [
            "src/insights/prompts/intent_detection.tpl",
            "src/watchdog_ai/insights/prompts/intent_detection.tpl",
            "prompt_templates/intent_detection.tpl"
        ]
        
        template = None
        for path in template_paths:
            try:
                with open(path, "r") as f:
                    template = f.read()
                    break
            except FileNotFoundError:
                continue
        
        # Fallback template if none found
        if template is None:
            template = """
            You are a data analyst assistant.
            Given the user query and available columns in a dataset, determine the most appropriate analysis to perform.
            
            Available columns:
            {available_columns}
            
            User query: {query}
            
            Return your analysis plan as a JSON object with the following structure:
            {
                "intent": "groupby | metric | trend | comparison",
                "metrics": [{"name": column, "aggregation": "sum | avg | count | min | max"}],
                "dimensions": [{"name": column, "type": "category | time | number"}],
                "filters": [{"column": column, "operator": "==, >, <, >=, <=", "value": value}],
                "sort": {"column": column, "direction": "asc | desc"},
                "limit": number
            }
            """
        
        # Format the template
        return template.format(
            available_columns="\n".join(f"- {col}" for col in columns),
            query=query
        )
    
    def _process_intent(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the intent and generate appropriate insight.
        
        Args:
            intent: Intent data extracted from LLM response
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Determine the intent type
            intent_type = intent.get('intent', 'unknown')
            
            # Process based on intent type
            if intent_type == "groupby":
                return self._handle_groupby(intent, df)
            elif intent_type == "metric":
                return self._handle_metric(intent, df)
            elif intent_type == "trend":
                return self._handle_trend(intent, df)
            elif intent_type == "comparison":
                return self._handle_comparison(intent, df)
            else:
                return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"Error processing intent: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _handle_groupby(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Handle groupby intent type.
        
        Args:
            intent: Intent data with metrics, dimensions, etc.
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= val]
                except Exception as filter_error:
                    logger.warning(f"Error applying filter: {str(filter_error)}")
            
            # Extract metrics and dimensions
            metrics = intent.get('metrics', [])
            dimensions = intent.get('dimensions', [])
            
            if not metrics or not dimensions:
                return self._generate_error_response("Missing metrics or dimensions for groupby")
            
            # Build aggregation dictionary
            aggs = {}
            for metric in metrics:
                aggs[metric['name']] = metric['aggregation']
            
            # Get dimension columns
            dimension_cols = [dim['name'] for dim in dimensions]
            
            # Group and aggregate
            result = filtered_df.groupby(dimension_cols).agg(aggs).reset_index()
            
            # Sort if specified
            if 'sort' in intent and intent['sort']:
                sort_col = intent['sort']['column']
                ascending = intent['sort']['direction'] == "asc"
                result = result.sort_values(sort_col, ascending=ascending)
            
            # Apply limit
            if 'limit' in intent and intent['limit']:
                result = result.head(intent['limit'])
            
            # Get primary metric and dimension
            primary_metric = metrics[0]
            primary_dimension = dimensions[0]
            
            # Get top result
            if not result.empty:
                top_row = result.iloc[0]
                
                # Format response
                return {
                    "summary": f"{top_row[primary_dimension['name']]} leads with {self._format_value(top_row[primary_metric['name']], primary_metric)} {primary_metric['name']}",
                    "value_insights": [
                        f"Top {len(result)} categories by {primary_metric['name']}",
                        f"Total {primary_metric['name']}: {self._format_value(result[primary_metric['name']].sum(), primary_metric)}"
                    ],
                    "actionable_flags": self._generate_recommendations(result, intent),
                    "metrics": {
                        "top_performer": top_row[primary_dimension['name']],
                        "value": self._format_value(top_row[primary_metric['name']], primary_metric),
                        "total": self._format_value(result[primary_metric['name']].sum(), primary_metric)
                    },
                    "chart_data": self._format_chart_data(result, primary_dimension['name'], primary_metric['name']),
                    "confidence": "high"
                }
            else:
                return {
                    "summary": "No data found matching the criteria",
                    "value_insights": [],
                    "actionable_flags": [
                        "Consider broadening your search criteria"
                    ],
                    "confidence": "medium"
                }
            
        except Exception as e:
            logger.error(f"Error in groupby handler: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error analyzing data: {str(e)}")
    
    def _handle_metric(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Handle metric intent type.
        
        Args:
            intent: Intent data with metrics, filters, etc.
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Get primary metric
            metrics = intent.get('metrics', [])
            if not metrics:
                return self._generate_error_response("No metrics specified")
                
            metric = metrics[0]
            
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                

"""
Conversation management system for insight generation and follow-ups.

This module provides a unified conversation management interface for:
- Processing natural language queries
- Managing conversation history
- Generating structured insights
- Handling follow-up questions
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from watchdog_ai.core.config import get_logger

# Import internal modules
from .metadata import InsightMetadata, extract_metadata
from .card import InsightCard
from ..config.secure import config

logger = get_logger(__name__)

class ConversationManager:
    """
    Manages conversation flow and LLM interactions for insights.
    
    This class provides a unified interface for:
    - Processing natural language queries
    - Generating insights based on data context
    - Managing conversation history
    - Handling follow-up questions
    """
    
    def __init__(self, use_mock: bool = None):
        """
        Initialize the conversation manager.
        
        Args:
            use_mock: Whether to use mock data (default: use environment setting)
        """
        # Initialize configuration from environment
        env_mock = config.get_bool('USE_MOCK', True)
        self.use_mock = use_mock if use_mock is not None else env_mock
        
        # Get API key from config
        self.api_key = config.get_secret('OPENAI_API_KEY')
    def generate_insight(self, query: str, validation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an insight based on the user query and data context.
        
        Args:
            query: User's question or prompt
            validation_context: Optional context with data validation information
            
        Returns:
            Structured insight response
        """
        try:
            logger.info(f"Generating insight for query: {query}")
            
            # Save the prompt in session state
            st.session_state['current_prompt'] = query
            
            # Get data from validation context
            df = None
            if validation_context and 'validated_data' in validation_context:
                df = validation_context.get('validated_data')
            
            # Process the query and get response
            if df is not None:
                # Use data-aware processing if DataFrame is available
                response = self.process_query(query, df)
            else:
                # Fallback to simple processing without data context
                response = self._generate_basic_insight(query)
            
            # Add timestamp
            timestamp = datetime.now().isoformat()
            
            # Add to conversation history
            st.session_state['conversation_history'].append({
                'prompt': query,
                'response': response,
                'timestamp': timestamp
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def regenerate_insight(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Regenerate an insight at a specific index.
        
        Args:
            index: Index of the insight to regenerate (default: last one)
            
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
def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a data-aware query using intent detection.
        
        Args:
            query: User's question
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Generate prompt with available columns
            prompt = self._generate_prompt(query, df.columns.tolist())
            
            # Get LLM response
            response = self.llm_engine.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse and validate response
            try:
                # Try to parse as JSON first
                response_text = response.choices[0].message.content
                intent_data = json.loads(response_text)
                return self._process_intent(intent_data, df)
            except Exception as e:
                logger.error(f"Intent parsing error: {str(e)}")
                return self._generate_error_response(
                    "I couldn't understand how to analyze that. Could you rephrase?"
                )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _generate_prompt(self, query: str, columns: List[str]) -> str:
        """
        Generate the prompt for intent detection.
        
        Args:
            query: User's question
            columns: List of available data columns
            
        Returns:
            Formatted prompt for LLM
        """
        # Look for the template file
        template_paths = [
            "src/insights/prompts/intent_detection.tpl",
            "src/watchdog_ai/insights/prompts/intent_detection.tpl",
            "prompt_templates/intent_detection.tpl"
        ]
        
        template = None
        for path in template_paths:
            try:
                with open(path, "r") as f:
                    template = f.read()
                    break
            except FileNotFoundError:
                continue
        
        # Fallback template if none found
        if template is None:
            template = """
            You are a data analyst assistant.
            Given the user query and available columns in a dataset, determine the most appropriate analysis to perform.
            
            Available columns:
            {available_columns}
            
            User query: {query}
            
            Return your analysis plan as a JSON object with the following structure:
            {
                "intent": "groupby | metric | trend | comparison",
                "metrics": [{"name": column, "aggregation": "sum | avg | count | min | max"}],
                "dimensions": [{"name": column, "type": "category | time | number"}],
                "filters": [{"column": column, "operator": "==, >, <, >=, <=", "value": value}],
                "sort": {"column": column, "direction": "asc | desc"},
                "limit": number
            }
            """
        
        # Format the template
        return template.format(
            available_columns="\n".join(f"- {col}" for col in columns),
            query=query
        )
    
    def _process_intent(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the intent and generate appropriate insight.
        
        Args:
            intent: Intent data extracted from LLM response
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Determine the intent type
            intent_type = intent.get('intent', 'unknown')
            
            # Process based on intent type
            if intent_type == "groupby":
                return self._handle_groupby(intent, df)
            elif intent_type == "metric":
                return self._handle_metric(intent, df)
            elif intent_type == "trend":
                return self._handle_trend(intent, df)
            elif intent_type == "comparison":
                return self._handle_comparison(intent, df)
            else:
                return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"Error processing intent: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _handle_groupby(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Handle groupby intent type."""
        try:
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= val]
                except Exception as filter_error:
                    logger.warning(f"Error applying filter: {str(filter_error)}")
            
            # Extract metrics and dimensions
            metrics = intent.get('metrics', [])
            dimensions = intent.get('dimensions', [])
            
            if not metrics or not dimensions:
                return self._generate_error_response("Missing metrics or dimensions for groupby")
            
            # Build aggregation dictionary
            aggs = {}
            for metric in metrics:
                aggs[metric['name']] = metric['aggregation']
            
            # Get dimension columns
            dimension_cols = [dim['name'] for dim in dimensions]
            
            # Group and aggregate
            result = filtered_df.groupby(dimension_cols).agg(aggs).reset_index()
            
            # Sort if specified
            if 'sort' in intent and intent['sort']:
                sort_col = intent['sort']['column']
                ascending = intent['sort']['direction'] == "asc"
                result = result.sort_values(sort_col, ascending=ascending)
            
            # Apply limit
            if 'limit' in intent and intent['limit']:
                result = result.head(intent['limit'])
            
            # Get primary metric and dimension
            primary_metric = metrics[0]
            primary_dimension = dimensions[0]
            
            # Get top result
            if not result.empty:
                top_row = result.iloc[0]
                
                # Format response
                return {
                    "summary": f"{top_row[primary_dimension['name']]} leads with {self._format_value(top_row[primary_metric['name']], primary_metric)} {primary_metric['name']}",
                    "value_insights": [
                        f"Top {len(result)} categories by {primary_metric['name']}",
                        f"Total {primary_metric['name']}: {self._format_value(result[primary_metric['name']].sum(), primary_metric)}"
                    ],
                    "actionable_flags": self._generate_recommendations(result, intent),
                    "metrics": {
                        "top_performer": top_row[primary_dimension['name']],
                        "value": self._format_value(top_row[primary_metric['name']], primary_metric),
                        "total": self._format_value(result[primary_metric['name']].sum(), primary_metric)
                    },
                    "chart_data": self._format_chart_data(result, primary_dimension['name'], primary_metric['name']),
                    "confidence": "high"
                }
            else:
                return {
                    "summary": "No data found matching the criteria",
                    "value_insights": [],
                    "actionable_flags": [
                        "Consider broadening your search criteria"
                    ],
                    "confidence": "medium"
                }
            
        except Exception as e:
            logger.error(f"Error in groupby handler: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error analyzing data: {str(e)}")
    
    def _handle_metric(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Handle metric intent type."""
        try:
            # Get primary metric
            metrics = intent.get('metrics', [])
            if not metrics:
                return self._generate_error_response("No metrics specified")
                
            metric = metrics[0]
            
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "<=":
                        filtered_df = filtered_df[filtere


    
    def _handle_trend(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Handle trend intent type for time series analysis."""
        try:
            # Extract metrics and dimensions
            metrics = intent.get('metrics', [])
            dimensions = intent.get('dimensions', [])
            
            if not metrics:
                return self._generate_error_response("No metrics specified for trend analysis")
                
            # Get the primary metric
            metric = metrics[0]
            
            # Find time dimension
            time_dim = None
            for dim in dimensions:
                if dim.get('type') == 'time' or 'time' in dim.get('name', '').lower() or 'date' in dim.get('name', '').lower():
                    time_dim = dim
                    break
                    
            if not time_dim:
                return self._generate_error_response("No time dimension found for trend analysis")
            
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= val]
                except Exception as filter_error:
                    logger.warning(f"Error applying filter: {str(filter_error)}")
            
            # Ensure time column is datetime
            time_col = time_dim['name']
            try:
                filtered_df[time_col] = pd.to_datetime(filtered_df[time_col])
            except Exception as e:
                logger.warning(f"Failed to convert {time_col} to datetime: {str(e)}")
                
            # Group by time and aggregate metric
            try:
                # Try different time frequencies based on data range
                date_range = (filtered_df[time_col].max() - filtered_df[time_col].min()).days
                if date_range > 365:
                    freq = 'M'  # Monthly for more than a year
                elif date_range > 30:
                    freq = 'W'  # Weekly for more than a month
                else:
                    freq = 'D'  # Daily for less than a month
                
                # Group by time with appropriate frequency
                result = filtered_df.groupby(pd.Grouper(key=time_col, freq=freq))[metric['name']].agg(metric['aggregation']).reset_index()
                result = result.sort_values(time_col)
            except Exception as e:
                logger.warning(f"Error in time grouping: {str(e)}")
                # Fallback to simple groupby
                result = filtered_df.groupby(time_col)[metric['name']].agg(metric['aggregation']).reset_index()
                result = result.sort_values(time_col)
            
            # Calculate trend
            if len(result) >= 2:
                # Calculate the percentage change
                first_value = result[metric['name']].iloc[0]
                last_value = result[metric['name']].iloc[-1]
                
                if first_value != 0:
                    pct_change = ((last_value - first_value) / first_value) * 100
                else:
                    pct_change = 0
                
                # Determine trend direction
                if pct_change > 5:
                    direction = "increased"
                elif pct_change < -5:
                    direction = "decreased"
                else:
                    direction = "remained stable"
                
                summary = (
                    f"{metric['name']} has {direction} by {abs(pct_change):.1f}% "
                    f"from {self._format_value(first_value, metric)} to {self._format_value(last_value, metric)}"
                )
                
                # Generate insight response
                return {
                    "summary": summary,
                    "value_insights": [
                        f"Analyzed {len(result)} time periods",
                        f"Trend direction: {direction.capitalize()}"
                    ],
                    "actionable_flags": self._generate_trend_recommendations(pct_change),
                    "metrics": {
                        "start_value": self._format_value(first_value, metric),
                        "end_value": self._format_value(last_value, metric),
                        "change_percentage": f"{pct_change:.1f}%",
                        "period_count": len(result)
                    },
                    "chart_data": {
                        "type": "line",
                        "data": {
                            "x": result[time_col].dt.strftime('%Y-%m-%d').tolist(),
                            "y": result[metric['name']].tolist()
                        },
                        "title": f"{metric['name']} over time"
                    },
                    "confidence": "high"
                }
            else:
                return {
                    "summary": f"Not enough time periods to analyze trends for {metric['name']}",
                    "value_insights": [
                        f"Only {len(result)} time periods available",
                        "Need at least 2 periods for trend analysis"
                    ],
                    "actionable_flags": [
                        "Consider expanding the time range",
                        "Ensure time data is properly formatted"
                    ],
                    "confidence": "low"
                }
        
        except Exception as e:
            logger.error(f"Error in trend handler: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error analyzing trend: {str(e)}")
    
    def _handle_comparison(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Handle comparison intent type."""
        try:
            # Extract metrics and dimensions
            metrics = intent.get('metrics', [])
            dimensions = intent.get('dimensions', [])
            filters = intent.get('filters', [])
            
            if not metrics or not dimensions:
                return self._generate_error_response("Missing metrics or dimensions for comparison")
            
            # Get primary metric and dimension
            metric = metrics[0]
            dimension = dimensions[0]
            
            # Identify comparison categories
            comparison_values = []
            comparison_filters = []
            
            # See if we can extract comparison categories from filters
            for filter in filters:
                if filter.get('column') == dimension['name'] and filter.get('operator') == '==':
                    comparison_values.append(filter.get('value'))
                    comparison_filters.append(filter)
            
            # If no comparison values found in filters, try to find top categories
            if not comparison_values:
                # Get top categories by the metric
                top_categories = df.groupby(dimension['name'])[metric['name']].agg(metric['aggregation']).nlargest(5)
                comparison_values = top_categories.index.tolist()[:2]  # Use top 2
            
            # Ensure we have at least two categories for comparison
            if len(comparison_values) < 2:
                # Get the most frequent/largest categories
                top_categories = df[dimension['name']].value_counts().nlargest(2).index.tolist()
                comparison_values = top_categories
            
            # Limit to at most 5 categories for clarity
            comparison_values = comparison_values[:5]
            
            # Prepare data for each category
            comparison_data = {}
            category_totals = []
            
            for category in comparison_values:
                # Filter to just this category
                category_df = df[df[dimension['name']] == category]
                
                # Calculate metric for this category
                if metric['aggregation'] == "count":
                    value = len(category_df)
                else:
                    value = getattr(category_df[metric['name']], metric['aggregation'])()
                
                comparison_data[category] = value
                category_totals.append({
                    "category": category,
                    "value": value
                })
            
            # Sort from highest to lowest
            category_totals = sorted(category_totals, key=lambda x: x["value"], reverse=True)
            
            # Generate comparison summary
            if len(category_totals) >= 2:
                top_category = category_totals[0]["category"]
                second_category = category_totals[1]["category"]
                top_value = category_totals[0]["value"]
                second_value = category_totals[1]["value"]
                
                if top_value > 0 and second_value > 0:
                    pct_difference = ((top_value - second_value) / second_value) * 100
                    summary = (
                        f"{top_category} has {self._format_value(top_value, metric)} {metric['name']}, "
                        f"which is {abs(pct_difference):.1f}% higher than {second_category} "
                        f"with {self._format_value(second_value, metric)}"
                    )
                else:
                    summary = (
                        f"{top_category} leads with {self._format_value(top_value, metric)} {metric['name']}, "
                        f"followed by {second_category} with {self._format_value(second_value, metric)}"
                    )
            else:
                top_category = category_totals[0]["category"]
                top_value = category_totals[0]["value"]
                summary = f"{top_category} has {self._format_value(top_value, metric)} {metric['name']}"
            
            # Prepare chart data
            chart_data = {
                "type": "bar",
                "data": {
                    "x": [item["category"] for item in category_totals],
                    "y": [item["value"] for item in category_totals]
                },
                "title": f"Comparison of {metric['name']} by {dimension['name']}"
            }
            
            # Generate insight response
            return {
                "summary": summary,
                "value_insights": [
                    f"Compared {len(category_totals)} categories",
                    f"Top performer: {top_category}"
                ],
                "actionable_flags": self._generate_comparison_recommendations(category_totals, metric),
                "metrics": comparison_data,
                "chart_data": chart_data,
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"Error in comparison handler: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error performing comparison: {str(e)}")
    
    def _format_value(self, value: Any, metric: Dict[str, Any]) -> str:
        """Format a value based on the metric type."""
        try:
            # Check for monetary metrics
            if "price" in metric.get('name', '').lower() or "gross" in metric.get('name', '').lower() or "profit" in metric.get('name', '').lower() or "revenue" in metric.get('name', '').lower():
                return f"${value:,.2f}"
                
            # Check for percentage metrics
            if "percent" in metric.get('name', '').lower() or "rate" in metric.get('name', '').lower():
                return f"{value:.1f}%"
                
            # Standard number formatting
            if isinstance(value, (int, float)):
                # Check if it's a large number or has decimals
                if value % 1 == 0 and value < 1000000:
                    return f"{int(value):,d}"
                else:
                    return f"{value:,.2f}"
            
            # Return string representation for non-numeric
            return str(value)
        except Exception as e:
            logger.warning(f"Error formatting value: {str(e)}")
            return str(value)
    
    def _format_chart_data(self, df: pd.DataFrame, x_column: str, y_column: str) -> Dict[str, Any]:
        """Format data for visualization."""
        try:
            # Basic chart data structure
            chart_data = {
                "type": "bar",  # Default to bar chart
                "data": {
                    "x": df[x_column].tolist(),
                    "y": df[y_column].tolist()
                }
            }
            
            # Check if x-axis data looks like dates
            if pd.api.types.is_datetime64_any_dtype(df[x_column]):
                chart_data["type"] = "line"
                # Convert dates to strings for serialization
                chart_data["data"]["x"] = df[x_column].dt.strftime('%Y-%m-%d').tolist()
            
            return chart_data
        except Exception as e:
            logger.warning(f"Error formatting chart data: {str(e)}")
            return {
                "type": "bar",
                "data": {
                    "x": ["Error"],
                    "y": [0]
                }
            }
    
    def _generate_recommendations(self, df: pd.DataFrame, intent: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on data analysis."""
        try:
            recommendations = []
            
            # Get primary metric
            metrics = intent.get('metrics', [])
            if not metrics:                "summary": summary,
                "value_insights": [
                    f"Calculated from {len(filtered_df):,} records",
                    f"Using {metric['aggregation']} aggregation"
                ],
                "actionable_flags": self._generate_recommendations_for_metric(value, metric),
                "metrics": {
                    "value": self._format_value(value, metric),
                    "record_count": len(filtered_df),
                    "aggregation": metric['aggregation']
                },
                "confidence": "high"
            }
        # Initialize LLM engine
        self._initialize_llm_engine()
        
        # Initialize conversation state
        self._initialize_state()
        
        logger.info(
            "Initialized ConversationManager",
            extra={
                "component": "conversation_manager",
                "use_mock": self.use_mock
            }
        )
    
    def _initialize_llm_engine(self):
        """Initialize the LLM engine for conversation processing."""
        try:
            from watchdog_ai.llm.llm_engine import LLMEngine
            self.llm_engine = LLMEngine(use_mock=self.use_mock, api_key=self.api_key)
        except ImportError:
            logger.warning("Could not import LLMEngine, using mock implementation")
            # Create a mock LLM engine class
            class MockLLMEngine:
                def __init__(self, *args, **kwargs):
                    self.chat = type('obj', (object,), {
                        'completions': type('obj', (object,), {
                            'create': lambda **kwargs: type('obj', (object,), {
                                'choices': [type('obj', (object,), {
                                    'message': type('obj', (object,), {
                                        'content': '{"intent": "metric", "metrics": [{"name": "sales", "aggregation": "sum"}], "dimensions": [], "filters": []}'
                                    })
                                })]
                            })
                        })
                    })
            self.llm_engine = MockLLMEngine()
    
    def _initialize_state(self):
        """Initialize conversation state in Streamlit session."""
        # Initialize conversation history in session state if not exists
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'validation_context' not in st.session_state:
            st.session_state['validation_context'] = {}
    
    def generate_insight(self, query: str, validation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an insight based on the user query and data context.
        
        Args:
            query: User's question or prompt
            validation_context: Optional context with data validation information
            
        Returns:
            Structured insight response
        """
        try:
            logger.info(f"Generating insight for query: {query}")
            
            # Save the prompt in session state
            st.session_state['current_prompt'] = query
            
            # Get data from validation context
            df = None
            if validation_context and 'validated_data' in validation_context:
                df = validation_context.get('validated_data')
            
            # Process the query and get response
            if df is not None:
                # Use data-aware processing if DataFrame is available
                response = self.process_query(query, df)
            else:
                # Fallback to simple processing without data context
                response = self._generate_basic_insight(query)
            
            # Add timestamp
            timestamp = datetime.now().isoformat()
            
            # Add to conversation history
            st.session_state['conversation_history'].append({
                'prompt': query,
                'response': response,
                'timestamp': timestamp
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a data-aware query using intent detection.
        
        Args:
            query: User's question
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Generate prompt with available columns
            prompt = self._generate_prompt(query, df.columns.tolist())
            
            # Get LLM response
            response = self.llm_engine.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse and validate response
            try:
                # Try to parse as JSON first
                response_text = response.choices[0].message.content
                intent_data = json.loads(response_text)
                return self._process_intent(intent_data, df)
            except Exception as e:
                logger.error(f"Intent parsing error: {str(e)}")
                return self._generate_error_response(
                    "I couldn't understand how to analyze that. Could you rephrase?"
                )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _generate_prompt(self, query: str, columns: List[str]) -> str:
        """
        Generate the prompt for intent detection.
        
        Args:
            query: User's question
            columns: List of available data columns
            
        Returns:
            Formatted prompt for LLM
        """
        # Look for the template file
        template_paths = [
            "src/insights/prompts/intent_detection.tpl",
            "src/watchdog_ai/insights/prompts/intent_detection.tpl",
            "prompt_templates/intent_detection.tpl"
        ]
        
        template = None
        for path in template_paths:
            try:
                with open(path, "r") as f:
                    template = f.read()
                    break
            except FileNotFoundError:
                continue
        
        # Fallback template if none found
        if template is None:
            template = """
            You are a data analyst assistant.
            Given the user query and available columns in a dataset, determine the most appropriate analysis to perform.
            
            Available columns:
            {available_columns}
            
            User query: {query}
            
            Return your analysis plan as a JSON object with the following structure:
            {
                "intent": "groupby | metric | trend | comparison",
                "metrics": [{"name": column, "aggregation": "sum | avg | count | min | max"}],
                "dimensions": [{"name": column, "type": "category | time | number"}],
                "filters": [{"column": column, "operator": "==, >, <, >=, <=", "value": value}],
                "sort": {"column": column, "direction": "asc | desc"},
                "limit": number
            }
            """
        
        # Format the template
        return template.format(
            available_columns="\n".join(f"- {col}" for col in columns),
            query=query
        )
    
    def _process_intent(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the intent and generate appropriate insight.
        
        Args:
            intent: Intent data extracted from LLM response
            df: DataFrame to analyze
            
        Returns:
            Structured insight response
        """
        try:
            # Determine the intent type
            intent_type = intent.get('intent', 'unknown')
            
            # Process based on intent type
            if intent_type == "groupby":
                return self._handle_groupby(intent, df)
            elif intent_type == "metric":
                return self._handle_metric(intent, df)
            elif intent_type == "trend":
                return self._handle_trend(intent, df)
            elif intent_type == "comparison":
                return self._handle_comparison(intent, df)
            else:
                return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"Error processing intent: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))
    
    def _handle_groupby(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Handle groupby intent type."""
        try:
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= val]
                except Exception as filter_error:
                    logger.warning(f"Error applying filter: {str(filter_error)}")
            
            # Extract metrics and dimensions
            metrics = intent.get('metrics', [])
            dimensions = intent.get('dimensions', [])
            
            if not metrics or not dimensions:
                return self._generate_error_response("Missing metrics or dimensions for groupby")
            
            # Build aggregation dictionary
            aggs = {}
            for metric in metrics:
                aggs[metric['name']] = metric['aggregation']
            
            # Get dimension columns
            dimension_cols = [dim['name'] for dim in dimensions]
            
            # Group and aggregate
            result = filtered_df.groupby(dimension_cols).agg(aggs).reset_index()
            
            # Sort if specified
            if 'sort' in intent and intent['sort']:
                sort_col = intent['sort']['column']
                ascending = intent['sort']['direction'] == "asc"
                result = result.sort_values(sort_col, ascending=ascending)
            
            # Apply limit
            if 'limit' in intent and intent['limit']:
                result = result.head(intent['limit'])
            
            # Get primary metric and dimension
            primary_metric = metrics[0]
            primary_dimension = dimensions[0]
            
            # Get top result
            if not result.empty:
                top_row = result.iloc[0]
                
                # Format response
                return {
                    "summary": f"{top_row[primary_dimension['name']]} leads with {self._format_value(top_row[primary_metric['name']], primary_metric)} {primary_metric['name']}",
                    "value_insights": [
                        f"Top {len(result)} categories by {primary_metric['name']}",
                        f"Total {primary_metric['name']}: {self._format_value(result[primary_metric['name']].sum(), primary_metric)}"
                    ],
                    "actionable_flags": self._generate_recommendations(result, intent),
                    "metrics": {
                        "top_performer": top_row[primary_dimension['name']],
                        "value": self._format_value(top_row[primary_metric['name']], primary_metric),
                        "total": self._format_value(result[primary_metric['name']].sum(), primary_metric)
                    },
                    "chart_data": self._format_chart_data(result, primary_dimension['name'], primary_metric['name']),
                    "confidence": "high"
                }
            else:
                return {
                    "summary": "No data found matching the criteria",
                    "value_insights": [],
                    "actionable_flags": [
                        "Consider broadening your search criteria"
                    ],
                    "confidence": "medium"
                }
            
        except Exception as e:
            logger.error(f"Error in groupby handler: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error analyzing data: {str(e)}")
    
    def _handle_metric(self, intent: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Handle metric intent type."""
        try:
            # Get primary metric
            metrics = intent.get('metrics', [])
            if not metrics:
                return self._generate_error_response("No metrics specified")
                
            metric = metrics[0]
            
            # Apply filters
            filtered_df = df.copy()
            for filter in intent.get('filters', []):
                try:
                    col = filter['column']
                    op = filter['operator']
                    val = filter['value']
                    
                    # Apply filter based on operator
                    if op == "==":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == ">":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "<":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= val]
                except Exception as filter_error:
                    logger.warning(f"Error applying filter: {str(filter_error)}")
            
            # Calculate metric
            if metric['aggregation'] == "count":
                value = len(filtered_df)
            else:
                value = getattr(filtered_df[metric['name']], metric['aggregation'])()
            
            # Format summary based on metric type
            if "gross" in metric['name'].lower() or "profit" in metric['name'].lower():
                summary = f"Total {metric['name']} is ${value:,.2f}"
            else:
                summary = f"Total {metric['name']} is {value:,}"
            
            return {
                "summary":

