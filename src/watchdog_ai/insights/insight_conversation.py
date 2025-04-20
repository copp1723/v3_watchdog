"""
Conversation manager for insight generation and follow-ups. Now uses an intent-based pipeline.
"""

import streamlit as st
from typing import Dict, Any, Optional, Literal, List, Tuple
import logging
from datetime import datetime
import json
import re
import pandas as pd
from pydantic import BaseModel, ValidationError, Field, validator
import numpy as np
from .insight_functions import InsightFunctions
from .utils import validate_numeric_columns

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from ..config import SessionKeys
from watchdog_ai.llm.llm_engine import LLMEngine
from ..models import InsightResponse
from watchdog_ai.insights.insight_functions import find_column
from watchdog_ai.insights.utils import is_all_sales_dataset
from watchdog_ai.llm.llm_engine import LLMEngine
from validators.validator_service import DataValidator
from watchdog_ai.insights.middleware import InsightMiddleware

# --- New Pydantic Schema for Intent ---
class IntentSchema(BaseModel):
    """Schema for validating intents from LLM."""
    intent: str = Field(..., description="Type of analysis to perform")
    metric: Optional[str] = Field(None, description="Metric to analyze")
    category: Optional[str] = Field(None, description="Category to group by")
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    filter: Optional[str] = Field(None, description="Filter condition to apply")

    @validator('intent')
    def validate_intent(cls, v):
        valid_intents = ["groupby_summary", "total_summary", "fallback"]
        if v not in valid_intents:
            raise ValueError(f"Invalid intent: {v}. Must be one of {valid_intents}")
        return v

    @validator('aggregation')
    def validate_aggregation(cls, v, values):
        # Only require aggregation for non-fallback intents
        if 'intent' in values and values['intent'] != 'fallback' and not v:
            # Default to sum if not specified
            return "sum"
        if v and v not in ["sum", "count", "mean", "avg", "average", "max", "min"]:
            raise ValueError(f"Invalid aggregation: {v}")
        return v

# --- Aggregation Logic ---
class Aggregator:
    @staticmethod
    def sum(series):
        return series.sum()

    @staticmethod
    def max(series):
        # Returns index (category) and value
        if pd.api.types.is_numeric_dtype(series):
            return series.idxmax(), series.max()
        else: # Handle count case (index is already the category)
             return series.index[0], series.iloc[0] # Assuming series is sorted descending for count


    @staticmethod
    def mean(series):
        return series.mean()

def expand_variants(name):
    """
    Generate common column name variants for robust matching.
    """
    if not name:
        return []
    base = name.replace("_", "").replace(" ", "")
    variants = set([
        name,
        name.lower(),
        name.upper(),
        name.replace("_", ""),
        name.replace(" ", ""),
        name.lower().replace("_", ""),
        name.lower().replace(" ", ""),
        name.capitalize(),
        name.title(),
    ])
    return list(variants)

AGG_FUNCS = {
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "count": "count"
}

class ConversationManager:
    """Manages conversation flow and LLM interactions using intent detection."""

    def __init__(self, use_mock: bool = False):
        """Initialize the conversation manager."""
        self.llm_engine = LLMEngine()
        self.use_mock = use_mock
        self.insight_functions = InsightFunctions()
        self.data_validator = DataValidator()
        self.middleware = InsightMiddleware()
        
        # Initialize session state if not exists
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "last_intent" not in st.session_state:
            st.session_state["last_intent"] = None
        if "last_result" not in st.session_state:
            st.session_state["last_result"] = None
        if "last_query" not in st.session_state:
            st.session_state["last_query"] = None
        if "query_text" not in st.session_state:
            st.session_state["query_text"] = ""
        if "selected_example" not in st.session_state:
            st.session_state["selected_example"] = None
        if "intent_cache" not in st.session_state:
            st.session_state["intent_cache"] = {}
        if "result_cache" not in st.session_state:
            st.session_state["result_cache"] = {}
        if "conversation_state" not in st.session_state:
            st.session_state["conversation_state"] = {}
        if "metrics_history" not in st.session_state:
            st.session_state["metrics_history"] = []
        if "analysis_state" not in st.session_state:
            st.session_state["analysis_state"] = {}

    def _clear_state(self) -> None:
        """Completely reset all chat-related session state."""
        logger.info("Clearing all chat state")
        
        # Log state before clearing
        logger.debug(f"BEFORE CLEAR STATE: {st.session_state}")
        
        # Common keys - expanded list to ensure comprehensive reset
        keys_to_clear = [
            "chat_history", "last_intent", "last_result", "last_query", 
            "query_text", "selected_example", "intent_cache", "result_cache",
            "conversation_state", "metrics_history", "analysis_state"
        ]
        
        # Clear each key
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = [] if key == "chat_history" else None
                logger.debug(f"Cleared state key: {key}")
        
        # Debug logging to verify state reset
        logger.debug(f"AFTER CLEAR STATE: {st.session_state}")
    
    def load_prompt(self, template_name: str, context: dict) -> str:
        """Load and render a prompt template."""
        return self.llm_engine.load_prompt(template_name, context)

    def process_query(self, query: str, validation_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user query through intent detection and analysis pipeline."""
        self._clear_state()
        
        # Get columns from validation context or uploaded data
        try:
            if validation_context and 'columns' in validation_context:
                columns = validation_context['columns']
            else:
                df = st.session_state.get(SessionKeys.UPLOADED_DATA)
                if df is not None:
                    columns = df.columns.tolist()
                else:
                    columns = []
            logger.debug(f"SCHEMA COLUMNS PROVIDED: {columns}")
            
            # Load and render prompt template with query and columns
            prompt_template_str = self.load_prompt("intent_detection.tpl", {"query": query, "columns": columns})
            logger.debug(f"\U0001f9e0 FINAL INTENT PROMPT:\n{prompt_template_str}")
            
            # Get LLM response
            response = self.llm_engine.client.chat.completions.create(
                model=self.llm_engine.model,
                messages=[{"role": "user", "content": prompt_template_str}],
                temperature=0.1
            )
            raw = response.choices[0].message.content
            logger.debug(f"\U0001f4ac LLM raw response:\n{raw}")
            print("\n=== LLM RAW RESPONSE ===\n", raw, "\n========================\n")
            
            # Defensive JSON parsing
            if "{" not in raw:
                logger.warning("\u26a0\ufe0f No JSON detected in response, forcing fallback.")
                return {
                    "summary": "\u26a0\ufe0f Couldn't interpret your request - the AI didn't return valid output.",
                    "metrics": {}, "breakdown": [], "recommendations": [],
                    "confidence": "low", "error_type": "LLM_EMPTY_RESPONSE"
                }
                
            try:
                intent_data = json.loads(raw)
                logger.debug(f"Parsed intent data: {intent_data}")
                
                # Validate against schema
                validated_intent = IntentSchema(**intent_data)
                logger.debug(f"Validated intent: {validated_intent}")
                
                # Process the validated intent
                return self._process_intent(validated_intent, query)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return {
                    "summary": "\u26a0\ufe0f Error parsing AI response",
                    "metrics": {}, "breakdown": [], "recommendations": [],
                    "confidence": "low", "error_type": "JSON_PARSE_ERROR"
                }
            except ValidationError as e:
                logger.error(f"Schema validation error: {e}")
                return {
                    "summary": "\u26a0\ufe0f Invalid intent format from AI",
                    "metrics": {}, "breakdown": [], "recommendations": [],
                    "confidence": "low", "error_type": "SCHEMA_VALIDATION_ERROR"
                }
            except Exception as e:
                logger.error(f"Unexpected error in process_query: {e}")
                return {
                    "summary": f"\u26a0\ufe0f Unexpected error: {str(e)}",
                    "metrics": {}, "breakdown": [], "recommendations": [],
                    "confidence": "low", "error_type": "UNEXPECTED_ERROR"
                }
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return {
                "summary": f"⚠️ Error processing your request: {str(e)}",
                "metrics": {}, "breakdown": [], "recommendations": [],
                "confidence": "low", "error_type": "PROCESSING_ERROR"
            }

    def _process_intent(self, intent: IntentSchema, query: str) -> Dict[str, Any]:
        """Process a validated intent and return analysis results."""
        try:
            # Get the DataFrame from session state
            df = st.session_state.get(SessionKeys.UPLOADED_DATA)
            if df is None or df.empty:
                return {
                    "summary": "⚠️ No data available for analysis. Please upload a dataset first.",
                    "metrics": {},
                    "breakdown": [],
                    "recommendations": ["Upload a dataset to analyze"],
                    "confidence": "low",
                    "error_type": "NO_DATA"
                }

            # Get columns directly from DataFrame
            columns = df.columns.tolist()

            # Validate numeric columns
            df = validate_numeric_columns(df)
            
            # Run the analysis based on intent
            result = self.run_analysis(intent, query, df)
            
            # Cache the result
            st.session_state["last_result"] = result
            st.session_state["last_intent"] = intent
            st.session_state["last_query"] = query
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing intent: {e}")
            return {
                "summary": f"⚠️ Error processing your request: {str(e)}",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Try rephrasing your query"],
                "confidence": "low",
                "error_type": "PROCESSING_ERROR"
            }

    def run_analysis(self, intent: IntentSchema, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the appropriate analysis based on the intent."""
        logger.info("Running analysis for intent: %s", intent.intent)
        
        # Check if DataFrame is None or empty
        if df is None or df.empty:
            logger.warning("DataFrame is None or empty")
            return {
                "summary": "⚠️ No data available for analysis. Please upload a dataset first.",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Upload a dataset to analyze"],
                "confidence": "low",
                "error_type": "NO_DATA"
            }
        
        # Apply middleware pre-processing
        df = self.middleware.pre_process(df)
        
        # Validate data
        validation_result = self.data_validator.validate(df)
        if not validation_result["is_valid"]:
            logger.warning("Data validation failed: %s", validation_result["errors"])
            return {
                "summary": f"⚠️ Data validation failed: {validation_result['errors'][0]}",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "VALIDATION_ERROR"
            }
        
        # Apply filter if specified
        if intent.filter:
            try:
                logger.info("Applying filter: %s", intent.filter)
                # Parse the filter condition
                if "=" in intent.filter:
                    col, val = intent.filter.split("=", 1)
                    col = col.strip()
                    val = val.strip().strip("'").strip('"')
                    df = df[df[col] == val]
                elif ">" in intent.filter:
                    col, val = intent.filter.split(">", 1)
                    col = col.strip()
                    val = float(val.strip())
                    df = df[df[col] > val]
                elif "<" in intent.filter:
                    col, val = intent.filter.split("<", 1)
                    col = col.strip()
                    val = float(val.strip())
                    df = df[df[col] < val]
                else:
                    logger.warning("Unsupported filter condition: %s", intent.filter)
            except Exception as e:
                logger.error("Error applying filter: %s", e)
                # Continue without the filter
        
        # Handle different intent types
        if intent.intent == "groupby_summary":
            logger.info("Handling groupby_summary: metric='%s', category='%s', aggregation='%s'", 
                       intent.metric, intent.category, intent.aggregation)
            
            # Special handling for sales metrics
            if intent.metric and intent.metric.lower() in ["issale", "is_sale", "sold", "sale", "numberofsales", "sales"]:
                # Check if this is a sales dataset
                is_sales = is_all_sales_dataset(df)
                logger.info("Fallback: interpreting sales metric '%s' as row count (all rows are sales). Dataset validation: %s", 
                           intent.metric, is_sales)
                
                # Use row count as the metric
                result = self.insight_functions.groupby_summary(
                    df, 
                    "IsSale",  # This is a special metric that will be interpreted as row count
                    intent.category,
                    intent.aggregation or "count"
                )
                
                # Add total sales count
                result["metrics"]["total_sales"] = len(df)
                
                return result
            
            # Special handling for profit margin
            if "profit margin" in query.lower():
                logger.info("Special handling for profit margin calculation")
                # Create a new column for profit margin
                df["profit_margin"] = df["profit"] / df["sold_price"]
                result = self.insight_functions.groupby_summary(
                    df,
                    "profit_margin",
                    intent.category,
                    intent.aggregation or "mean"
                )
                return result
            
            # Normal handling for other metrics
            try:
                return self.insight_functions.groupby_summary(
                    df, 
                    intent.metric,
                    intent.category,
                    intent.aggregation or "sum"
                )
            except Exception as e:
                logger.error("Error in groupby_summary: %s", e, exc_info=True)
                return {
                    "summary": f"⚠️ {str(e)}",
                    "metrics": {},
                    "breakdown": [],
                    "recommendations": [],
                    "confidence": "low",
                    "error_type": "DATA_ERROR"
                }
                
        elif intent.intent == "total_summary":
            logger.info("Handling total_summary: metric='%s'", intent.metric)
            
            # Special handling for sales metrics
            if intent.metric and intent.metric.lower() in ["issale", "is_sale", "sold", "sale", "numberofsales", "sales"]:
                # Use row count as the metric
                result = self.insight_functions.total_summary(
                    df, 
                    "IsSale"  # This is a special metric that will be interpreted as row count
                )
                
                # Add total sales count
                result["metrics"]["total_sales"] = len(df)
                
                return result
            
            # Special handling for profit margin
            if "profit margin" in query.lower():
                logger.info("Special handling for profit margin calculation")
                # Create a new column for profit margin
                df["profit_margin"] = df["profit"] / df["sold_price"]
                result = self.insight_functions.total_summary(
                    df,
                    "profit_margin"
                )
                return result
            
            # Normal handling for other metrics
            try:
                return self.insight_functions.total_summary(
                    df, 
                    intent.metric
                )
            except Exception as e:
                logger.error("Error in total_summary: %s", e, exc_info=True)
                return {
                    "summary": f"⚠️ {str(e)}",
                    "metrics": {},
                    "breakdown": [],
                    "recommendations": [],
                    "confidence": "low",
                    "error_type": "DATA_ERROR"
                }

        else:  # fallback
            logger.warning("Handling fallback for query: %s", query)
            return {
                "summary": "⚠️ I'm not sure how to analyze that. Could you try rephrasing your question?",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "INTENT_ERROR"
            }

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history."""
        return st.session_state.get("chat_history", [])
    
    def add_to_chat_history(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        st.session_state["chat_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        st.session_state["chat_history"] = []