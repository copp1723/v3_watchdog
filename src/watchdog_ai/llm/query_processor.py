"""
Query processor for efficient LLM-powered data analysis.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .llm_engine import LLMEngine
from ..models.query_models import QueryContext, QueryResult, IntentSchema

logger = logging.getLogger(__name__)

# Cache for storing processed results
query_cache = {}

def process_query(context: QueryContext) -> QueryResult:
    """
    Process a user query with the LLM.
    
    Args:
        context: QueryContext object containing query and DataFrame
        
    Returns:
        QueryResult object containing the result
    """
    try:
        logger.info(f"Processing query: {context.query}")
        
        # Basic validation
        if context.df is None or context.df.empty:
            return QueryResult(
                status="error",
                message="No data available. Please upload data first.",
                data=None,
                intent=None
            )
        
        # Get LLM engine
        llm_engine = LLMEngine()
        
        # Analyze query to detect if it's a specific lookup request
        is_specific_lookup = False
        
        # Keywords that suggest a specific entity lookup
        specific_keywords = [
            'specific', 'particular', 'exact', 'individual', 'single', 
            'sold by', 'for the', 'what was the', 'what are the', 'made by',
            'days to close', 'closing time', 'time to close', 'closing days', 'sale duration'
        ]
        
        # Check for specific sale rep names, vehicle models, or specific customer queries
        for keyword in specific_keywords:
            if keyword.lower() in context.query.lower():
                is_specific_lookup = True
                break
        
        # Create simple intent detection prompt
        intent_prompt = f"""Detect the user's intent from this query: "{context.query}"
        
        The data has these columns: {', '.join(context.df.columns)}
        
        Return the intent as a JSON object.
        """
        
        # Get intent prediction from LLM
        try:
            intent_response = llm_engine.simple_query(intent_prompt)
            # Parse intent response if needed
            
            # Set intent based on analysis
            if is_specific_lookup:
                intent = IntentSchema(
                    intent="specific_lookup",  # Use specific lookup intent
                    metric=None,
                    category=None
                )
            else:
                intent = IntentSchema(
                    intent="groupby_summary",  # Default to groupby_summary
                    metric=None,
                    category=None
                )
        except Exception as e:
            logger.warning(f"Intent detection error: {e}")
            intent = IntentSchema(
                intent="fallback",
                metric=None,
                category=None
            )
            
        # Update the context with intent
        context.intent = intent
        
        # Create enhanced context with column information and query type
        enhanced_context = {
            "columns": context.df.columns.tolist(),
            "data_types": {col: str(context.df[col].dtype) for col in context.df.columns},
            "is_specific_lookup": is_specific_lookup,
            "query_type": intent.intent
        }
        
        # Process the query using the LLM
        response = llm_engine.generate_insight(
            query=context.query,
            df=context.df,
            context=enhanced_context
        )
        
        # Check for errors
        if response.get("error_type"):
            return QueryResult(
                status="error",
                message=response.get("summary", "Error processing query"),
                data=response,
                intent=intent
            )
            
        # Return successful result
        return QueryResult(
            status="success",
            message=response.get("summary", ""),
            data=response,
            intent=intent
        )
            
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        return QueryResult(
            status="error",
            message=f"Error processing your request: {str(e)}",
            data=None,
            intent=None
        )