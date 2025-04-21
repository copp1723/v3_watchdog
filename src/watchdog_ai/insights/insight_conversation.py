"""
Enhanced conversation manager with precision scoring and fallback handling.

This version uses parameter-based architecture and explicit context passing
instead of session state dependencies.
"""

from typing import Dict, Any, Optional, Literal, List, Tuple
import logging
from datetime import datetime
import json
import re
import pandas as pd
import time
from pydantic import BaseModel, ValidationError, Field, validator
import numpy as np
import uuid

from .insight_functions import InsightFunctions
from .utils import validate_numeric_columns
from .precision_scoring import PrecisionScoringEngine
from .fallback_renderer import FallbackRenderer, FallbackContext, FallbackReason
from .contracts import InsightContract, InsightContractEnforcer, create_default_contract
from .context import InsightExecutionContext, InsightPipeline, InsightPipelineStage
from .chart_renderer import ChartRenderer

# Configure logging
logger = logging.getLogger(__name__)

from ..config import SessionKeys
from ..llm.llm_engine import LLMEngine
from ..models import InsightResponse, InsightErrorType
from ..utils.adaptive_schema import SchemaProfile
from .insight_functions import find_column
from .utils import is_all_sales_dataset, expand_variants
from .middleware import InsightMiddleware
from validators.validator_service import DataValidator

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
        if 'intent' in values and values['intent'] != 'fallback' and not v:
            return "sum"  # Default to sum if not specified
        if v and v not in ["sum", "count", "mean", "avg", "average", "max", "min"]:
            raise ValueError(f"Invalid aggregation: {v}")
        return v

class QueryContext(BaseModel):
    """Context for a query, used to improve intent understanding."""
    query: str
    columns: List[str]
    data_types: Dict[str, str]
    query_history: List[str] = Field(default_factory=list)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    example_values: Dict[str, List[Any]] = Field(default_factory=dict)
    column_mappings: Dict[str, str] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ValidationStage(InsightPipelineStage):
    """Pipeline stage for validating input data."""
    
    def __init__(self):
        """Initialize the validation stage."""
        super().__init__(
            name="validation",
            description="Validates input data for analysis"
        )
        self.data_validator = DataValidator()
    
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        """Validate input data and update context."""
        df = context.df
        query = context.query
        
        # Check for empty dataframe
        if df is None or df.empty:
            result = {
                "summary": "⚠️ No data available for analysis. Please upload a dataset first.",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Upload a dataset to analyze"],
                "confidence": "low",
                "error_type": str(InsightErrorType.NO_VALID_DATA)
            }
            return context.with_additional_context(result=result, validation_status="error")
        
        # Validate numeric columns
        df = validate_numeric_columns(df)
        
        # Extract required columns based on query
        required_cols = extract_required_columns(query, df)
        
        # Check data quality
        is_valid, validation_info = check_data_quality(df, required_cols)
        
        if not is_valid:
            logger.warning(f"Data quality issues: {validation_info['issues']}")
            fallback_context = FallbackContext(
                reason=FallbackReason.DATA_QUALITY,
                details={"errors": validation_info['issues'], "query": query},
                original_query=query
            )
            result = FallbackRenderer().render_fallback(fallback_context)
            return context.with_additional_context(
                result=result, 
                validation_status="error",
                validation_info=validation_info
            )
        
        # Return updated context with validated dataframe and validation info
        return context.with_additional_context(
            df=df,
            validation_status="success",
            validation_info=validation_info,
            required_columns=required_cols
        )


class IntentDetectionStage(InsightPipelineStage):
    """Pipeline stage for detecting query intent."""
    
    def __init__(self):
        """Initialize the intent detection stage."""
        super().__init__(
            name="intent_detection",
            description="Detects user intent from query"
        )
        self.llm_engine = LLMEngine()
        self.precision_scorer = PrecisionScoringEngine()
    
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        """Detect intent and update context."""
        df = context.df
        query = context.query
        
        # Skip intent detection if validation failed
        if context.context_vars.get("validation_status") == "error":
            return context
        
        # Build enhanced context for intent detection
        enhanced_context = build_enhanced_context(df, query, context.context_vars)
        
        # Predict precision score
        precision_result = self.precision_scorer.predict_precision(query, enhanced_context)
        
        # Check if precision is too low for reliable answer
        if precision_result["confidence_level"] == "low" and precision_result["score"] < 0.3:
            logger.warning(f"Low precision score ({precision_result['score']:.2f}) for query: {query}")
            fallback_context = FallbackContext(
                reason=FallbackReason.LOW_PRECISION,
                details={"precision_score": precision_result["score"], "query": query},
                original_query=query
            )
            result = FallbackRenderer().render_fallback(fallback_context)
            return context.with_additional_context(
                result=result,
                intent_status="error",
                precision_score=precision_result
            )
        
        # Attempt to detect intent
        intent_result = get_intent(self.llm_engine, query, enhanced_context)
        
        # Update context with intent information
        if intent_result and isinstance(intent_result, dict) and "intent" in intent_result:
            try:
                # Validate intent schema
                intent_obj = IntentSchema(**intent_result)
                
                # Update context with validated intent
                return context.with_additional_context(
                    intent=intent_obj.dict(),
                    intent_status="success",
                    precision_score=precision_result
                )
                
            except ValidationError as ve:
                logger.warning(f"Intent validation failed: {ve}")
                return context.with_additional_context(
                    intent_status="error",
                    intent_error=str(ve),
                    precision_score=precision_result,
                    partial_intent=intent_result
                )
        
        # No valid intent detected, continue to direct LLM query
        return context.with_additional_context(
            intent_status="direct_query",
            precision_score=precision_result
        )


class IntentProcessingStage(InsightPipelineStage):
    """Pipeline stage for processing detected intent."""
    
    def __init__(self):
        """Initialize the intent processing stage."""
        super().__init__(
            name="intent_processing",
            description="Processes detected intent to generate insights"
        )
        self.insight_functions = InsightFunctions()
        self.data_validator = DataValidator()
        self.middleware = InsightMiddleware()
        self.fallback_renderer = FallbackRenderer()
    
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        """Process intent and update context."""
        # Skip if validation failed or no valid intent
        if context.context_vars.get("validation_status") == "error" or \
           context.context_vars.get("intent_status") in ["error", "direct_query"]:
            return context
        
        # Get intent from context
        intent_dict = context.context_vars.get("intent")
        if not intent_dict:
            return context
        
        # Create intent object
        try:
            intent = IntentSchema(**intent_dict)
            
            # Process the intent
            df = context.df
            query = context.query
            
            # Validate data
            df = validate_numeric_columns(df)
            validation_result = self.data_validator.validate(df)
            
            if not validation_result["is_valid"]:
                logger.warning(f"Data validation failed: {validation_result['errors']}")
                fallback_context = FallbackContext(
                    reason=FallbackReason.DATA_QUALITY,
                    details={"errors": validation_result["errors"]},
                    original_query=query
                )
                result = self.fallback_renderer.render_fallback(fallback_context)
                return context.with_additional_context(
                    result=result,
                    intent_processing_status="error"
                )
            
            # Apply middleware preprocessing
            df = self.middleware.pre_process(df)
            
            # Handle different intent types
            if intent.intent == "groupby_summary":
                logger.info(f"Processing groupby summary for metric={intent.metric}, category={intent.category}")
                result = handle_groupby(self.insight_functions, intent, df)
                
                # Add metadata
                result["_intent"] = {
                    "type": "groupby_summary",
                    "metric": intent.metric,
                    "category": intent.category,
                    "aggregation": intent.aggregation
                }
                
                return context.with_additional_context(
                    result=result,
                    intent_processing_status="success"
                )
                
            elif intent.intent == "total_summary":
                logger.info(f"Processing total summary for metric={intent.metric}")
                result = handle_total(self.insight_functions, intent, df)
                
                # Add metadata
                result["_intent"] = {
                    "type": "total_summary",
                    "metric": intent.metric
                }
                
                return context.with_additional_context(
                    result=result,
                    intent_processing_status="success"
                )
                
            else:
                logger.warning(f"Unknown intent: {intent.intent}")
                fallback_context = FallbackContext(
                    reason=FallbackReason.AMBIGUOUS_INTENT,
                    details={"intent": intent.intent},
                    original_query=query
                )
                result = self.fallback_renderer.render_fallback(fallback_context)
                return context.with_additional_context(
                    result=result,
                    intent_processing_status="error"
                )
        
        except Exception as e:
            logger.error(f"Error processing intent: {e}")
            result = self.fallback_renderer.render_error(e, context.query)
            return context.with_additional_context(
                result=result,
                intent_processing_status="error",
                error=str(e)
            )


class LLMQueryStage(InsightPipelineStage):
    """Pipeline stage for direct LLM query processing."""
    
    def __init__(self):
        """Initialize the LLM query stage."""
        super().__init__(
            name="llm_query",
            description="Processes query using direct LLM generation"
        )
        self.llm_engine = LLMEngine()
        self.contract_enforcer = InsightContractEnforcer()
    
    def process(self, context: InsightExecutionContext) -> InsightExecutionContext:
        """Process query with LLM and update context."""
        # Skip if validation failed or intent processing succeeded
        if context.context_vars.get("validation_status") == "error" or \
           (context.context_vars.get("intent_processing_status") == "success" and
            "result" in context.context_vars):
            return context
        
        # Get data and query
        df = context.df
        query = context.query
        schema = context.schema
        
        # Build enhanced context
        enhanced_context = build_enhanced_context(df, query, context.context_vars)
        
        # Get response from LLM
        response = self.llm_engine.generate_insight(
            query=query,
            df=df,
            schema=schema,
            context=enhanced_context
        )
        
        # Validate with contract enforcer
        if response and not response.get("error_type"):
            try:
                # Create a default contract
                contract = create_default_contract("general_insight")
                
                # Validate output against contract
                validation_result = self.contract_enforcer.validate_output(response, contract)
                
                # Add validation info to result
                response["_validation"] = {
                    "is_valid": validation_result["is_valid"],
                    "warnings": validation_result["warnings"]
                }
                
                # If validation failed and we're in strict mode, handle accordingly
                if not validation_result["is_valid"]:
                    logger.warning(f"Contract validation failed: {validation_result['errors']}")
                    # We still return the result, but with validation info
            except Exception as ve:
                logger.error(f"Error during contract validation: {ve}")
                # Continue without contract validation
        
        # Update context with result
        return context.with_additional_context(
            result=response,
            llm_query_status="success"
        )


class ConversationManager:
    """Manages conversation flow with precision scoring and fallbacks."""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the conversation manager."""
        self.llm_engine = LLMEngine(use_mock=use_mock)
        self.use_mock = use_mock
        
        # Set up pipeline
        self.pipeline = InsightPipeline()
        self.pipeline.add_stage(ValidationStage())
        self.pipeline.add_stage(IntentDetectionStage())
        self.pipeline.add_stage(IntentProcessingStage())
        self.pipeline.add_stage(LLMQueryStage())
        
        # Initialize components
        self.precision_scorer = PrecisionScoringEngine()
        self.fallback_renderer = FallbackRenderer()
        self.chart_renderer = ChartRenderer()
        self.contract_enforcer = InsightContractEnforcer()
        
        # Initialize state
        self.chat_history = []
        self.query_count = 0
        self.fallback_count = 0
        self.processing_times = []
        self.visualization_count = 0
    
    def process_query(
        self, 
        query: str, 
        df: pd.DataFrame, 
        schema: Optional[SchemaProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the insight pipeline with comprehensive tracking.
        
        Args:
            query: User query string
            df: DataFrame to analyze
            schema: Optional schema profile
            context: Optional additional context variables
            
        Returns:
            Dict containing the insight result
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Data validation pre-check
            from ..utils.dataframe_utils import DataFrameUtils
            df_valid, validation_issues = DataFrameUtils.validate_dataframe(df)
            
            if not df_valid:
                logger.warning(f"DataFrame validation failed: {validation_issues}")
                fallback_context = FallbackContext(
                    reason=FallbackReason.DATA_QUALITY,
                    details={"issues": validation_issues, "query": query},
                    original_query=query
                )
                result = self.fallback_renderer.render_fallback(fallback_context)
                self.fallback_count += 1
                
                # Update chat history
                self.add_to_chat_history("user", query)
                self.add_to_chat_history("assistant", json.dumps(result))
                
                # Update metrics
                end_time = time.time()
                self.processing_times.append(end_time - start_time)
                
                return result
            
            # Create execution context
            exec_context = InsightExecutionContext(
                df=df,
                query=query,
                user_role=context.get("user_role", "general_manager") if context else "general_manager",
                schema=schema,
                context_vars=context or {},
                trace_id=str(uuid.uuid4())
            )
            
            # Execute pipeline
            result = self.pipeline.execute(exec_context)
            
            # Track visualizations
            if "_visualization" in result:
                self.visualization_count += 1
                
            # Add confidence score if not present
            if "confidence" not in result:
                # Use precision scorer to add confidence score
                precision_context = {
                    "columns": df.columns.tolist(),
                    "query": query,
                    "nan_percentage": sum(df.isnull().sum()) / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
                }
                precision_result = self.precision_scorer.predict_precision(query, precision_context)
                result["confidence"] = precision_result["confidence_level"]
                result["_precision_score"] = precision_result["score"]
            
            # Apply contract validation
            try:
                # Only do contract validation for successful insights
                if not result.get("error_type"):
                    contract = create_default_contract("general_insight")
                    validation_result = self.contract_enforcer.validate_output(result, contract)
                    
                    # Add validation metadata
                    result["_contract_validation"] = {
                        "is_valid": validation_result["is_valid"],
                        "warnings": validation_result.get("warnings", []),
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as ve:
                logger.warning(f"Contract validation error: {ve}")
                # Continue even if contract validation fails
            
            # Update metrics
            end_time = time.time()
            self.processing_times.append(end_time - start_time)
            
            # Check if result indicates a fallback/error
            if result.get("error_type"):
                self.fallback_count += 1
            
            # Add execution metadata
            result["_execution_metadata"] = {
                "execution_time_ms": (end_time - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
                "query_id": str(uuid.uuid4()),
                "trace_id": exec_context.trace_id
            }
            
            # Update chat history
            self.add_to_chat_history("user", query)
            self.add_to_chat_history("assistant", json.dumps(result))
            
            return result
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Error processing query: {e}")
            
            end_time = time.time()
            self.processing_times.append(end_time - start_time)
            self.fallback_count += 1
            
            # Create error response with enhanced context
            error_result = {
                "summary": f"⚠️ Error processing your request: {str(e)}",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Try rephrasing your query"],
                "confidence": "low",
                "error_type": str(InsightErrorType.PROCESSING_ERROR),
                "_error_details": {
                    "exception": str(e),
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": str(uuid.uuid4())
                }
            }
            
            # Update chat history
            self.add_to_chat_history("user", query)
            self.add_to_chat_history("assistant", json.dumps(error_result))
            
            return error_result
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history."""
        return self.chat_history
    
    def add_to_chat_history(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics including visualization stats."""
        avg_time = sum(self.processing_times) / max(1, len(self.processing_times))
        
        # Calculate success rates
        success_count = self.query_count - self.fallback_count
        success_rate = success_count / max(1, self.query_count)
        
        # Get visualization rate
        visualization_rate = self.visualization_count / max(1, success_count) if success_count > 0 else 0
        
        return {
            "query_count": self.query_count,
            "fallback_count": self.fallback_count,
            "visualization_count": self.visualization_count,
            "avg_processing_time_ms": avg_time * 1000,
            "success_rate": success_rate,
            "visualization_rate": visualization_rate,
            "llm_metrics": self.llm_engine.get_metrics(),
            "timestamp": datetime.now().isoformat(),
            "pipeline_metrics": {
                "total_errors": self.fallback_count,
                "validation_errors": sum(1 for msg in self.chat_history 
                                        if "role" in msg and msg["role"] == "assistant" 
                                        and "validation_status" in msg.get("content", {}) 
                                        and msg["content"]["validation_status"] == "error")
            }
        }
    
    def get_trace_info(self, trace_id: str = None, query_index: int = None) -> Dict[str, Any]:
        """
        Get detailed trace information for debugging and auditing.
        
        Args:
            trace_id: Optional trace ID to look up
            query_index: Optional query index in history (0-based)
            
        Returns:
            Dict with trace information
        """
        # If neither parameter is provided, return the latest query
        if trace_id is None and query_index is None:
            # Find the latest assistant message
            for i in range(len(self.chat_history) - 1, -1, -1):
                if self.chat_history[i]["role"] == "assistant":
                    try:
                        content = json.loads(self.chat_history[i]["content"])
                        if "_execution_metadata" in content:
                            trace_info = content["_execution_metadata"].copy()
                            trace_info["query"] = self.chat_history[i-1]["content"] if i > 0 else ""
                            trace_info["result"] = content
                            return trace_info
                    except:
                        pass
            
            return {"error": "No trace information found in recent history"}
        
        # Look up by trace ID
        if trace_id:
            for i, msg in enumerate(self.chat_history):
                if msg["role"] == "assistant":
                    try:
                        content = json.loads(msg["content"])
                        if "_execution_metadata" in content and content["_execution_metadata"].get("trace_id") == trace_id:
                            trace_info = content["_execution_metadata"].copy()
                            trace_info["query"] = self.chat_history[i-1]["content"] if i > 0 else ""
                            trace_info["result"] = content
                            return trace_info
                    except:
                        pass
            
            return {"error": f"Trace ID not found: {trace_id}"}
        
        # Look up by query index
        if query_index is not None:
            # Calculate the assistant message index (assuming user-assistant pairs)
            assistant_index = query_index * 2 + 1
            
            if assistant_index < len(self.chat_history):
                try:
                    content = json.loads(self.chat_history[assistant_index]["content"])
                    trace_info = content.get("_execution_metadata", {}).copy()
                    trace_info["query"] = self.chat_history[assistant_index-1]["content"] if assistant_index > 0 else ""
                    trace_info["result"] = content
                    return trace_info
                except:
                    return {"error": f"Could not parse trace info for query index {query_index}"}
            
            return {"error": f"Query index out of range: {query_index}"}
        
        return {"error": "Could not find trace information"}


# Helper functions

def extract_required_columns(query: str, df: pd.DataFrame) -> List[str]:
    """
    Extract columns that are likely required for the given query.
    Uses semantic understanding to identify relevant columns.
    """
    query_lower = query.lower()
    required_cols = []
    
    # Check for profit-related queries
    if any(term in query_lower for term in ['profit', 'margin', 'earning']):
        required_cols.append('profit')
    
    # Check for sales rep related queries
    if any(term in query_lower for term in ['sales rep', 'representative', 'agent']):
        required_cols.append('sales_rep_name')
    
    # Check for lead source related queries
    if any(term in query_lower for term in ['lead', 'source', 'origin']):
        required_cols.append('lead_source')
    
    # Check for vehicle related queries
    if any(term in query_lower for term in ['vehicle', 'car', 'make', 'model']):
        vehicle_cols = ['vehicle_make', 'vehicle_model', 'vehicle_year']
        for col in vehicle_cols:
            if col in df.columns:
                required_cols.append(col)
    
    # Check for time/date related queries
    if any(term in query_lower for term in ['time', 'days', 'close', 'date']):
        if 'days_to_close' in df.columns:
            required_cols.append('days_to_close')
    
    # Check for price related queries
    if any(term in query_lower for term in ['price', 'cost', 'value']):
        price_cols = ['listing_price', 'sold_price']
        for col in price_cols:
            if col in df.columns:
                required_cols.append(col)
    
    # For each word in the query, look for matching or similar column names
    words = re.findall(r'\b\w+\b', query_lower)
    for word in words:
        if len(word) < 3:  # Skip short words
            continue
            
        # Check for direct column matches
        for col in df.columns:
            col_lower = col.lower()
            if word in col_lower or any(word in variant for variant in expand_variants(col_lower)):
                required_cols.append(col)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(required_cols))

def build_enhanced_context(df: pd.DataFrame, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build enhanced context combining schema information and validation context.
    This enriched context helps the LLM understand both data structure and query intent.
    """
    # Start with basic data info
    enhanced_context = {
        "columns": df.columns.tolist(),
        "record_count": len(df),
        "data_types": {col: str(df[col].dtype) for col in df.columns},
        "query": query
    }
    
    # Add data quality metrics
    null_percentages = {col: (df[col].isnull().sum() / len(df)) * 100 for col in df.columns}
    enhanced_context["null_percentages"] = null_percentages
    enhanced_context["nan_percentage"] = sum(df.isnull().sum()) / (len(df) * len(df.columns)) * 100
    
    # Add example values for categorical columns
    example_values = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        try:
            # Get top 5 values for each categorical column
            example_values[col] = df[col].value_counts().head(5).index.tolist()
        except:
            pass
    enhanced_context["example_values"] = example_values
    
    # Add numeric column statistics
    numeric_stats = {}
    for col in df.select_dtypes(include=['number']).columns:
        try:
            numeric_stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median())
            }
        except:
            pass
    enhanced_context["numeric_stats"] = numeric_stats
    
    # Add date/time information if available
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_cols:
        date_ranges = {}
        for col in date_cols:
            try:
                date_ranges[col] = {
                    "min": df[col].min().isoformat(),
                    "max": df[col].max().isoformat()
                }
            except:
                pass
        enhanced_context["date_ranges"] = date_ranges
    
    # Generate potential column mappings
    column_mappings = {}
    for col in df.columns:
        variants = expand_variants(col)
        if variants:
            column_mappings[col] = variants[:5]  # Limit to 5 variants
    enhanced_context["column_mappings"] = column_mappings
    
    # Add context if provided
    if context:
        for key, value in context.items():
            if key not in enhanced_context and not key.startswith('_'):
                enhanced_context[key] = value
    
    return enhanced_context

def get_intent(llm_engine: LLMEngine, query: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get intent from query with error handling."""
    try:
        # Prepare context for intent detection
        intent_context = {
            "query": query,
            "columns": context["columns"],
            "data_types": context.get("data_types", {}),
            "example_values": context.get("example_values", {})
        }
        
        # Load and render prompt template
        prompt = llm_engine.load_prompt("intent_detection.tpl", intent_context)
        
        # Get LLM response
        response = llm_engine.client.chat.completions.create(
            model=llm_engine.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        # Parse response
        raw = response.choices[0].message.content
        if "{" not in raw:
            logger.warning("No JSON in LLM response for intent detection")
            return None
        
        try:
            intent_data = json.loads(raw)
            logger.info(f"Intent detected: {intent_data.get('intent', 'unknown')}")
            return intent_data
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse intent JSON: {je}")
            return None
        
    except Exception as e:
        logger.error(f"Error getting intent: {e}")
        return None

def handle_groupby(insight_functions: InsightFunctions, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
    """Handle groupby summary intent with visualization support."""
    try:
        # Find the actual column names in the DataFrame
        metric_col = find_column(df, intent.metric)
        category_col = find_column(df, intent.category)
        
        if not metric_col:
            logger.warning(f"Metric column '{intent.metric}' not found in DataFrame")
            return {
                "summary": f"Could not find a column matching '{intent.metric}' to analyze.",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Try specifying a valid column name"],
                "confidence": "low",
                "error_type": str(InsightErrorType.MISSING_COLUMNS)
            }
        
        if not category_col:
            logger.warning(f"Category column '{intent.category}' not found in DataFrame")
            return {
                "summary": f"Could not find a column matching '{intent.category}' to group by.",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Try specifying a valid column name for grouping"],
                "confidence": "low",
                "error_type": str(InsightErrorType.MISSING_COLUMNS)
            }
        
        # Call the insight function with validated column names
        result = insight_functions.groupby_summary(
            df,
            metric_col,
            category_col,
            intent.aggregation or "sum"
        )
        
        # Add metadata about the columns used
        result["_columns_used"] = {
            "metric": metric_col,
            "category": category_col,
            "aggregation": intent.aggregation or "sum"
        }
        
        # Add visualization if breakdown data is available
        if result.get("breakdown") and len(result["breakdown"]) > 0:
            try:
                chart_type = determine_chart_type(df, metric_col, category_col)
                chart_title = f"{metric_col} by {category_col}"
                
                # Create chart
                chart, chart_metadata = ChartRenderer.create_chart(
                    breakdown=result["breakdown"],
                    df=df,
                    chart_type=chart_type,
                    title=chart_title
                )
                
                # Add chart metadata to result
                if chart_metadata["success"]:
                    result["_visualization"] = {
                        "chart_spec": chart.to_dict() if chart else None,
                        "chart_type": chart_type,
                        "title": chart_title,
                        "metadata": chart_metadata
                    }
                else:
                    logger.warning(f"Chart creation failed: {chart_metadata.get('error')}")
                    
            except Exception as chart_err:
                logger.error(f"Error creating chart: {chart_err}")
                # Continue without chart if visualization fails
        
        return result
        
    except Exception as e:
        logger.error(f"Error in groupby summary: {e}")
        return {
            "summary": f"Error analyzing data: {str(e)}",
            "metrics": {},
            "breakdown": [],
            "recommendations": ["Try a different analysis approach"],
            "confidence": "low",
            "error_type": str(InsightErrorType.PROCESSING_ERROR)
        }

def determine_chart_type(df: pd.DataFrame, metric_col: str, category_col: str = None) -> str:
    """
    Determine the best chart type for the given data.
    
    Args:
        df: Source DataFrame
        metric_col: Metric column
        category_col: Optional category column
        
    Returns:
        str: Chart type recommendation
    """
    # Default to bar chart
    chart_type = "bar"
    
    try:
        # If no category column, default to a single metric summary
        if not category_col:
            return "bar"
            
        # Check cardinality of category column
        category_cardinality = df[category_col].nunique()
        
        # For time-based data, prefer line charts
        date_patterns = [r'date', r'time', r'year', r'month', r'quarter', r'week', r'day']
        is_date_category = any(re.search(pattern, category_col.lower()) for pattern in date_patterns)
        
        # For high-cardinality time series, use line chart
        if is_date_category:
            chart_type = "line"
            
        # For high-cardinality categories (non-date), use bar chart limited to top N
        elif category_cardinality > 10:
            chart_type = "bar"
            
        # For low-cardinality categories, prefer bar chart
        else:
            chart_type = "bar"
            
        # For percentage-based data, consider pie chart for low cardinality
        percent_patterns = [r'percent', r'ratio', r'share', r'proportion', r'distribution']
        is_percent_metric = any(re.search(pattern, metric_col.lower()) for pattern in percent_patterns)
        
        if is_percent_metric and category_cardinality <= 7:
            chart_type = "pie"
            
        return chart_type
        
    except Exception as e:
        logger.warning(f"Error determining chart type: {e}. Using default bar chart.")
        return "bar"

def handle_total(insight_functions: InsightFunctions, intent: IntentSchema, df: pd.DataFrame) -> Dict[str, Any]:
    """Handle total summary intent with enhanced visualization."""
    try:
        # Find the actual column name in the DataFrame
        metric_col = find_column(df, intent.metric)
        
        if not metric_col:
            logger.warning(f"Metric column '{intent.metric}' not found in DataFrame")
            return {
                "summary": f"Could not find a column matching '{intent.metric}' to analyze.",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Try specifying a valid column name"],
                "confidence": "low",
                "error_type": str(InsightErrorType.MISSING_COLUMNS)
            }
        
        # Call the insight function with validated column name
        result = insight_functions.total_summary(
            df,
            metric_col
        )
        
        # Add metadata about the column used
        result["_columns_used"] = {
            "metric": metric_col
        }
        
        # If there's time-based data available, add a trend analysis
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year'])]
        
        if date_cols and result.get("metrics"):
            try:
                # Get the first date column to use for time analysis
                date_col = date_cols[0]
                
                # Generate time-based breakdown data if not already present
                if not result.get("breakdown") or len(result.get("breakdown", [])) == 0:
                    # Use the DataFrame utilities for this operation
                    from ..utils.dataframe_utils import DataFrameUtils
                    
                    # Get breakdown data
                    breakdowns, metadata = DataFrameUtils.get_breakdown(
                        df=df,
                        group_by=date_col,
                        metric=metric_col,
                        agg_func="sum",
                        top_n=20,
                        handle_nulls=True
                    )
                    
                    if breakdowns:
                        result["breakdown"] = breakdowns
                        result["_breakdown_metadata"] = metadata
                
                # Add visualization if breakdown data is available
                if result.get("breakdown") and len(result.get("breakdown")) > 0:
                    chart_title = f"{metric_col} over time"
                    
                    # Create line chart for time-based data
                    chart, chart_metadata = ChartRenderer.create_chart(
                        breakdown=result["breakdown"],
                        df=df,
                        chart_type="line",
                        title=chart_title
                    )
                    
                    # Add chart metadata to result
                    if chart_metadata["success"]:
                        result["_visualization"] = {
                            "chart_spec": chart.to_dict() if chart else None,
                            "chart_type": "line",
                            "title": chart_title,
                            "metadata": chart_metadata
                        }
                    else:
                        logger.warning(f"Chart creation failed: {chart_metadata.get('error')}")
                
            except Exception as trend_err:
                logger.error(f"Error creating trend analysis: {trend_err}")
                # Continue without trend analysis if it fails
        
        return result
        
    except Exception as e:
        logger.error(f"Error in total summary: {e}")
        return {
            "summary": f"Error analyzing data: {str(e)}",
            "metrics": {},
            "breakdown": [],
            "recommendations": ["Try a different analysis approach"],
            "confidence": "low",
            "error_type": str(InsightErrorType.PROCESSING_ERROR)
        }

def check_data_quality(df: pd.DataFrame, required_cols: Optional[List[str]]) -> Tuple[bool, Dict[str, Any]]:
    """
    Enhanced data quality check that leverages the dedicated data validation module.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    # Import the dedicated data validation module
    from .data_validation import check_data_quality as data_validation_check
    
    # Use the dedicated validation function
    is_valid, validation_info = data_validation_check(df, required_cols)
    
    # Add additional metrics if needed
    if 'metrics' not in validation_info:
        validation_info['metrics'] = {}
        
    validation_info['metrics'].update({
        "record_count": len(df),
        "column_count": len(df.columns),
        "overall_null_percentage": sum(df.isnull().sum()) / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
    })
    
    return is_valid, validation_info