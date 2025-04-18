"""
Celery tasks for async insight generation.

This module contains Celery tasks that handle async processing of insights.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import sentry_sdk
from celery import Task, shared_task
import json
import io

from ..celery_app import app
from .intent_manager import intent_manager
from .models import InsightResult
from .engine import InsightEngine
from ..utils.data_io import load_data, validate_data
from ..utils.errors import InsightGenerationError

logger = logging.getLogger(__name__)

class BaseInsightTask(Task):
    """Base task for insight generation with error handling and metrics."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handler for task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        sentry_sdk.capture_exception(exc)
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handler for task success."""
        logger.info(f"Task {task_id} completed successfully")
        super().on_success(retval, task_id, args, kwargs)

@shared_task(bind=True, base=BaseInsightTask, name="src.insights.tasks.generate_insight")
def generate_insight(self, prompt: str, serialized_df: str) -> Dict[str, Any]:
    """
    Generate insights asynchronously based on the prompt and data.
    
    Args:
        prompt: The user's prompt
        serialized_df: JSON string of serialized DataFrame
        
    Returns:
        Dictionary containing the insight result
    """
    try:
        logger.info(f"Generating insight for prompt: {prompt[:50]}...")
        
        # Deserialize DataFrame
        df = pd.read_json(serialized_df, orient='split')
        
        # Input validation
        if not prompt or not isinstance(prompt, str):
            error_msg = "Invalid prompt: Prompt must be a non-empty string"
            logger.error(error_msg)
            return {
                "summary": "Failed to generate insight",
                "error": error_msg,
                "error_type": "input_validation",
                "timestamp": datetime.now().isoformat(),
                "is_error": True,
                "is_direct_calculation": True,
                "task_id": self.request.id
            }
        
        if df is None or df.empty:
            error_msg = "No data available for analysis"
            logger.error(error_msg)
            return {
                "summary": error_msg,
                "error": error_msg,
                "error_type": "missing_data",
                "timestamp": datetime.now().isoformat(),
                "is_error": True,
                "is_direct_calculation": True,
                "task_id": self.request.id
            }
        
        # Generate insight using intent manager
        response = intent_manager.generate_insight(prompt, df)
        
        # Add timestamp if not present
        if "timestamp" not in response:
            response["timestamp"] = datetime.now().isoformat()
        
        # Add task ID for tracking
        response["task_id"] = self.request.id
        
        return response
        
    except Exception as e:
        error_msg = f"Error generating insight: {str(e)}"
        logger.error(error_msg)
        sentry_sdk.capture_exception(e)
        return {
            "summary": "Failed to generate insight",
            "error": str(e),
            "error_type": "unknown",
            "timestamp": datetime.now().isoformat(),
            "is_error": True,
            "is_direct_calculation": True,
            "task_id": self.request.id
        }

@shared_task(bind=True, base=BaseInsightTask, name="src.insights.tasks.run_insight_pipeline")
def run_insight_pipeline(self, uploaded_file_data: bytes, filename: str, llm_client_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full insight pipeline on uploaded data asynchronously.
    
    Args:
        uploaded_file_data: Binary data from the uploaded file
        filename: Name of the uploaded file
        llm_client_config: Configuration for the LLM client
        
    Returns:
        Dictionary containing processed data and insights
    """
    try:
        # Create file-like object from binary data
        file_obj = io.BytesIO(uploaded_file_data)
        file_obj.name = filename
        
        # Initialize LLM client based on config
        if llm_client_config.get("provider") == "openai":
            from ..utils.openai_client import OpenAIClient
            llm_client = OpenAIClient(
                api_key=llm_client_config.get("api_key"),
                model=llm_client_config.get("model", "gpt-4"),
                temperature=llm_client_config.get("temperature", 0.0)
            )
        else:
            # Use mock client for testing
            from unittest.mock import MagicMock
            llm_client = MagicMock()
            
        # Create insight engine
        engine = InsightEngine(llm_client)
        
        # Initialize session tracking
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        sentry_sdk.set_tag("session_id", session_id)
        
        # Step 1: Load & validate data
        sentry_sdk.set_tag("pipeline_step", "load_data")
        logger.info("Loading and validating data...")
        
        df = load_data(file_obj)
        df, validation_summary = validate_data(df)
        
        sentry_sdk.set_tag("data_rows", len(df))
        sentry_sdk.set_tag("data_quality_score", validation_summary["quality_score"])
        
        # Run insight generation pipeline
        result = engine.run_impl(df)
        
        # Add task metadata
        result["metadata"]["task_id"] = self.request.id
        result["metadata"]["celery_task"] = True
        
        # Serialize DataFrame to JSON for response
        result["data"] = df.to_json(orient='split')
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sentry_sdk.capture_exception(e)
        return {
            "error": str(e),
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat()
        }