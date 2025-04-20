"""
LLM Engine for generating insights using OpenAI's API.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from ..config import OPENAI_API_KEY
from ..models import InsightResponse
from ..utils import parse_llm_response

logger = logging.getLogger(__name__)

# Configure OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up Jinja environment
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=False,
)

def sanitize_query(q: str) -> str:
    """Very basic: strip Jinja tokens & control chars."""
    cleaned = q.replace("{{", "").replace("{%", "")
    cleaned = re.sub(r'[^\w\s\?\.,]', "", cleaned)
    return cleaned.strip()

def build_prompt(columns: List[str], query: str, record_count: int = 0, 
                data_types: Optional[Dict[str, str]] = None) -> str:
    """Build a prompt using the template and provided data."""
    tpl = jinja_env.get_template("insight_generation.tpl")
    return tpl.render(
        columns=columns,
        query=query,
        record_count=record_count,
        data_types=data_types or {}
    )

class LLMEngine:
    """Handles LLM interactions for insight generation."""
    
    def __init__(self, use_mock: bool = False, api_key: Optional[str] = None):
        """Initialize the LLM engine."""
        self.use_mock = use_mock
        self.api_key = api_key or OPENAI_API_KEY
        self.model = "gpt-4"
        
        if not self.use_mock:
            if not self.api_key:
                logger.warning("No OpenAI API key found. Falling back to mock mode.")
                self.use_mock = True
            else:
                try:
                    self.client = OpenAI(api_key=self.api_key)
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.use_mock = True
    
    def load_prompt(self, template_name: str, context: dict) -> str:
        """
        Load and render a prompt template.
        
        Args:
            template_name: Name of the template file
            context: Dictionary of variables to render in the template
            
        Returns:
            Rendered prompt string
        """
        try:
            template = jinja_env.get_template(template_name)
            # Add empty default values for intent, metric, category, and aggregation
            # This ensures the template has values to render even when not provided
            render_context = context.copy()
            render_context.setdefault('intent', 'fallback')
            render_context.setdefault('metric', 'null')
            render_context.setdefault('category', 'null')
            render_context.setdefault('aggregation', 'null')
            
            rendered = template.render(**render_context)
            logger.debug(f"Loaded template '{template_name}' with context {list(context.keys())}")
            return rendered
        except Exception as e:
            logger.error(f"Error loading prompt template '{template_name}': {e}")
            # Fallback to a basic prompt if template loading fails
            return f"Please analyze the following: {context.get('query', '')}"
    
    def generate_insight(self, query: str, validation_context: Optional[Dict] = None) -> InsightResponse:
        """
        Generate an insight based on the query and validation context.
        
        Args:
            query: The user's query
            validation_context: Optional context including DataFrame info
            
        Returns:
            InsightResponse object containing the insight
        """
        try:
            # Return mock response if in mock mode
            if self.use_mock:
                return InsightResponse.mock_insight()
            
            # Get data from validation context
            df = validation_context.get('df') if validation_context else None
            
            # Sanitize the query
            safe_query = sanitize_query(query)
            
            # Prepare data for the prompt
            if df is not None and isinstance(df, pd.DataFrame):
                columns = list(df.columns)
                record_count = len(df)
                data_types = {col: str(df[col].dtype) for col in df.columns}
            else:
                columns = validation_context.get('columns', []) if validation_context else []
                record_count = validation_context.get('record_count', 0) if validation_context else 0
                data_types = validation_context.get('data_types', {}) if validation_context else {}
            
            # Build the prompt
            prompt = build_prompt(columns, safe_query, record_count, data_types)
            logger.info(f"LLM Prompt:\n{prompt}")
            
            # Call OpenAI API
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                
                # Extract the response content
                raw = response.choices[0].message.content
                logger.info(f"LLM Raw Response:\n{raw}")
                
                return parse_llm_response(raw)
                    
            except OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                return InsightResponse.error_response(f"API Error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error in generate_insight: {e}")
            return InsightResponse.error_response(str(e))

# Module-level function for direct use
def generate_insight(columns: List[str], question: str) -> InsightResponse:
    """
    Generate an insight using the LLM engine.
    
    Args:
        columns: List of column names
        question: The user's question
        
    Returns:
        Structured insight response
    """
    sanitized = sanitize_query(question)
    prompt = build_prompt(columns, sanitized)
    logger.info(f"LLM Prompt:\n{prompt}")
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content
        logger.info(f"LLM Raw Response:\n{raw}")
        return parse_llm_response(raw)
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return parse_llm_response("")  # will fallback to mock