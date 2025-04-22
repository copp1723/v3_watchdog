#!/usr/bin/env python
"""
Insight Generator for the Watchdog AI system.

This module provides functionality to generate data-driven insights from
analytics summaries using OpenAI's API. It handles prompt construction,
API interaction, response validation, and error handling.
"""

import json
import logging
import time
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from functools import wraps
import tiktoken
import jsonschema
import backoff
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .prompts import (
    build_prompt,
    validate_insight_response,
    INSIGHT_RESPONSE_SCHEMA
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API cost mapping ($/1K tokens)
OPENAI_COST_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.01, "output": 0.03},  # Example rates, adjust with actual pricing
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
}

# Default OpenAI model to use
DEFAULT_MODEL = "gpt-4o"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 1
MAX_BACKOFF = 60

# Token limits
MAX_TOKENS_GPT4O = 128000  # Maximum context length for GPT-4o
MAX_INPUT_TOKENS = int(MAX_TOKENS_GPT4O * 0.85)  # Reserve 15% for response
MAX_OUTPUT_TOKENS = 4096  # Maximum tokens for the response


class InsightGenerationError(Exception):
    """Base exception for all insight generation errors."""
    pass


class InputValidationError(InsightGenerationError):
    """Exception raised for input validation errors."""
    pass


class TokenLimitError(InsightGenerationError):
    """Exception raised when input exceeds token limits."""
    pass


class APIError(InsightGenerationError):
    """Exception raised for OpenAI API errors."""
    pass


class ResponseValidationError(InsightGenerationError):
    """Exception raised when response validation fails."""
    pass


def retry_with_exponential_backoff(max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """
    Decorator for implementing retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries before giving up
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        The decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff_time = initial_backoff
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (APIError, Exception) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {str(e)}")
                        raise
                    
                    # Exponential backoff with some jitter
                    jitter = 0.1 * backoff_time * (2 * (0.5 - time.time() % 1))
                    sleep_time = backoff_time + jitter
                    logger.warning(
                        f"API call failed with error: {str(e)}. "
                        f"Retrying in {sleep_time:.2f} seconds... (Attempt {retries+1}/{max_retries})"
                    )
                    
                    time.sleep(sleep_time)
                    backoff_time = min(backoff_time * 2, MAX_BACKOFF)
            
            # This should never be reached due to the raise in the loop
            return None
        return wrapper
    return decorator


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The input text
        model: The model name to use for tokenization
        
    Returns:
        The number of tokens in the text
    """
    try:
        # Map model names to encoding names
        encoding_name = "cl100k_base"  # Default for current OpenAI models
        encoder = tiktoken.get_encoding(encoding_name)
        return len(encoder.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}. Using approximate count.")
        # Fallback to approximate count (rough estimate)
        return len(text) // 4


def sanitize_input(text: str) -> str:
    """
    Sanitize input text by removing control characters and normalizing whitespace.
    
    Args:
        text: The input text to sanitize
        
    Returns:
        The sanitized text
    """
    if not text:
        return ""
    
    # Remove control characters except for \n and \t
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace (convert multiple spaces to single space)
    text = re.sub(r' +', ' ', text)
    
    # Trim leading/trailing whitespace
    return text.strip()


def truncate_or_summarize(text: str, max_tokens: int, model: str = DEFAULT_MODEL) -> str:
    """
    Truncate or summarize text to fit within token limits.
    
    Args:
        text: The input text
        max_tokens: Maximum allowed tokens
        model: The model name for tokenization
        
    Returns:
        The truncated or summarized text
    """
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    # For simple truncation, we just cut off at approximate char position
    # In a real implementation, a more sophisticated summarization would be used
    chars_per_token = len(text) / current_tokens
    approximate_chars = int(max_tokens * chars_per_token * 0.9)  # 10% safety margin
    
    truncated = text[:approximate_chars]
    
    # Add a note about truncation
    truncation_note = "\n\n[Note: This data has been truncated to fit within token limits.]"
    
    return truncated + truncation_note


class InsightGenerator:
    """
    Generator for data-driven insights using OpenAI's API.
    
    This class handles the generation of insights from analytics data by
    constructing prompts, calling the OpenAI API, and validating responses.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None
    ):
        """
        Initialize the InsightGenerator.
        
        Args:
            api_key: OpenAI API key (optional, defaults to environment variable)
            model: The OpenAI model to use
            max_output_tokens: Maximum tokens for the response
            log_to_file: Whether to log costs to a file
            log_file_path: Path to the log file
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or "openai_cost_log.csv"
        
        if log_to_file and not api_key:
            logger.warning(
                "Cost logging enabled but API key not provided. "
                "Cost logging may not be accurate."
            )
    
    def _log_usage_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        request_id: str
    ) -> None:
        """
        Log the usage cost of an API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: The model used
            request_id: The request ID
        """
        # Calculate cost
        model_costs = OPENAI_COST_PER_1K_TOKENS.get(
            model,
            {"input": 0.01, "output": 0.03}  # Default if unknown
        )
        
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        total_cost = input_cost + output_cost
        
        # Log to console
        logger.info(
            f"OpenAI API call cost - "
            f"Model: {model}, "
            f"Input: {input_tokens} tokens (${input_cost:.4f}), "
            f"Output: {output_tokens} tokens (${output_cost:.4f}), "
            f"Total: ${total_cost:.4f}"
        )
        
        # Log to file if enabled
        if self.log_to_file:
            import csv
            import os
            from datetime import datetime
            
            timestamp = datetime.now().isoformat()
            file_exists = os.path.isfile(self.log_file_path)
            
            try:
                with open(self.log_file_path, 'a', newline='') as csvfile:
                    fieldnames = [
                        'timestamp', 'request_id', 'model',
                        'input_tokens', 'output_tokens',
                        'input_cost', 'output_cost', 'total_cost'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerow({
                        'timestamp': timestamp,
                        'request_id': request_id,
                        'model': model,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'input_cost': f"${input_cost:.4f}",
                        'output_cost': f"${output_cost:.4f}",
                        'total_cost': f"${total_cost:.4f}"
                    })
            except Exception as e:
                logger.error(f"Failed to log cost to file: {str(e)}")
    
    @retry_with_exponential_backoff()
    def _call_openai(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Call the OpenAI API with retry logic.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            A tuple of (response_text, usage_info)
            
        Raises:
            APIError: If the API call fails after retries
        """
        try:
            input_tokens = count_tokens(prompt, self.model)
            logger.info(f"Sending request to OpenAI API (model: {self.model}, input tokens: {input_tokens})")
            
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=self.max_output_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            response_text = response.choices[0].message.content
            usage_info = {
                "input_tokens": input_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "request_id": response.id
            }
            
            # Log usage cost
            self._log_usage_cost(
                input_tokens=input_tokens,
                output_tokens=response.usage.completion_tokens,
                model=self.model,
                request_id=response.id
            )
            
            return response_text, usage_info
            
        except Exception as e:
            error_msg = f"OpenAI API call failed: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg) from e
    
    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate the API response.
        
        Args:
            response_text: The raw response text from the API
            
        Returns:
            The parsed and validated response as a dictionary
            
        Raises:
            ResponseValidationError: If parsing or validation fails
        """
        # Extract JSON from response (sometimes LLM adds markdown code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
        
        try:
            parsed_response = json.loads(json_str)
            
            # Validate response against schema
            validate_insight_response(parsed_response)
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response as JSON: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Raw response: {response_text}")
            raise ResponseValidationError(error_msg) from e
            
        except jsonschema.exceptions.ValidationError as e:
            error_msg = f"Response validation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Parsed response: {json_str}")
            raise ResponseValidationError(error_msg) from e
    
    def generate_insight(
        self,
        analytics_summary: Union[str, Dict[str, Any]],
        query: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an insight based on analytics data and a query.
        
        Args:
            analytics_summary: The analytics data as a JSON string or dictionary
            query: The specific question or analysis request
            custom_params: Optional dictionary of custom parameters for the analysis
                This can include:
                    - focus_metric: The specific metric to focus on
                    - timeframe: The time period to analyze
                    - chart_type: Override the default chart type
                    - depth: Level of detail in the analysis
                    
        Returns:
            A dictionary containing the generated insight
            
        Raises:
            InputValidationError: If input validation fails
            TokenLimitError: If input exceeds token limits
            APIError: If the OpenAI API call fails
            ResponseValidationError: If response validation fails
            InsightGenerationError: If any other error occurs
        """
        try:
            # Validate and sanitize inputs
            if not query or not analytics_summary:
                raise InputValidationError("Query and analytics summary are required")
            
            # Sanitize query
            sanitized_query = sanitize_input(query)
            if not sanitized_query:
                raise InputValidationError("Query is empty after sanitization")
            
            # Convert analytics_summary to string if it's a dictionary
            if isinstance(analytics_summary, dict):
                try:
                    analytics_json = json.dumps(analytics_summary, indent=2)
                except Exception as e:
                    raise InputValidationError(f"Failed to serialize analytics data: {str(e)}") from e
            else:
                analytics_json = str(analytics_summary)
                # Try to validate it's proper JSON
                try:
                    json.loads(analytics_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Analytics summary is not valid JSON: {str(e)}")
                    # We'll still try to use it, but it might cause issues
            
            # Sanitize analytics JSON
            sanitized_analytics = sanitize_input(analytics_json)
            
            # Check token limits
            max_input_tokens = MAX_INPUT_TOKENS  # This is 85% of the model's context window
            
            # Estimate tokens for the complete prompt
            estimated_prompt_tokens = count_tokens(
                build_prompt("", sanitized_query, custom_params),
                self.model
            )
            
            # Calculate how many tokens we have left for the analytics data
            max_analytics_tokens = max_input_tokens - estimated_prompt_tokens
            
            if max_analytics_tokens <= 0:
                raise TokenLimitError(
                    "Query and custom parameters are too large, "
                    "no room left for analytics data"
                )
            
            # Truncate analytics data if needed
            sanitized_analytics = truncate_or_summarize(
                sanitized_analytics,
                max_analytics_tokens,
                self.model
            )
            
            # Build the prompt
            prompt = build_prompt(
                sanitized_analytics, 
                sanitized_query, 
                custom_params
            )
            
            # Final token check
            final_token_count = count_tokens(prompt, self.model)
            if final_token_count > max_input_tokens:
                raise TokenLimitError(
                    f"Generated prompt exceeds token limit: {final_token_count} > {max_input_tokens}"
                )
            
            logger.info(f"Generated prompt with {final_token_count} tokens")
            
            # Call OpenAI API
            response_text, usage_info = self._call_openai(prompt)
            
            # Parse and validate response
            try:
                insight = self._parse_and_validate_response(response_text)
                
                # Add metadata to the insight
                insight["_metadata"] = {
                    "usage": usage_info,
                    "model": self.model,
                    "timestamp": time.time()
                }
                
                return insight
                
            except ResponseValidationError as e:
                # Attempt to fix the response with a follow-up request
                logger.warning(f"Attempting to fix invalid response: {str(e)}")
                fixed_insight = self._attempt_response_fix(response_text, e)
                
                # Add metadata to the fixed insight
                fixed_insight["_metadata"] = {
                    "usage": usage_info,
                    "model": self.model,
                    "timestamp": time.time(),
                    "fixed": True
                }
                
                return fixed_insight
                
        except InsightGenerationError:
            # Re-raise any of our custom exceptions
            raise
        except Exception as e:
            # Wrap any other exceptions
            error_msg = f"Unexpected error generating insight: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise InsightGenerationError(error_msg) from e
    
    def _attempt_response_fix(
        self, 
        original_response: str, 
        validation_error: ResponseValidationError
    ) -> Dict[str, Any]:
        """
        Attempt to fix an invalid response by sending a follow-up request.
        
        Args:
            original_response: The original invalid response
            validation_error: The validation error that occurred
            
        Returns:
            A fixed and validated response
            
        Raises:
            ResponseValidationError: If the fix attempt also fails
        """
        try:
            # Create a fix prompt
            fix_prompt = f"""
You previously provided a response that couldn't be properly parsed or validated.

Your original response:
{original_response}

The error was: {str(validation_error)}

Please provide a corrected response that strictly follows this JSON schema:

```json
{json.dumps(INSIGHT_RESPONSE_SCHEMA, indent=2)}
```

Return ONLY the valid JSON with no additional text, markdown formatting, or explanation.
"""
            
            # Call OpenAI API with the fix prompt
            fixed_response_text, _ = self._call_openai(fix_prompt)
            
            # Parse and validate the fixed response
            fixed_insight = self._parse_and_validate_response(fixed_response_text)
            
            logger.info("Successfully fixed invalid response")
            return fixed_insight
            
        except Exception as e:
            error_msg = f"Failed to fix invalid response: {str(e)}"
            logger.error(error_msg)
            raise ResponseValidationError(error_msg) from e
    
    def generate_custom_report(
        self,
        analytics_summary: Union[str, Dict[str, Any]],
        query: str,
        report_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a custom report with specific parameters.
        
        This is a convenience method that wraps generate_insight with report-specific
        custom parameters.
        
        Args:
            analytics_summary: The analytics data as a JSON string or dictionary
            query: The specific question or analysis request
            report_params: Dictionary of report parameters:
                - time_period: The time period to analyze (e.g., "last_month", "last_quarter")
                - metrics: List of metrics to focus on
                - comparisons: What to compare against (e.g., "previous_period", "target")
                - segments: Specific customer segments to analyze
                - chart_type: Type of chart to use
                
        Returns:
            A dictionary containing the generated insight
        """
        # Validate report parameters
        if not isinstance(report_params, dict):
            raise InputValidationError("Report parameters must be a dictionary")
        
        # Convert report parameters to custom parameters
        custom_params = {
            "report_type": "custom",
            **report_params
        }
        
        # Add report-specific context to the query
        enhanced_query = f"Custom Report - {query}"
        if "time_period" in report_params:
            enhanced_query += f" for {report_params['time_period']}"
        
        return self.generate_insight(
            analytics_summary,
            enhanced_query,
            custom_params
        )

