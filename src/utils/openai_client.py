"""
Module for managing OpenAI API interactions.
Provides a centralized client and helper functions for OpenAI API calls.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def get_openai_client():
    """
    Get an initialized OpenAI client.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If API key is not found or client initialization fails
    """
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
            
        openai_version = getattr(openai, "__version__", "unknown")
        logger.debug(f"OpenAI library version: {openai_version}")
        
        # Initialize client based on version
        if openai_version.startswith("0."):
            logger.debug("Using OpenAI v0.x API style")
            openai.api_key = api_key
            return openai
        else:
            logger.debug("Using OpenAI v1.x API style")
            from openai import OpenAI
            return OpenAI(api_key=api_key)
            
    except ImportError as e:
        logger.error(f"Failed to import openai library: {e}")
        raise ValueError(f"OpenAI library not found. Please install it with 'pip install openai'")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise ValueError(f"Failed to initialize OpenAI client: {e}")

def generate_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> Any:
    """
    Generate a completion using the OpenAI API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model to use for completion
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Completion response from OpenAI
        
    Raises:
        ValueError: If API call fails
    """
    try:
        client = get_openai_client()
        openai_version = getattr(client, "__version__", "unknown")
        
        if openai_version.startswith("0."):
            return client.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise ValueError(f"Failed to generate completion: {e}") 