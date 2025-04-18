"""
Module for managing conversation flow with LLM interactions.
Handles conversation history, prompt management, and insight generation.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
import os
import pandas as pd
from datetime import datetime
# import requests # Removed unused import
# import numpy as np # Removed unused import
from src.insight_card import InsightOutputFormatter, render_insight_card
from src.insight_flow import PromptGenerator, generate_llm_prompt
from src.insight_templates import TemplateManager, InsightTemplate

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
    for entry in reversed(history):
        with st.container():
            st.markdown(f"**Prompt:** {entry['prompt']}")
            render_insight_card(entry['response'], show_buttons=show_buttons)
            st.markdown("---")

def _load_system_prompt(filepath="automotive_analyst_prompt.md") -> str:
    """Loads the system prompt from a specified file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[ERROR] System prompt file not found at {filepath}. Using default fallback.")
        # Fallback prompt if file is missing
        return (
            "You are a helpful data analyst. Analyze the provided context and user question. "
            "Respond ONLY in valid JSON format: "
            '{"summary": "<summary>", "value_insights": [], "actionable_flags": [], "confidence": "medium"}. '
        )
    except Exception as e:
        print(f"[ERROR] Failed to read system prompt file {filepath}: {e}. Using default fallback.")
        return (
            "You are a helpful data analyst. Analyze the provided context and user question. "
            "Respond ONLY in valid JSON format: "
            '{"summary": "<summary>", "value_insights": [], "actionable_flags": [], "confidence": "medium"}. '
        )

class ConversationManager:
    """Manages conversation flow and LLM interactions."""
    
    def __init__(self, schema: Optional[Dict[str, str]] = None, use_mock: bool = None):
        """
        Initialize the conversation manager.
        """
        # FINAL robust use_mock logic
        env_mock = os.getenv("USE_MOCK", "true").strip().lower() in ["true", "1", "yes"]
        self.use_mock = use_mock if use_mock is not None else env_mock
        self.schema = schema or {}  # Use empty dict if schema is None
        self.formatter = InsightOutputFormatter()
        # LLM client settings
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI
        print(f"[DEBUG] Loaded API Key (first 5 chars): {str(self.api_key)[:5] if self.api_key else 'None'}")
        # Debug prints for LLM mode
        print(f"[DEBUG] Initializing ConversationManager...")
        print(f"[DEBUG] Explicit use_mock param: {use_mock}")
        print(f"[DEBUG] Loaded USE_MOCK env var: {os.getenv('USE_MOCK')}")
        print(f"[DEBUG] Final self.use_mock: {self.use_mock}")
        print(f"[DEBUG] LLM_PROVIDER env var: {os.getenv('LLM_PROVIDER')}")
        print(f"[DEBUG] Final self.llm_provider: {self.llm_provider}")
        
        # Initialize client based on provider
        self.client = None
        if not self.use_mock:
            print(f"[DEBUG] Attempting to initialize LLM client: {self.llm_provider}")
            if self.llm_provider == "openai" and self.api_key:
                try:
                    print("[DEBUG] Attempting to import openai...")
                    import openai
                    openai_version = getattr(openai, "__version__", "unknown")
                    print(f"[DEBUG] OpenAI library version: {openai_version}")
                    
                    # Set the API key
                    if hasattr(openai, "api_key"):
                        # For older versions (0.x)
                        openai.api_key = self.api_key
                        print("[DEBUG] Set API key using openai.api_key (older style)")
                    else:
                        # For newer versions (1.x+)
                        print("[DEBUG] Setting up OpenAI client with newer API style")
                        openai.Client = openai.OpenAI
                        self.client = openai.Client(api_key=self.api_key)
                    
                    # For older versions, use the module as the client
                    if not self.client:
                        self.client = openai
                    
                    print(f"[DEBUG] OpenAI client initialized successfully. Client type: {type(self.client)}")
                    # Test that the client works
                    print("[DEBUG] Testing API key validity...")
                    try:
                        if hasattr(openai, "ChatCompletion"):
                            # Older API
                            models = openai.Model.list()
                            print(f"[DEBUG] API key is valid. Found {len(models.data) if hasattr(models, 'data') else 'some'} models.")
                        else:
                            # Newer API
                            models = self.client.models.list()
                            print(f"[DEBUG] API key is valid. Found {len(models.data) if hasattr(models, 'data') else 'some'} models.")
                    except Exception as e:
                        print(f"[ERROR] API key validation failed: {e}")
                        self.use_mock = True  # API key isn't working, force mock mode
                        
                except ImportError as import_err:
                    st.warning(f"Failed to import openai library: {import_err}. Install it (`pip install openai`). Falling back to mock.")
                    print(f"[ERROR] ImportError for openai: {import_err}")
                    self.use_mock = True # Force mock if import fails
            elif self.llm_provider == "anthropic" and self.api_key:
                self.client = "anthropic" # Placeholder, actual client uses requests
                print(f"[DEBUG] Anthropic client mode set.")
            else:
                print(f"[DEBUG] Conditions not met for LLM client init (provider: {self.llm_provider}, api_key present: {self.api_key is not None})")
        else:
            print("[DEBUG] Skipping LLM client initialization because use_mock is True.") # Debug print
        
        # ADDED: Final check at end of __init__
        print(f"[DEBUG] State AT END of __init__: use_mock={self.use_mock}, client_type={type(self.client)}, api_key_present={self.api_key is not None}")
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'regenerate_insight' not in st.session_state:
            st.session_state['regenerate_insight'] = False 