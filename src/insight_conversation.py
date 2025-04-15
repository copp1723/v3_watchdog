"""
Module for managing conversation flow with LLM interactions.
Handles conversation history, prompt management, and insight generation.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from src.insight_card import InsightOutputFormatter, render_insight_card
from src.insight_flow import PromptGenerator, generate_llm_prompt

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

class ConversationManager:
    """Manages conversation flow and LLM interactions."""
    
    def __init__(self, schema: Optional[Dict[str, str]] = None, use_mock: bool = True):
        """
        Initialize the conversation manager.
        
        Args:
            schema: Dictionary mapping entity types to their descriptions
            use_mock: Whether to use mock responses for testing
        """
        self.schema = schema or {}  # Use empty dict if schema is None
        self.use_mock = use_mock
        self.formatter = InsightOutputFormatter()
        self.client = None # Initialize client attribute
        # TODO: Initialize real LLM client if use_mock is False
        # Example:
        # if not self.use_mock:
        #     from openai import OpenAI
        #     self.client = OpenAI() # Requires OPENAI_API_KEY env var
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'regenerate_insight' not in st.session_state:
            st.session_state['regenerate_insight'] = False
    
    def get_mock_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a mock response for testing.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Mock response dictionary
        """
        return {
            "summary": f"Mock insight for: {prompt}",
            "chart_data": {
                "type": "line",
                "data": {"x": [1, 2, 3], "y": [10, 20, 30]},
                "title": "Sample Trend"
            },
            "recommendation": "Consider reviewing the data in more detail.",
            "risk_flag": False
        }
    
    def generate_insight(self, prompt: str, add_to_history: bool = True) -> Dict[str, Any]:
        """
        Generate an insight based on the prompt.
        
        Args:
            prompt: The input prompt
            add_to_history: Whether to add the generated insight to history.
            
        Returns:
            Formatted insight dictionary
        """
        # Get raw response
        if self.use_mock:
            raw_response = self.get_mock_response(prompt)
        else:
            # TODO: Implement real LLM call using self.client
            # raw_response = self.client.chat.completions.create(...)
            raw_response = self.get_mock_response(f"LLM response for: {prompt}") # Placeholder
        
        # Format and validate response
        formatted_response = self.formatter.format_output(raw_response)
        
        # Add metadata
        formatted_response['timestamp'] = datetime.now().isoformat()
        formatted_response['prompt'] = prompt
        
        # Store in conversation history if requested
        if add_to_history:
            st.session_state['conversation_history'].append({
                'prompt': prompt,
                'response': formatted_response,
                'timestamp': formatted_response['timestamp']
            })
        
        return formatted_response
    
    def regenerate_insight(self, original_prompt_index: int) -> Optional[Dict[str, Any]]:
        """
        Regenerate an insight for a previous prompt index with enhanced context.
        Does NOT add the result to history automatically.

        Args:
            original_prompt_index: The index in the conversation history to regenerate.
            
        Returns:
            New formatted insight dictionary or None if index is invalid.
        """
        if not 0 <= original_prompt_index < len(st.session_state['conversation_history']):
            st.error(f"Invalid index {original_prompt_index} for regeneration.")
            return None
            
        original_entry = st.session_state['conversation_history'][original_prompt_index]
        original_prompt = original_entry['prompt']
        
        # Get previous insights for context (up to the one being regenerated)
        previous_insights = [entry['response'] for entry in st.session_state['conversation_history'][:original_prompt_index+1]]
        
        # Generate enhanced prompt
        enhanced_prompt = generate_llm_prompt(original_prompt, {}, previous_insights)
        
        # Generate new insight without adding to history
        new_insight = self.generate_insight(enhanced_prompt, add_to_history=False)
        return new_insight
    
    def render_conversation(self) -> None:
        """Render the conversation history in the UI."""
        render_conversation_history(st.session_state['conversation_history'])

def render_conversation_interface(schema: Dict[str, str], use_mock: bool = True) -> None:
    """
    Render the main conversation interface.
    
    Args:
        schema: Dictionary mapping entity types to their descriptions
        use_mock: Whether to use mock responses
    """
    manager = ConversationManager(schema, use_mock)
    
    # Input prompt
    prompt = st.text_input("Enter your prompt:", key="prompt_input")
    
    if prompt:
        st.session_state['current_prompt'] = prompt
    
    # Generate or regenerate insight
    if st.session_state['current_prompt']:
        if st.session_state['regenerate_insight']:
            response = manager.regenerate_insight(st.session_state['current_prompt'])
            st.session_state['regenerate_insight'] = False
        else:
            response = manager.generate_insight(st.session_state['current_prompt'])
    
    # Render conversation history
    manager.render_conversation() 