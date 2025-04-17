"""
Module for managing conversation flow with LLM interactions.
Handles conversation history, prompt management, and insight generation.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
import os
import traceback
import pandas as pd
from datetime import datetime
import logging

from src.insight_card import InsightOutputFormatter
from src.insight_flow import PromptGenerator
from src.utils.openai_client import get_openai_client, generate_completion
from src.insights.intent_manager import intent_manager

logger = logging.getLogger(__name__)

def _load_system_prompt(filepath="automotive_analyst_prompt.md") -> str:
    """Loads the system prompt from a specified file."""
    try:
        # Try relative path first
        script_dir = os.path.dirname(__file__)
        rel_path = os.path.join(script_dir, '..', filepath)
        if os.path.exists(rel_path):
            filepath = rel_path
        elif not os.path.exists(filepath):
            # Try project root path as fallback
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            root_path = os.path.join(project_root, filepath)
            if os.path.exists(root_path):
                 filepath = root_path
            else:
                 # If still not found, check one level up from src for prompt_templates
                 prompts_dir = os.path.join(project_root, 'prompt_templates')
                 template_path = os.path.join(prompts_dir, os.path.basename(filepath))
                 if os.path.exists(template_path):
                      filepath = template_path
                 else:
                      # Fallback to default if not found anywhere sensible
                      raise FileNotFoundError(f"System prompt not found near src or in project root: {os.path.basename(filepath)}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Using default fallback system prompt.")
        # Fallback prompt if file is missing
        return (
            "You are a helpful data analyst. Analyze the provided context and user question. "
            "Respond ONLY in valid JSON format: "
            '{"summary": "<summary>", "value_insights": [], "actionable_flags": [], "confidence": "medium"}. '
        )
    except Exception as e:
        print(f"[ERROR] Error loading system prompt: {e}")
        return (
            "You are a helpful data analyst. Analyze the provided context and user question. "
            "Respond ONLY in valid JSON format: "
            '{"summary": "<summary>", "value_insights": [], "actionable_flags": [], "confidence": "medium"}. '
        )

def _generate_fallback_response(prompt: str, available_intents: List[str]) -> Dict[str, Any]:
    """Generate a helpful fallback response when no intent matches."""
    return {
        "title": "Unsupported Analysis Type",
        "summary": (
            "I'm not sure how to analyze that. Currently, I can help with:\n"
            "• Finding highest/top values of metrics\n"
            "• Finding lowest/bottom values of metrics\n"
            "• Calculating averages/means of metrics\n"
            "• Counting specific data points (e.g., negative profits)\n"
            "• Analyzing profit patterns\n\n"
            "Try asking about one of these!"
        ),
        "value_insights": [
            "Example queries:",
            "• What lead source had the highest gross profit?",
            "• Show me the sales rep with the lowest revenue",
            "• What's the average price by make?",
            "• How many deals had negative profit?",
            "• Analyze deals with negative gross"
        ],
        "actionable_flags": [],
        "confidence": "low",
        "timestamp": datetime.now().isoformat(),
        "is_error": False
    }

class ConversationManager:
    """Manages conversation flow and LLM interactions."""

    def __init__(self, schema: Optional[Dict[str, str]] = None, use_mock: bool = None):
        """Initialize the conversation manager."""
        # FINAL robust use_mock logic
        env_mock = os.getenv("USE_MOCK", "true").strip().lower() in ["true", "1", "yes"]
        self.use_mock = use_mock if use_mock is not None else env_mock
        self.schema = schema or {}  # Use empty dict if schema is None
        self.formatter = InsightOutputFormatter()
        
        # LLM client settings
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI
        
        logger.debug(f"Initializing ConversationManager with provider: {self.llm_provider}")
        logger.debug(f"Mock mode: {self.use_mock}")
        
        # Initialize client based on provider
        self.client = None
        if not self.use_mock:
            if self.llm_provider == "openai" and self.api_key:
                try:
                    self.client = get_openai_client()
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    st.warning(f"Could not initialize OpenAI client: {e}. Check API key and version. Falling back to mock.")
                    self.use_mock = True
            elif self.llm_provider == "anthropic" and self.api_key:
                self.client = "anthropic"  # Placeholder, actual client uses requests
                logger.info("Anthropic client mode set")
            else:
                logger.warning(f"Conditions not met for LLM client init (provider: {self.llm_provider}, api_key present: {self.api_key is not None})")
        else:
            logger.info("Skipping LLM client initialization because use_mock is True")

        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'current_prompt' not in st.session_state:
            st.session_state['current_prompt'] = None
        if 'regenerate_insight' not in st.session_state:
            st.session_state['regenerate_insight'] = False

    def generate_insight(self, prompt: str, validation_context: Dict[str, Any] = None, add_to_history: bool = True) -> Dict[str, Any]:
        """
        Generate a new insight based on the user prompt and validation context.
        
        Args:
            prompt: The user's prompt
            validation_context: Context including DataFrame and validation info
            add_to_history: Whether to add this interaction to history
            
        Returns:
            The insight response
        """
        try:
            # Input validation
            if not prompt or not isinstance(prompt, str):
                error_msg = "Invalid prompt: Prompt must be a non-empty string"
                logger.error(error_msg)
                return {
                    "summary": "Failed to generate insight",
                    "error": error_msg,
                    "error_type": "input_validation",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True
                }

            logger.info(f"Generating insight for prompt: {prompt[:50]}...")
            
            # Get DataFrame from validation context
            df = validation_context.get('df') if validation_context else None
            
            if df is None:
                return {
                    "summary": "No data available for analysis",
                    "error": "Missing DataFrame in validation context",
                    "error_type": "missing_data",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True
                }
            
            # Try direct calculation first
            response = intent_manager.generate_insight(prompt, df)
            
            # If direct calculation failed or returned fallback, try LLM
            if response.get("is_error") or (not response.get("is_direct_calculation") and not self.use_mock):
                logger.info("Direct calculation failed or returned fallback, trying LLM")
                llm_response = self._generate_llm_insight(prompt, validation_context)
                if not llm_response.get("is_error"):
                    response = llm_response
            
            # Add to history if requested
            if add_to_history:
                st.session_state.conversation_history.append({
                    "prompt": prompt,
                    "response": response,
                    "timestamp": response["timestamp"]
                })
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating insight: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "summary": "Failed to generate insight",
                "error": str(e),
                "error_type": "unknown",
                "timestamp": datetime.now().isoformat(),
                "is_error": True
            }

    def _generate_llm_insight(self, prompt: str, validation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a new insight from the LLM based on the user prompt and validation context."""
        try:
            # Input validation
            if not prompt or not isinstance(prompt, str):
                error_msg = "Invalid prompt: Prompt must be a non-empty string"
                logger.error(error_msg)
                return {
                    "summary": "Failed to generate insight", "error": error_msg,
                    "error_type": "input_validation", "timestamp": datetime.now().isoformat(),
                    "is_mock": self.use_mock, "is_error": True, "is_direct_calculation": False
                }

            logger.info(f"Generating insight for prompt: {prompt[:50]}...")
            
            # Get DataFrame from validation context
            df = validation_context.get('df') if validation_context else None
            
            if df is None:
                return {
                    "summary": "No data available for analysis",
                    "error": "Missing DataFrame in validation context",
                    "error_type": "missing_data",
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True
                }
            
            # Prepare context for LLM
            if validation_context is None:
                validation_context = {}

            # Add DataFrame info to validation context
            if df is not None and not df.empty:
                try:
                    validation_context['data_shape'] = df.shape
                    validation_context['columns'] = df.columns.tolist()
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        validation_context['numeric_columns'] = numeric_cols
                        validation_context['basic_stats'] = df[numeric_cols].describe().to_dict()
                    logger.info(f"Added data info to validation_context for LLM")
                except Exception as e:
                    logger.error(f"Failed to add data stats to validation_context: {e}")
            else:
                logger.warning("[WARN] No validated data for LLM context enrichment.")

            # Generate the LLM prompt
            try:
                system_prompt_text = _load_system_prompt()
                data_summary_context = ""
                if df is not None and not df.empty:
                    df_llm_ctx = df.copy() # Use a copy for context gen
                    # Clean monetary columns for context generation
                    monetary_cols_ctx = [col for col in df_llm_ctx.columns if any(term in col.lower() for term in ['price', 'gross', 'revenue', 'profit', 'cost'])]
                    for col in monetary_cols_ctx:
                        if df_llm_ctx[col].dtype == 'object':
                            df_llm_ctx[col] = df_llm_ctx[col].astype(str).str.replace('[$,]', '', regex=True)
                            df_llm_ctx[col] = pd.to_numeric(df_llm_ctx[col], errors='coerce').fillna(0.0)
                        else: df_llm_ctx[col] = pd.to_numeric(df_llm_ctx[col], errors='coerce').fillna(0.0)
                    if 'Total Gross' not in df_llm_ctx.columns and 'FrontGross' in df_llm_ctx.columns and 'BackGross' in df_llm_ctx.columns:
                        df_llm_ctx['Total Gross'] = df_llm_ctx['FrontGross'] + df_llm_ctx['BackGross']

                    data_summary_context += f"Dataset Info: {df_llm_ctx.shape[0]} rows, {df_llm_ctx.shape[1]} columns. Columns: {df_llm_ctx.columns.tolist()}. "
                    if 'LeadSource' in df_llm_ctx.columns:
                        data_summary_context += f"Lead Sources: {df_llm_ctx['LeadSource'].nunique()} unique values. "
                    if 'Total Gross' in df_llm_ctx.columns:
                        try:
                            avg_gross = df_llm_ctx['Total Gross'].mean()
                            data_summary_context += f"Avg Total Gross: ${avg_gross:.2f}. "
                        except Exception as context_err: logger.error(f"[WARN] Error generating gross context: {context_err}")
                    system_prompt_text += f"\n\nAVAILABLE DATA CONTEXT:\n{data_summary_context}"

                prompt_generator = PromptGenerator()
                full_prompt = prompt_generator.generate_prompt(
                    system_prompt=system_prompt_text,
                    user_query=prompt,
                    validation_context=validation_context # Pass original context here
                )
                logger.info(f"Generated LLM prompt with length: {len(full_prompt)}")
            except Exception as e:
                error_msg = f"Error generating LLM prompt: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return {
                    "summary": "Failed to generate insight", "error": error_msg,
                    "error_type": "prompt_generation", "timestamp": datetime.now().isoformat(),
                    "is_mock": self.use_mock, "is_error": True, "is_direct_calculation": False
                }

            # Generate response using LLM or Mock
            response_text = ""
            llm_error = None
            if self.use_mock:
                logger.info("[DEBUG] Using mock response")
                response_text = self._generate_mock_response(prompt, validation_context)
            else:
                logger.info(f"[DEBUG] Using real LLM with provider: {self.llm_provider}")
                try:
                    if self.llm_provider == "openai":
                        response_text = self._call_openai(full_prompt)
                    elif self.llm_provider == "anthropic":
                        response_text = self._call_anthropic(full_prompt)
                    else:
                        raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                except Exception as e:
                    llm_error = f"Error calling LLM API: {str(e)}"
                    logger.error(f"{llm_error}\n{traceback.format_exc()}")
                    # Return error immediately if API call fails
                    return {
                        "summary": "Failed to generate insight", "error": llm_error,
                        "error_type": "api_call", "timestamp": datetime.now().isoformat(),
                        "is_mock": self.use_mock, "is_error": True, "is_direct_calculation": False
                    }

            # Process the LLM response
            try:
                logger.info(f"[DEBUG] Raw LLM response: {response_text[:100]}..." if response_text else "[DEBUG] Empty LLM response")
                response = self.formatter.format_response(response_text)
                logger.info(f"[DEBUG] Formatted LLM response: {response}")

                # Add standard fields
                response["timestamp"] = datetime.now().isoformat()
                response["is_mock"] = self.use_mock
                response["is_direct_calculation"] = False # It came from LLM

                # Add to conversation history if requested
                if add_to_history:
                    st.session_state.conversation_history.append({
                        "prompt": prompt,
                        "response": response,
                        "timestamp": response["timestamp"]
                    })
                    logger.info(f"[DEBUG] Added LLM result to history. Total: {len(st.session_state.conversation_history)}")

                return response

            except Exception as e:
                error_msg = f"Error processing LLM response: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return {
                    "summary": "Failed to generate insight", "error": str(e),
                    "error_type": "response_processing", "timestamp": datetime.now().isoformat(),
                    "is_mock": self.use_mock, "is_error": True, "is_direct_calculation": False
                }

        except Exception as e:
            error_msg = f"Unexpected error in generate_insight: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "summary": "Failed to generate insight", "error": str(e),
                "error_type": "unknown", "timestamp": datetime.now().isoformat(),
                "is_mock": self.use_mock, "is_error": True, "is_direct_calculation": False
            }

    def _generate_mock_response(self, prompt: str, validation_context: Dict[str, Any] = None) -> str:
        """Generate a mock response for testing purposes."""
        # Extract columns and build a more specific mock response based on the prompt
        cols = validation_context.get('columns', []) if validation_context else []

        # Default mock response
        mock_response = {
            "summary": f"Analysis of your data based on: '{prompt}'",
            "value_insights": [
                "This is a mock response for demonstration purposes.",
                "Here's what I found in the data:"
            ],
            "actionable_flags": [
                "Consider exploring additional patterns in your data."
            ],
            "confidence": "medium",
            "is_mock": True
        }

        # Add context-aware insights based on the prompt and columns
        prompt_lower = prompt.lower()

        # Look for references to lead sources
        if 'lead source' in prompt_lower or 'lead_source' in prompt_lower:
            mock_response["summary"] = "Analysis of Lead Sources in Your Data"

            # Check if a specific lead source is mentioned
            lead_source_mentions = [
                "cargurus", "autotrader", "truecar", "cars.com", "dealer website",
                "facebook", "instagram", "walk-in", "referral", "google"
            ]

            mentioned_source = next((source for source in lead_source_mentions if source in prompt_lower), None)

            if mentioned_source:
                # Case for specific lead source question
                source_name = mentioned_source.title()
                mock_response["summary"] = f"Analysis of {source_name} Lead Source"
                # Make insights dictionaries
                mock_response["value_insights"] = [
                    {"Metric": f"Sales from {source_name}", "Value": f"{7 if mentioned_source == 'cargurus' else 12}"},
                    {"Metric": f"Percentage of Total Sales", "Value": f"{12 if mentioned_source == 'cargurus' else 20}%"},
                    {"Metric": f"Average Profit Margin ({source_name})", "Value": f"${1250 if mentioned_source == 'cargurus' else 980:,.2f}"}
                ]
                mock_response["actionable_flags"] = [
                    f"Consider increasing marketing budget for {source_name} due to strong performance.",
                    f"Set up A/B testing for {source_name} leads to optimize conversion rates."
                ]
                mock_response["confidence"] = "high"
            else:
                # General lead source analysis
                mock_response["value_insights"] = [
                    {"Metric": "Website Lead Contribution", "Value": "35% of total sales"},
                    {"Metric": "Social Media Growth (MoM)", "Value": "23%"},
                    {"Metric": "Highest Avg Profit Margin", "Value": "Walk-in ($1,450)"}
                ]
                mock_response["actionable_flags"] = [
                    "Increase focus on website optimization for lead generation.",
                    "Review underperforming lead sources like print advertising (only 3% of sales)."
                ]

        # Look for references to sales performance
        elif any(term in prompt_lower for term in ['sales', 'sold', 'purchased', 'bought']):
            mock_response["summary"] = "Sales Performance Analysis"

            if 'model' in prompt_lower or 'make' in prompt_lower:
                # Questions about makes/models
                mock_response["value_insights"] = [
                    {"Metric": "Top Make by Sales", "Value": "Toyota (28%)"},
                    {"Metric": "Top Selling Model", "Value": "Honda Accord (15 units)"},
                    {"Metric": "Luxury Brand Profit Share", "Value": "34% (from 22% sales)"}
                ]
                mock_response["actionable_flags"] = [
                    "Consider increasing inventory of top-selling Toyota and Honda models.",
                    "Review pricing strategy for luxury vehicles to optimize margins."
                ]
            else:
                # General sales questions
                mock_response["value_insights"] = [
                    {"Metric": "Total Vehicles Sold", "Value": "120"},
                    {"Metric": "Sales Volume Growth", "Value": "8% (vs previous period)"},
                    {"Metric": "Average Sales Price", "Value": "$32,450 (up 3%)"}
                ]
                mock_response["actionable_flags"] = [
                    "Monitor sales velocity of recently added inventory.",
                    "Review sales team performance metrics for optimization opportunities."
                ]

        # Look for profit/revenue/financial questions
        elif any(term in prompt_lower for term in ['profit', 'revenue', 'margin', 'finance', 'money']):
            mock_response["summary"] = "Financial Performance Analysis"
            mock_response["value_insights"] = [
                {"Metric": "Total Gross Profit", "Value": "$420,500"},
                {"Metric": "Average Profit Margin", "Value": "12.3% (Target: 13%)"},
                {"Metric": "F&I Profit Contribution", "Value": "22%"}
            ]
            mock_response["actionable_flags"] = [
                "Identify vehicles with below-target margins for pricing adjustment.",
                "Provide additional F&I training to boost attachment rates."
            ]

        # Add data context if available
        if cols:
            sample_cols = cols[:5] if len(cols) > 5 else cols
            col_list = ", ".join(sample_cols)
            mock_response["value_insights"].append({"Available Data Columns": f"{col_list}"})

        return json.dumps(mock_response)

    def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API with the given prompt.
        
        Args:
            prompt (str): The prompt to send to OpenAI
            
        Returns:
            str: The response from OpenAI
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ]
            
            completion = generate_completion(messages=messages)
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    def _call_anthropic(self, prompt: str) -> str:
        """
        Call the Anthropic API with the given prompt.

        Args:
            prompt: The complete prompt to send to the API

        Returns:
            The response text from the API
        """
        try:
            print(f"[DEBUG] Calling Anthropic API with prompt length: {len(prompt)}")

            # Construct the API request
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            data = {
                "model": "claude-2",
                "max_tokens_to_sample": 1000,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "temperature": 0.3
            }

            # Make the API request
            response = requests.post(
                "https://api.anthropic.com/v1/complete",
                headers=headers,
                json=data
            )

            # Parse and return the response
            if response.status_code == 200:
                response_data = response.json()
                response_text = response_data.get("completion", "")
                print(f"[DEBUG] Received Anthropic response with length: {len(response_text)}" if response_text else "[DEBUG] Received empty Anthropic response")
                return response_text
            else:
                error_msg = f"Anthropic API error: {response.status_code}, {response.text}"
                logger.error(f"{error_msg}")
                raise ValueError(error_msg)

        except Exception as e:
            print(f"[ERROR] Anthropic API call failed: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Re-raise to be caught by the main generate_insight error handler
            raise ValueError(f"Error calling Anthropic API: {str(e)}")