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
import requests
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
    
    def get_mock_response(self, prompt: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate a mock response for testing that's more informative and relevant.
        
        Args:
            prompt: The input prompt
            df: Optional DataFrame to base mock data on
            
        Returns:
            Mock response dictionary with more realistic content
        """
        # Extract potential keywords from prompt
        keywords = prompt.lower().split()
        
        # Default mock data
        mock_data = {
            "summary": f"Analysis of the data based on: '{prompt}'",
            "chart_data": {
                "type": "line",
                "data": {"x": [1, 2, 3, 4], "y": [10, 25, 15, 30]},
                "title": "Sample Trend"
            },
            "recommendation": "Review the data patterns to identify opportunities for improvement.",
            "risk_flag": False
        }
        
        # Try to make the mock response more relevant to the prompt
        if any(word in keywords for word in ["2024", "model", "vehicle", "vehicles", "sold"]):
            mock_data["summary"] = "Based on the data, there were 12 vehicles sold from the 2024 model year."
            mock_data["chart_data"] = {
                "type": "bar",
                "data": {
                    "x": ["2021", "2022", "2023", "2024"],
                    "y": [5, 8, 10, 12]
                },
                "title": "Vehicle Sales by Model Year"
            }
            mock_data["recommendation"] = "The 2024 model year vehicles are showing a positive sales trend. Consider increasing inventory for these models."
        
        elif any(word in keywords for word in ["vin", "vins", "missing"]):
            mock_data["summary"] = "Analysis of records with missing VIN information."
            mock_data["chart_data"] = {
                "type": "pie",
                "data": {
                    "labels": ["Missing VINs", "Valid VINs"],
                    "values": [29, 0]
                },
                "title": "VIN Data Completeness"
            }
            mock_data["recommendation"] = "Implement a VIN validation process at data entry to improve data quality."
            mock_data["risk_flag"] = True
        
        # Add a flag to indicate this is a mock response
        mock_data["is_mock"] = True
        
        return mock_data
    
    def generate_insight(self, prompt: str, validation_context: Optional[Dict[str, Any]] = None, add_to_history: bool = True) -> Dict[str, Any]:
        """
        Generate an insight based on the prompt using LLM or mock responses.
        """
        # DEBUG: Print LLM mode on every insight generation
        print(f"[DEBUG][generate_insight] Starting insight generation...")
        print(f"[DEBUG][generate_insight] LLM Provider: {self.llm_provider}")
        print(f"[DEBUG][generate_insight] use_mock: {self.use_mock}")
        print(f"[DEBUG][generate_insight] client exists: {self.client is not None}")
        print(f"[DEBUG][generate_insight] api_key exists: {bool(self.api_key)}")
        
        # Extract DataFrame if provided in context
        df = None
        if validation_context and 'validation_report' in validation_context:
            df = validation_context.get('validation_report')
            print(f"[DEBUG][generate_insight] DataFrame provided in context: {df is not None}")
            
        # PRE-COMPUTE: Handle specific selling price queries directly
        if df is not None and "SellingPrice" in df.columns and "average selling price" in prompt.lower():
            print(f"[DEBUG][generate_insight] Pre-computing selling price metrics")
            # Convert to numeric and handle non-numeric values
            numeric_series = pd.to_numeric(df["SellingPrice"], errors="coerce").dropna()
            if not numeric_series.empty:
                avg_price = numeric_series.mean()
                # Create a pre-computed prompt for average selling price
                if "completed deals" in prompt.lower():
                    enriched_prompt = f"""The average selling price across all deals is ${avg_price:,.2f}. 
                    Present this insight with a title, a brief explanation, and a recommendation for sales leadership.
                    Focus on completed deals and provide context about what this average price indicates about the business."""
                    print(f"[DEBUG][generate_insight] Using pre-computed selling price prompt")
                else:
                    # Build standard context-aware prompt
                    enriched_prompt = self._build_enriched_prompt(prompt, validation_context)
            else:
                # Build standard context-aware prompt
                enriched_prompt = self._build_enriched_prompt(prompt, validation_context)
        else:
            # Build standard context-aware prompt
            enriched_prompt = self._build_enriched_prompt(prompt, validation_context)
            
        print(f"[DEBUG][generate_insight] Enriched prompt created (length: {len(enriched_prompt)})")
        
        # Get raw response
        raw_response = None
        is_mock = False  # Default to non-mock
        
        if self.use_mock:
            print("[DEBUG][generate_insight] Using mock response (use_mock is True)")
            raw_response = self.get_mock_response(prompt, df)
            is_mock = True
        elif not self.client:
            print("[ERROR][generate_insight] No LLM client available")
            raise ValueError("No LLM client available")
        elif not self.api_key:
            print("[ERROR][generate_insight] No API key available")
            raise ValueError("No API key available")
        else:
            print("[DEBUG][generate_insight] Attempting real LLM call...")
            try:
                # Call the appropriate LLM API based on provider
                if self.llm_provider == "openai":
                    raw_response = self._call_openai_api(enriched_prompt)
                    is_mock = False
                elif self.llm_provider == "anthropic":
                    raw_response = self._call_anthropic_api(enriched_prompt)
                    is_mock = False
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            except Exception as e:
                print(f"[ERROR][generate_insight] LLM API call failed: {type(e).__name__}: {str(e)}")
                raise  # Re-raise the exception instead of falling back to mock
        
        if raw_response is None:
            raise ValueError("No response generated (neither mock nor real)")
            
        # Format and validate response
        print("[DEBUG][generate_insight] Formatting response...")
        formatted_response = self.formatter.format_output(raw_response)
        
        # Add metadata
        formatted_response['timestamp'] = datetime.now().isoformat()
        formatted_response['prompt'] = prompt
        formatted_response['is_mock'] = is_mock
        
        # Store in conversation history if requested
        if add_to_history:
            st.session_state['conversation_history'].append({
                'prompt': prompt,
                'response': formatted_response,
                'timestamp': formatted_response['timestamp']
            })
        
        print(f"[DEBUG][generate_insight] Completed. Response is_mock={is_mock}")
        return formatted_response
        
    def _build_enriched_prompt(self, prompt: str, validation_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build an enriched prompt with validation context and actual data summaries for better insights.
        
        Args:
            prompt: The original user prompt
            validation_context: Optional validation context with data summary and report
            
        Returns:
            Enriched prompt string with data context
        """
        if not validation_context:
            print("[DEBUG][_build_enriched_prompt] No validation context provided")
            return prompt

        # Get the DataFrame from context
        df = validation_context.get('validation_report')
        if df is None:
            print("[DEBUG][_build_enriched_prompt] No DataFrame found in context")
            return prompt

        # Added debug for validation_context keys and DataFrame columns
        print("[DEBUG][_build_enriched_prompt] validation_context keys:", validation_context.keys())
        print("[DEBUG] Columns available:", df.columns.tolist())
        
        # Build data summaries based on available columns
        context_parts = []
        
        # Add basic dataset info
        context_parts.append(f"\nDataset Overview:")
        context_parts.append(f"- Total Records: {len(df)}")
        
        # Sales Rep Summary if available
        if "SalesRepName" in df.columns:
            print("[DEBUG][_build_enriched_prompt] Generating sales rep summary")
            sales_summary = df.groupby("SalesRepName").size().sort_values(ascending=False)
            context_parts.append("\nSales by Representative:")
            for rep, count in sales_summary.items():
                context_parts.append(f"- {rep}: {count} deals")

        # Vehicle Summary if available
        if all(col in df.columns for col in ["VehicleYear", "VehicleMake", "VehicleModel"]):
            print("[DEBUG][_build_enriched_prompt] Generating vehicle summary")
            vehicle_summary = df.groupby("VehicleYear").size().sort_values(ascending=False)
            context_parts.append("\nSales by Model Year:")
            for year, count in vehicle_summary.items():
                context_parts.append(f"- {year}: {count} vehicles")

        # Gross Profit Summary if available
        if "Total Gross" in df.columns:
            print("[DEBUG][_build_enriched_prompt] Generating profit summary")
            total_gross = df["Total Gross"].sum()
            avg_gross = df["Total Gross"].mean()
            context_parts.append("\nProfit Summary:")
            context_parts.append(f"- Total Gross Profit: ${total_gross:,.2f}")
            context_parts.append(f"- Average Gross per Deal: ${avg_gross:,.2f}")

        # Lead Source Summary if available
        if "LeadSource" in df.columns:
            print("[DEBUG][_build_enriched_prompt] Generating lead source summary")
            lead_summary = df.groupby("LeadSource").size().sort_values(ascending=False)
            context_parts.append("\nSales by Lead Source:")
            for source, count in lead_summary.items():
                context_parts.append(f"- {source}: {count} deals")
                
        # Selling Price Summary if available - ADDED PER REQUIREMENT
        if "SellingPrice" in df.columns:
            print("[DEBUG][_build_enriched_prompt] Generating selling price summary")
            print("[DEBUG] SellingPrice dtype:", df["SellingPrice"].dtype)
            print("[DEBUG] SellingPrice sample:", df["SellingPrice"].head())
            
            # Convert to numeric and handle any non-numeric values
            numeric_series = pd.to_numeric(df["SellingPrice"], errors="coerce").dropna()
            if not numeric_series.empty:
                avg_price = numeric_series.mean()
                median_price = numeric_series.median()
                context_parts.append("\nSelling Price Summary:")
                context_parts.append(f"- Average Selling Price: ${avg_price:,.2f}")
                context_parts.append(f"- Median Selling Price: ${median_price:,.2f}")

        # Combine all parts
        data_context = "\n".join(context_parts)
        
        # Build the final prompt
        final_prompt = f"""Based on the following dealership data, please answer this question:

{prompt}

{data_context}

Important:
1. Base your answer ONLY on the data provided above
2. Be specific and include numbers from the data
3. If the data doesn't support answering the question, say so
"""

        print(f"[DEBUG][_build_enriched_prompt] Final prompt length: {len(final_prompt)}")
        return final_prompt
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the OpenAI API to get the response.
        """
        if not self.client or self.llm_provider != "openai":
            raise ValueError("OpenAI client not initialized or wrong provider.")
        
        # Define the messages for the chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing sales data. Provide concise insights and summaries."}, # Adjusted system prompt
            {"role": "user", "content": prompt}
        ]
        
        print(f"[DEBUG][_call_openai_api] Attempting OpenAI API call...")
        print(f"[DEBUG][_call_openai_api] Final Prompt Sent:\n{prompt}") # Added detailed logging
        
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo", # Using a standard model
                messages=messages,
                max_tokens=500, # Increased max tokens
                temperature=0.5, # Slightly lower temperature for more deterministic output
                response_format={"type": "json_object"} # Added JSON format request
            )
            
            # Extract the content from the response
            content = response.choices[0].message['content']
            print(f"[DEBUG][_call_openai_api] Response content (JSON string):\n{content}")
            
            # Attempt to parse the JSON content
            try:
                parsed_response = json.loads(content)
                print(f"[DEBUG][_call_openai_api] Successfully parsed JSON response.")
                # Validate basic structure (can be expanded)
                if not isinstance(parsed_response, dict) or "summary" not in parsed_response:
                    print("[WARNING][_call_openai_api] JSON response missing 'summary' key.")
                    # Fallback: treat the whole content as summary if parsing failed structurally
                    return {"summary": content.strip(), "error": True, "parsing_error": "Missing required keys"}
                return parsed_response
            except json.JSONDecodeError as json_err:
                print(f"[ERROR][_call_openai_api] Failed to parse JSON response: {json_err}")
                print(f"[DEBUG][_call_openai_api] Raw content causing error: {content}")
                # Fallback: return the raw content as summary with an error flag
                return {"summary": content.strip(), "error": True, "parsing_error": str(json_err)}
            
        except Exception as e:
            print(f"[ERROR][_call_openai_api] OpenAI API call failed: {e}")
            # Return a basic error structure
            return {"summary": f"Error communicating with OpenAI: {e}", "error": True}
    
    def _call_anthropic_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call Anthropic Claude API to generate insights.
        
        Args:
            prompt: The enriched prompt
            
        Returns:
            Structured insight dictionary
        """
        try:
            # Anthropic API endpoint
            url = "https://api.anthropic.com/v1/messages"
            
            # Construct the system prompt
            system_prompt = """
            You are an expert data analyst for a car dealership. Analyze the data and provide insights.
            Your responses should be in this JSON format:
            {
                "summary": "A clear, concise summary of the main insight",
                "chart_data": {
                    "type": "bar|line|pie",
                    "data": {"x": [labels], "y": [values]} or {"labels": [labels], "values": [values]},
                    "title": "Chart title",
                    "x_axis_label": "Label for X-axis (e.g., 'Model Year')",
                    "y_axis_label": "Label for Y-axis (e.g., 'Units Sold')"
                },
                "recommendation": "A specific, actionable recommendation based on the data",
                "risk_flag": true|false
            }
            """ 
            
            # API request headers and data
            headers = {
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-2.1",
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.2
            }
            
            # Make API call
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Parse the response
            try:
                content = response.json()
                answer = content['content'][0]['text']
                return json.loads(answer)
            except (KeyError, json.JSONDecodeError):
                raise ValueError("Invalid response format from Anthropic API")
                
        except Exception as e:
            st.error(f"Error calling Anthropic API: {str(e)}")
            raise
    
    def regenerate_insight(self, original_prompt_index: int, validation_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Regenerate an insight for a previous prompt index with enhanced context.
        Does NOT add the result to history automatically.

        Args:
            original_prompt_index: The index in the conversation history to regenerate
            validation_context: Optional validation context with data summary and report
            
        Returns:
            New formatted insight dictionary or None if index is invalid
        """
        if not 0 <= original_prompt_index < len(st.session_state['conversation_history']):
            st.error(f"Invalid index {original_prompt_index} for regeneration.")
            return None
            
        original_entry = st.session_state['conversation_history'][original_prompt_index]
        original_prompt = original_entry['prompt']
        
        # Get previous insights for context (up to the one being regenerated)
        previous_insights = [entry['response'] for entry in st.session_state['conversation_history'][:original_prompt_index+1]]
        
        # Generate new insight without adding to history
        new_insight = self.generate_insight(original_prompt, validation_context, add_to_history=False)
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