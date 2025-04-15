"""
Module for managing LLM interactions with structured output schemas.
Ensures responses follow a specific format for UI integration.
"""

import streamlit as st
import json
from typing import Dict, Any, Optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import time
import pandas as pd
import re

class LLMEngine:
    """Manages interactions with LLM, enforcing structured output schemas."""
    
    def __init__(self, use_mock: bool = True, api_key: Optional[str] = None):
        """
        Initialize the LLM engine.
        
        Args:
            use_mock: If True, use mock responses instead of real LLM calls
            api_key: OpenAI API key for real LLM interactions
        """
        self.use_mock = use_mock
        if not use_mock and OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
        
        # Define the expected response schema
        self.response_schema = {
            "summary": "A concise markdown-formatted insight (string)",
            "chart_data": "Optional chart configuration (dict with 'type' and 'data' keys, or empty dict)",
            "recommendation": "Actionable advice based on the insight (string)",
            "risk_flag": "Boolean indicating if a risk or concern is detected (bool)"
        }
        
        # System prompt for role conditioning and schema enforcement
        self.system_prompt = f"""
You are a dealership insights analyst. Your role is to provide actionable insights based on data analysis.
Always respond in a structured JSON format that matches the following schema:
{json.dumps(self.response_schema, indent=2)}

- 'summary' should be a concise, markdown-formatted insight.
- 'chart_data' should be a dictionary with 'type' (e.g., 'bar', 'line') and 'data' (a simple table of values as a list of lists or dict), or an empty dict if not relevant.
- 'recommendation' should be a clear, actionable piece of advice.
- 'risk_flag' should be true only if the insight indicates a concern or risk.

Ensure your response is always valid JSON and matches this schema exactly. Do not include additional keys or deviate from the format.
"""
    
    def parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the LLM response content into a structured dictionary.
        Handles cases where the response might not be perfect JSON.
        
        Args:
            content: Raw response content from the LLM
            
        Returns:
            Parsed and validated response dictionary
        """
        # Initialize default response based on schema
        default_response = {
            "summary": "Error: Unable to parse insight.",
            "chart_data": {},
            "recommendation": "No recommendation available due to parsing error.",
            "risk_flag": False
        }
        
        try:
            # Attempt to extract JSON from the content (sometimes wrapped in markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1)
            
            parsed_response = json.loads(content)
            
            # Validate against schema (check for required keys and types)
            validated_response = default_response.copy()
            
            if isinstance(parsed_response.get("summary"), str):
                validated_response["summary"] = parsed_response["summary"]
            if isinstance(parsed_response.get("chart_data"), dict):
                validated_response["chart_data"] = parsed_response["chart_data"]
                # Convert data to DataFrame if possible
                if "data" in validated_response["chart_data"] and isinstance(validated_response["chart_data"]["data"], (list, dict)):
                    try:
                        if isinstance(validated_response["chart_data"]["data"], list):
                            validated_response["chart_data"]["data"] = pd.DataFrame(validated_response["chart_data"]["data"][1:], columns=validated_response["chart_data"]["data"][0])
                        else:
                            validated_response["chart_data"]["data"] = pd.DataFrame(validated_response["chart_data"]["data"])
                    except Exception:
                        validated_response["chart_data"] = {}
            if isinstance(parsed_response.get("recommendation"), str):
                validated_response["recommendation"] = parsed_response["recommendation"]
            if isinstance(parsed_response.get("risk_flag"), bool):
                validated_response["risk_flag"] = parsed_response["risk_flag"]
            
            # Check for hallucinated keys
            extra_keys = set(parsed_response.keys()) - set(self.response_schema.keys())
            if extra_keys:
                st.warning(f"LLM response included unexpected keys: {extra_keys}. These were ignored.")
            
            return validated_response
        except json.JSONDecodeError:
            st.error("LLM response was not valid JSON. Falling back to default response.")
            return default_response
        except Exception as e:
            st.error(f"Error parsing LLM response: {str(e)}")
            return default_response
    
    def get_mock_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a mock response for testing purposes.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Mock response dictionary adhering to schema
        """
        # Extract query from prompt if it's JSON
        try:
            prompt_data = json.loads(prompt)
            query = prompt_data.get('query', prompt)
        except json.JSONDecodeError:
            query = prompt
        
        # Generate a relevant mock response based on the query
        summary = f"This is a mock insight in response to: '{query}'. "
        recommendation = "Consider reviewing current strategies based on this insight."
        risk_flag = False
        chart_data = {}
        
        if 'compare' in query.lower() or 'comparison' in query.lower():
            summary += "Here's a comparison showing a 15% improvement over the requested timeframe."
            chart_data = {
                'type': 'bar',
                'data': pd.DataFrame({
                    'Period': ['Previous', 'Current'],
                    'Value': [100, 115]
                })
            }
            recommendation = "Leverage the positive trend by increasing marketing efforts."
        elif 'factor' in query.lower() or 'driver' in query.lower():
            summary += "Key factors include improved marketing campaigns and better inventory management."
            recommendation = "Focus on sustaining these key drivers for continued success."
        elif 'recommendation' in query.lower() or 'improve' in query.lower():
            summary += "We recommend focusing on customer retention strategies and optimizing pricing."
            recommendation = "Implement a loyalty program to boost retention by 10%."
            risk_flag = True
        else:
            summary += "This insight provides detailed analysis of the requested metrics."
            chart_data = {
                'type': 'line',
                'data': pd.DataFrame({
                    'Month': ['Jan', 'Feb', 'Mar'],
                    'Value': [100, 120, 140]
                })
            }
        
        return {
            'summary': summary,
            'chart_data': chart_data,
            'recommendation': recommendation,
            'risk_flag': risk_flag,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
    
    def get_llm_response(self, prompt: str) -> Dict[str, Any]:
        """
        Get response from OpenAI LLM with schema enforcement.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Response dictionary from LLM, parsed to match schema
        """
        if not OPENAI_AVAILABLE or self.client is None:
            raise ValueError("OpenAI client not available. Ensure API key is provided and library is installed.")
        
        try:
            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5,  # Lower temperature for more structured output
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            parsed_response = self.parse_llm_response(content)
            parsed_response['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            return parsed_response
        except Exception as e:
            st.error(f"Error getting LLM response: {str(e)}")
            return {
                'summary': 'Error: Unable to generate insight.',
                'chart_data': {},
                'recommendation': 'Unable to provide recommendation due to error.',
                'risk_flag': False,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
    
    def generate_insight(self, prompt: str) -> Dict[str, Any]:
        """
        Generate an insight based on the given prompt.
        
        Args:
            prompt: The input prompt for generating insight
            
        Returns:
            Response dictionary with insight data, adhering to schema
        """
        if self.use_mock or not OPENAI_AVAILABLE or self.client is None:
            return self.get_mock_response(prompt)
        else:
            return self.get_llm_response(prompt) 