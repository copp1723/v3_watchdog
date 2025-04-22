"""
Simple, direct LLM querying for data analysis without complex pipelines.
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
import time

from ..llm.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

class DirectQueryProcessor:
    """Process queries directly with the LLM without complex pipelines."""
    
    def __init__(self, use_mock: bool = False):
        """Initialize the direct query processor."""
        self.llm_engine = LLMEngine(use_mock=use_mock)
        self.history = []
        self.query_count = 0
        self.error_count = 0
        
    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a query directly using the LLM with minimal preprocessing.
        
        Args:
            query: User's natural language question
            df: DataFrame containing the data to analyze
            
        Returns:
            Dict with the analysis results
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Basic validation - just make sure we have data
            if df is None or df.empty:
                return {
                    "summary": "No data available for analysis",
                    "recommendations": ["Please upload a dataset first"],
                    "error_type": "NO_DATA"
                }
            
            # Prepare minimal context about the data
            data_context = {
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "sample": df.head(3).to_dict(orient="records"),
                "query": query
            }
            
            # Get column summaries for key metrics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                data_context["metrics"] = {
                    col: {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0
                    } for col in numeric_cols[:5]  # Limit to first 5 numeric columns
                }
                
            # Direct LLM call - no complex pipeline
            prompt = f"""
            Analyze the following data and answer the user's question: "{query}"
            
            Data summary:
            - Dataset has {len(df)} rows and {len(df.columns)} columns
            - Columns: {', '.join(df.columns.tolist())}
            
            Please provide your analysis in JSON format with these keys:
            - summary: A concise answer to the user's question
            - metrics: Key metrics supporting your answer
            - breakdown: Detailed breakdown if applicable
            - recommendations: Any recommendations based on the data
            - confidence: Your confidence in the answer (high, medium, low)
            
            Return ONLY valid JSON.
            """
            
            # Call LLM
            response = self.llm_engine.client.chat.completions.create(
                model=self.llm_engine.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Extract the JSON response
            try:
                # Find JSON in response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                json_str = result_text[json_start:json_end] if json_start >= 0 else result_text
                
                # Parse the JSON
                result = json.loads(json_str)
                
                # Ensure minimum response structure
                required_keys = ["summary", "metrics", "recommendations", "confidence"]
                for key in required_keys:
                    if key not in result:
                        result[key] = "" if key == "summary" else [] if key in ["metrics", "recommendations"] else "medium"
                
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                result = {
                    "summary": result_text[:500],  # Truncate to first 500 chars
                    "metrics": {},
                    "breakdown": [],
                    "recommendations": [],
                    "confidence": "medium"
                }
            
            # Add execution metadata
            end_time = time.time()
            result["_execution_metadata"] = {
                "execution_time_ms": (end_time - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
                "query": query
            }
            
            # Update history
            self.history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            self.history.append({
                "role": "assistant",
                "content": json.dumps(result),
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            # Simple error handling
            logger.error(f"Error processing query: {str(e)}")
            self.error_count += 1
            
            end_time = time.time()
            error_result = {
                "summary": f"Sorry, I couldn't process your question: {str(e)}",
                "metrics": {},
                "breakdown": [],
                "recommendations": ["Try asking in a different way"],
                "confidence": "low",
                "error_type": "PROCESSING_ERROR"
            }
            
            # Update history
            self.history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            self.history.append({
                "role": "assistant",
                "content": json.dumps(error_result),
                "timestamp": datetime.now().isoformat()
            })
            
            return error_result
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        return {"history": [], "ui_state": {"chat_bar_text": ""}}
        
    def get_metrics(self):
        """Get usage metrics."""
        return {
            "query_count": self.query_count,
            "error_count": self.error_count,
            "success_rate": (self.query_count - self.error_count) / max(1, self.query_count)
        } 