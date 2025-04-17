"""
Module for managing LLM interactions with structured output schemas.
Ensures responses follow a specific format for UI integration.
"""

import streamlit as st
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import re
from datetime import datetime, timedelta

from .direct_processors import (
    process_generic_lead_source_query,
    extract_lead_source_from_prompt,
    clean_lead_source_name
)

class LLMEngine:
    """Manages interactions with LLM, enforcing structured output schemas."""
    
    def __init__(self, use_mock: bool = True, api_key: Optional[str] = None):
        """Initialize the LLM engine."""
        self.use_mock = use_mock
        self.api_key = api_key
        self.client = None
        
        if not use_mock and api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                st.warning("OpenAI package not installed. Using mock responses.")
                self.use_mock = True
    
    def generate_insight(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an insight based on the prompt and context."""
        # Check for direct processing opportunities
        if context and 'validated_data' in context:
            df = context['validated_data']
            
            # Check for lead source queries
            lead_source = extract_lead_source_from_prompt(prompt)
            if lead_source:
                return process_generic_lead_source_query(df, lead_source)
        
        # Fall back to standard LLM processing
        if self.use_mock or not self.client:
            return self.get_mock_response(prompt, context)
        else:
            return self.get_llm_response(prompt, context)

    def _generate_system_prompt(self) -> str:
        """Generate a comprehensive system prompt for the LLM."""
        return f"""
You are an expert automotive dealership analyst with deep understanding of business metrics, trends, and industry patterns.

Your role is to:
1. Analyze data thoroughly and identify meaningful patterns
2. Provide specific, quantified insights
3. Highlight actionable opportunities and risks
4. Support findings with statistical evidence
5. Maintain consistent output structure

RESPONSE REQUIREMENTS:
1. Always return valid JSON matching this schema:
{json.dumps(self.response_schema, indent=2)}

2. Insight Quality Guidelines:
- Lead with the most impactful finding
- Include specific numbers and percentages
- Compare metrics to relevant benchmarks
- Identify patterns and their significance
- Quantify business impact where possible

3. Statistical Rigor:
- Use appropriate statistical tests
- Report confidence intervals
- Note sample sizes and limitations
- Indicate statistical significance

4. Pattern Recognition:
- Look for trends over time
- Identify seasonal patterns
- Find correlations between metrics
- Flag anomalies and outliers

5. Action Orientation:
- Provide specific, measurable recommendations
- Prioritize suggestions by impact
- Include expected outcomes
- Note implementation considerations

CRITICAL: Return ONLY valid JSON matching the schema. No additional text or explanation.
"""
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data for significant patterns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = []
        
        try:
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Find date columns
            date_cols = [col for col in df.columns 
                        if any(term in col.lower() 
                              for term in ['date', 'time', 'month', 'year'])]
            
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])
                
                # Analyze each numeric column
                for col in numeric_cols:
                    if col == date_col:
                        continue
                    
                    # Sort by date
                    series = df.sort_values(date_col)[col].dropna()
                    
                    if len(series) < 2:
                        continue
                    
                    # Trend analysis
                    x = np.arange(len(series))
                    y = series.values
                    slope, _, r_value, p_value, _ = stats.linregress(x, y)
                    
                    if p_value < 0.05:
                        strength = abs(r_value)
                        if strength > 0.7:
                            trend_strength = "strong"
                        elif strength > 0.4:
                            trend_strength = "moderate"
                        else:
                            trend_strength = "weak"
                        
                        patterns.append({
                            "type": "trend",
                            "metric": col,
                            "direction": "increasing" if slope > 0 else "decreasing",
                            "strength": trend_strength,
                            "p_value": float(p_value),
                            "r_squared": float(r_value ** 2)
                        })
                    
                    # Seasonality analysis (if enough data)
                    if len(series) >= 12:
                        autocorr = pd.Series(y).autocorr(lag=12)
                        if abs(autocorr) > 0.6:
                            patterns.append({
                                "type": "seasonality",
                                "metric": col,
                                "strength": "strong" if abs(autocorr) > 0.8 else "moderate",
                                "correlation": float(autocorr)
                            })
                    
                    # Outlier analysis
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = series[(series < (Q1 - 1.5 * IQR)) | 
                                    (series > (Q3 + 1.5 * IQR))]
                    
                    if len(outliers) > 0:
                        patterns.append({
                            "type": "anomaly",
                            "metric": col,
                            "count": len(outliers),
                            "percentage": float(len(outliers) / len(series) * 100),
                            "values": outliers.tolist()[:5]  # First 5 outliers
                        })
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.6:
                            patterns.append({
                                "type": "correlation",
                                "metrics": [numeric_cols[i], numeric_cols[j]],
                                "correlation": float(corr),
                                "strength": "strong" if abs(corr) > 0.8 else "moderate"
                            })
        
        except Exception as e:
            st.warning(f"Error in pattern analysis: {str(e)}")
        
        return patterns
    
    def _calculate_metrics(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Calculate relevant metrics based on the query.
        
        Args:
            df: DataFrame to analyze
            query: User's query
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            # Find relevant columns based on query
            query_terms = set(query.lower().split())
            
            # Look for specific metrics in query
            metric_terms = {
                'sales': ['sales', 'revenue', 'volume'],
                'profit': ['profit', 'gross', 'margin'],
                'inventory': ['inventory', 'stock', 'vehicles'],
                'leads': ['leads', 'prospects', 'opportunities']
            }
            
            # Find matching columns
            for metric_type, terms in metric_terms.items():
                if any(term in query_terms for term in terms):
                    matching_cols = [col for col in df.columns 
                                   if any(term in col.lower() 
                                         for term in terms)]
                    
                    if matching_cols:
                        col = matching_cols[0]
                        if pd.api.types.is_numeric_dtype(df[col]):
                            stats = df[col].describe()
                            
                            metrics[f"{metric_type}_metrics"] = {
                                "current": float(stats['mean']),
                                "min": float(stats['min']),
                                "max": float(stats['max']),
                                "std_dev": float(stats['std']),
                                "sample_size": int(stats['count'])
                            }
                            
                            # Calculate confidence interval
                            ci = stats.mean() + stats.std() * stats.t.interval(0.95, len(df)-1)
                            metrics[f"{metric_type}_confidence_interval"] = {
                                "lower": float(ci[0]),
                                "upper": float(ci[1])
                            }
            
            # Time-based analysis if date column exists
            date_cols = [col for col in df.columns 
                        if any(term in col.lower() 
                              for term in ['date', 'time', 'month', 'year'])]
            
            if date_cols and matching_cols:
                date_col = date_cols[0]
                metric_col = matching_cols[0]
                
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])
                
                # Calculate period-over-period change
                df_sorted = df.sort_values(date_col)
                if len(df_sorted) >= 2:
                    first_value = df_sorted[metric_col].iloc[0]
                    last_value = df_sorted[metric_col].iloc[-1]
                    
                    if first_value != 0:
                        change_pct = ((last_value - first_value) / first_value) * 100
                        metrics["period_change"] = {
                            "absolute": float(last_value - first_value),
                            "percentage": float(change_pct),
                            "direction": "increasing" if change_pct > 0 else "decreasing"
                        }
        
        except Exception as e:
            st.warning(f"Error in metrics calculation: {str(e)}")
        
        return metrics
    
    def parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1)
            
            # Parse JSON
            response = json.loads(content)
            
            # Validate required fields
            required_fields = ["summary", "value_insights", "actionable_flags", "confidence"]
            for field in required_fields:
                if field not in response:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate and clean value insights
            if not isinstance(response["value_insights"], list):
                response["value_insights"] = [str(response["value_insights"])]
            response["value_insights"] = [str(insight) for insight in response["value_insights"]]
            
            # Validate and clean actionable flags
            if not isinstance(response["actionable_flags"], list):
                response["actionable_flags"] = [str(response["actionable_flags"])]
            response["actionable_flags"] = [str(flag) for flag in response["actionable_flags"]]
            
            # Validate confidence
            valid_confidence = ["high", "medium", "low"]
            response["confidence"] = response["confidence"].lower()
            if response["confidence"] not in valid_confidence:
                response["confidence"] = "medium"
            
            # Add timestamp
            response["timestamp"] = datetime.now().isoformat()
            
            return response
            
        except Exception as e:
            st.error(f"Error parsing LLM response: {str(e)}")
            return {
                "summary": "Error parsing insight response",
                "value_insights": [
                    "The system received a response that could not be properly parsed.",
                    f"Error: {str(e)}"
                ],
                "actionable_flags": [
                    "Please try regenerating the insight."
                ],
                "confidence": "low",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_llm_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get response from OpenAI with enhanced context."""
        if not self.client:
            return self.get_mock_response(prompt, context)
        
        try:
            # Analyze data if provided
            patterns = []
            metrics = {}
            if context and 'validated_data' in context:
                df = context['validated_data']
                patterns = self._analyze_patterns(df)
                metrics = self._calculate_metrics(df, prompt)
            
            # Enhance prompt with patterns and metrics
            enhanced_prompt = {
                "query": prompt,
                "patterns_detected": patterns,
                "calculated_metrics": metrics
            }
            
            # Call OpenAI
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(enhanced_prompt)}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return self.parse_llm_response(content)
            
        except Exception as e:
            st.error(f"Error getting LLM response: {str(e)}")
            return {
                "summary": "Error generating insight",
                "value_insights": [
                    "The system encountered an error while generating insights.",
                    f"Error: {str(e)}"
                ],
                "actionable_flags": [
                    "Please try your request again."
                ],
                "confidence": "low",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_mock_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an intelligent mock response."""
        try:
            # Analyze data if provided
            patterns = []
            metrics = {}
            if context and 'validated_data' in context:
                df = context['validated_data']
                patterns = self._analyze_patterns(df)
                metrics = self._calculate_metrics(df, prompt)
            
            # Generate response based on patterns and metrics
            response = {
                "summary": f"Analysis based on {len(patterns)} detected patterns and {len(metrics)} key metrics.",
                "value_insights": [],
                "actionable_flags": [],
                "confidence": "medium",
                "timestamp": datetime.now().isoformat()
            }
            
            # Add pattern-based insights
            for pattern in patterns[:2]:  # Top 2 patterns
                if pattern["type"] == "trend":
                    response["value_insights"].append(
                        f"Found {pattern['strength']} {pattern['direction']} trend in {pattern['metric']} "
                        f"(R² = {pattern['r_squared']:.2f})"
                    )
                elif pattern["type"] == "seasonality":
                    response["value_insights"].append(
                        f"Detected {pattern['strength']} seasonal pattern in {pattern['metric']} "
                        f"(correlation = {pattern['correlation']:.2f})"
                    )
                elif pattern["type"] == "anomaly":
                    response["value_insights"].append(
                        f"Identified {pattern['count']} anomalies in {pattern['metric']} "
                        f"({pattern['percentage']:.1f}% of data)"
                    )
            
            # Add metric-based insights
            for metric_type, values in metrics.items():
                if "metrics" in metric_type:
                    response["value_insights"].append(
                        f"{metric_type.split('_')[0].title()}: "
                        f"Average {values['current']:.2f} "
                        f"(±{values['std_dev']:.2f})"
                    )
            
            # Add recommendations
            if patterns or metrics:
                response["actionable_flags"].append(
                    "Monitor identified patterns for continued relevance"
                )
                if any(p["type"] == "anomaly" for p in patterns):
                    response["actionable_flags"].append(
                        "Investigate detected anomalies for potential issues or opportunities"
                    )
            
            return response
            
        except Exception as e:
            st.error(f"Error generating mock response: {str(e)}")
            return {
                "summary": "Error generating mock insight",
                "value_insights": [
                    "The system encountered an error while generating mock insights.",
                    f"Error: {str(e)}"
                ],
                "actionable_flags": [
                    "Please try your request again."
                ],
                "confidence": "low",
                "timestamp": datetime.now().isoformat()
            }