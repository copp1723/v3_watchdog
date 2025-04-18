"""
Enhanced contextual intelligence module for guiding users through insights with smart follow-up prompts.
Added improved formatting, narrative structure, and clearer output parsing guidelines.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import re
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats

class EnhancedPromptGenerator:
    """Generates well-formatted, narrative-ready prompts with improved context structure."""
    
    def __init__(self, schema: Dict[str, str] = None):
        """Initialize the enhanced prompt generator with an optional schema."""
        self.schema = schema or {}
    
    def _format_column_stats(self, columns: List[str], data_types: Dict[str, str] = None, df: Optional[pd.DataFrame] = None) -> str:
        """Format column information with enhanced statistical context."""
        if not columns:
            return "No column information available."
            
        # Group columns by category
        grouped_columns = {}
        
        # Enhanced categories for automotive dealership data
        categories = {
            "financial": ["gross", "profit", "price", "cost", "commission", "payment", "down", "finance"],
            "vehicle": ["vin", "make", "model", "year", "stock", "trim", "color", "mileage", "condition"],
            "customer": ["customer", "buyer", "client", "lead", "source", "referral", "contact"],
            "sales": ["sales", "rep", "deal", "transaction", "close", "status"],
            "temporal": ["date", "time", "month", "day", "year", "quarter"],
            "location": ["location", "region", "territory", "market", "zone", "area"]
        }
        
        # Categorize columns with enhanced pattern matching
        for col in columns:
            col_lower = col.lower()
            assigned = False
            
            for category, keywords in categories.items():
                if any(keyword in col_lower for keyword in keywords):
                    if category not in grouped_columns:
                        grouped_columns[category] = []
                    grouped_columns[category].append(col)
                    assigned = True
                    break
            
            if not assigned:
                if "other" not in grouped_columns:
                    grouped_columns["other"] = []
                grouped_columns["other"].append(col)
        
        # Format output with enhanced statistics
        result = []
        result.append("## Data Structure Analysis")
        
        for category, cols in grouped_columns.items():
            result.append(f"\n### {category.title()} Metrics:")
            
            for col in cols:
                # Basic column info
                col_type = data_types.get(col, "unknown") if data_types else "unknown"
                result.append(f"- **{col}** ({col_type})")
                
                # Add statistical context if DataFrame is provided
                if df is not None and col in df.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # Calculate key statistics
                            stats = df[col].describe()
                            
                            # Add statistical insights
                            result.append(f"  - Range: {stats['min']:.2f} to {stats['max']:.2f}")
                            result.append(f"  - Average: {stats['mean']:.2f}")
                            
                            # Calculate additional insights
                            if len(df[col].dropna()) > 0:
                                # Detect outliers using IQR method
                                Q1 = stats['25%']
                                Q3 = stats['75%']
                                IQR = Q3 - Q1
                                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
                                if len(outliers) > 0:
                                    result.append(f"  - Found {len(outliers)} potential outliers")
                                
                                # Check for skewness
                                skewness = stats.skew()
                                if abs(skewness) > 1:
                                    direction = "right" if skewness > 0 else "left"
                                    result.append(f"  - Distribution is skewed {direction}")
                                
                        elif pd.api.types.is_string_dtype(df[col]):
                            # Analyze categorical data
                            value_counts = df[col].value_counts()
                            unique_count = len(value_counts)
                            result.append(f"  - {unique_count} unique values")
                            if unique_count < 10:  # Only show distribution for few categories
                                for val, count in value_counts.head(3).items():
                                    percentage = (count / len(df)) * 100
                                    result.append(f"  - {val}: {percentage:.1f}%")
                            
                    except Exception as e:
                        print(f"Error analyzing column {col}: {str(e)}")
        
        return "\n".join(result)
    
    def _analyze_trends(self, df: pd.DataFrame) -> str:
        """Analyze trends in time series data."""
        result = []
        result.append("\n## Trend Analysis")
        
        # Find date columns
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'month', 'year'])]
        if not date_cols:
            return ""
            
        date_col = date_cols[0]
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            # Find numeric columns for trend analysis
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            for col in numeric_cols:
                if col == date_col:
                    continue
                    
                # Sort by date and calculate period-over-period changes
                trend_df = df.sort_values(date_col)[[date_col, col]].dropna()
                if len(trend_df) < 2:
                    continue
                
                # Calculate overall trend
                x = np.arange(len(trend_df))
                y = trend_df[col].values
                slope, _, r_value, p_value, _ = stats.linregress(x, y)
                
                # Determine trend significance and direction
                if p_value < 0.05:  # Statistically significant trend
                    direction = "increasing" if slope > 0 else "decreasing"
                    strength = abs(r_value)
                    
                    if strength > 0.7:
                        trend_strength = "strong"
                    elif strength > 0.4:
                        trend_strength = "moderate"
                    else:
                        trend_strength = "weak"
                        
                    result.append(f"\n### {col} Trend:")
                    result.append(f"- Shows a {trend_strength} {direction} trend")
                    result.append(f"- Correlation strength: {strength:.2f}")
                    
                    # Calculate period-over-period change
                    total_change = ((y[-1] - y[0]) / y[0]) * 100
                    result.append(f"- Total change: {total_change:.1f}%")
                    
                    # Detect seasonality if enough data points
                    if len(trend_df) >= 12:
                        # Simple seasonality check using autocorrelation
                        autocorr = pd.Series(y).autocorr(lag=12)
                        if abs(autocorr) > 0.6:
                            result.append("- Shows seasonal patterns")
        
        except Exception as e:
            print(f"Error in trend analysis: {str(e)}")
            
        return "\n".join(result)
    
    def _analyze_relationships(self, df: pd.DataFrame) -> str:
        """Analyze relationships between variables."""
        result = []
        result.append("\n## Relationship Analysis")
        
        try:
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Calculate correlations
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.6:  # Strong correlation threshold
                            strong_correlations.append({
                                'var1': numeric_cols[i],
                                'var2': numeric_cols[j],
                                'correlation': corr
                            })
                
                if strong_correlations:
                    result.append("\n### Strong Relationships Found:")
                    for corr in strong_correlations:
                        direction = "positive" if corr['correlation'] > 0 else "negative"
                        strength = abs(corr['correlation'])
                        result.append(f"- {corr['var1']} and {corr['var2']}: {direction} correlation ({strength:.2f})")
                
        except Exception as e:
            print(f"Error in relationship analysis: {str(e)}")
            
        return "\n".join(result)
    
    def generate_prompt(self, system_prompt: str, user_query: str, validation_context: Dict[str, Any] = None) -> str:
        """Generate a well-formatted, narrative-ready prompt with enhanced context structure."""
        if validation_context is None:
            validation_context = {}
            
        # Build context section
        context_parts = []
        
        # Add header
        context_parts.append("# DATA CONTEXT AND ANALYSIS")
        context_parts.append("This section provides comprehensive analysis of the dealership dataset.")
        
        # Add data shape
        if 'data_shape' in validation_context:
            rows, cols = validation_context['data_shape']
            context_parts.append(f"\n## Dataset Overview:")
            context_parts.append(f"- Records: {rows:,} total transactions")
            context_parts.append(f"- Fields: {cols} metrics per record")
        
        # Add column information with enhanced statistics
        if 'columns' in validation_context:
            df = validation_context.get('validated_data')
            column_info = self._format_column_stats(
                validation_context['columns'],
                validation_context.get('data_types', {}),
                df
            )
            context_parts.append(column_info)
        
        # Add trend analysis if we have the data
        if 'validated_data' in validation_context and isinstance(validation_context['validated_data'], pd.DataFrame):
            df = validation_context['validated_data']
            
            # Add trend analysis
            trend_analysis = self._analyze_trends(df)
            if trend_analysis:
                context_parts.append(trend_analysis)
            
            # Add relationship analysis
            relationship_analysis = self._analyze_relationships(df)
            if relationship_analysis:
                context_parts.append(relationship_analysis)
        
        # Format the context section
        context_section = "\n\n".join(context_parts)
        
        # Add output format instructions
        format_instructions = """
# RESPONSE FORMAT REQUIREMENTS

Your response MUST be valid JSON following this schema:

{
  "summary": "Single sentence answering the question (15-25 words)",
  "value_insights": [
    "Key insight with supporting metric",
    "Additional insight showing trend or pattern",
    "Business impact or opportunity insight"
  ],
  "actionable_flags": [
    "Specific action recommendation",
    "Risk or opportunity that needs attention"
  ],
  "confidence": "high | medium | low",
  "metrics": {
    "key_metric": numeric_value,
    "trend": "increasing | decreasing | stable",
    "significance": "p_value if applicable"
  }
}

Guidelines:
- Summary: Direct answer to the question with key metric
- Value_insights: 2-4 data-backed findings
- Actionable_flags: 1-2 specific recommendations
- Confidence: Based on data quality and analysis
- Metrics: Include relevant numerical findings

Return ONLY valid JSON. No additional text or markdown.
"""
        
        # Combine all parts
        full_prompt = f"""
{system_prompt}

{format_instructions}

{context_section}

# USER QUERY
{user_query}
"""
        return full_prompt

def enhanced_generate_llm_prompt(selected_prompt: str, validation_context: Dict[str, Any] = None, previous_insights: List[Dict[str, Any]] = None) -> str:
    """
    Generate a structured, narrative-ready LLM prompt with enhanced context formatting.
    
    Args:
        selected_prompt: The selected follow-up prompt text
        validation_context: Dictionary with context information
        previous_insights: List of previous insights for context
        
    Returns:
        Structured, narrative-ready LLM prompt
    """
    # Load the enhanced system prompt
    try:
        system_prompt_path = "v3_watchdog/automotive_analyst_prompt_enhanced.md"
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except:
        # Fallback to original if enhanced not found
        system_prompt_path = "v3_watchdog/automotive_analyst_prompt.md"
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except:
            # Emergency fallback
            system_prompt = """
            # Automotive Data Analyst
            You analyze dealership data and provide concise, actionable insights.
            """
    
    # Initialize the enhanced prompt generator
    generator = EnhancedPromptGenerator()
    
    # Generate the prompt with enhanced formatting
    return generator.generate_prompt(
        system_prompt=system_prompt,
        user_query=selected_prompt,
        validation_context=validation_context
    )
