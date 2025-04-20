"""
Example usage of the Agent Sandbox for safe code execution.

This example demonstrates how to use the Agent Sandbox to execute
LLM-generated code safely with schema validation and error handling.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.agent_sandbox import (
    code_execution_in_sandbox,
    SandboxConfig,
    SandboxExecutionError,
    SchemaValidationError
)

# Mock LLM functions (in a real application, these would call an actual LLM API)
def mock_code_generation(prompt: str) -> str:
    """
    Mock function to generate code from a prompt.
    
    Args:
        prompt: User query or instruction
        
    Returns:
        Generated Python code as a string
    """
    if "sales by source" in prompt.lower():
        return """
# Analyze sales by lead source
sales_by_source = df.groupby('lead_source')['sales'].sum()
avg_by_source = df.groupby('lead_source')['sales'].mean()

# Calculate percentage of total
total_sales = df['sales'].sum()
pct_by_source = (sales_by_source / total_sales * 100).round(1)

# Create result
result = {
    "answer": f"Web leads generated ${sales_by_source['Web']:,.0f} in sales ({pct_by_source['Web']}% of total), "
              f"while Referrals generated ${sales_by_source['Referral']:,.0f} ({pct_by_source['Referral']}%).",
    "data": {
        "sales_by_source": sales_by_source.to_dict(),
        "percentage": pct_by_source.to_dict()
    },
    "chart_type": "pie",
    "confidence": 0.95
}
"""
    elif "profit margin" in prompt.lower():
        return """
# Calculate profit margins
df['margin'] = (df['gross'] / df['sales'] * 100).round(1)
avg_margin = df['margin'].mean()
margins_by_source = df.groupby('lead_source')['margin'].mean().round(1)

# Find best and worst sources
best_source = margins_by_source.idxmax()
worst_source = margins_by_source.idxmin()

# Create result
result = {
    "answer": f"The average profit margin is {avg_margin:.1f}%. {best_source} leads have the highest margin at {margins_by_source[best_source]:.1f}%, while {worst_source} leads have the lowest at {margins_by_source[worst_source]:.1f}%.",
    "data": {
        "margins_by_source": margins_by_source.to_dict()
    },
    "chart_type": "bar",
    "confidence": 0.9
}
"""
    elif "intentional error" in prompt.lower():
        # Generate code with an error to demonstrate error handling
        return """
# This code has an intentional error (accessing non-existent column)
total_revenue = df['revenue'].sum()  # 'revenue' column doesn't exist

result = {
    "answer": f"Total revenue: ${total_revenue:,.2f}",
    "data": {"total_revenue": total_revenue},
    "chart_type": "none",
    "confidence": 0.8
}
"""
    else:
        # Default analysis
        return """
# Basic summary analysis
total_sales = df['sales'].sum()
avg_sales = df['sales'].mean()
total_gross = df['gross'].sum()
avg_gross = df['gross'].mean()
record_count = len(df)

# Create result dictionary
result = {
    "answer": f"Found {record_count} records with ${total_sales:,.0f} in total sales and ${total_gross:,.0f} in gross profit. The average sale was ${avg_sales:.2f} with ${avg_gross:.2f} gross profit.",
    "data": {
        "total_sales": float(total_sales),
        "avg_sales": float(avg_sales),
        "total_gross": float(total_gross),
        "avg_gross": float(avg_gross),
        "record_count": record_count
    },
    "chart_type": "table",
    "confidence": 0.95
}
"""

def process_user_query(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a user query by generating and executing code safely.
    
    Args:
        query: User query or instruction
        df: DataFrame to analyze
        
    Returns:
        Dictionary with the analysis results
    """
    print(f"Processing query: {query}")
    
    try:
        # Generate code based on the query
        print("Generating code...")
        code = mock_code_generation(query)
        
        # Define sandbox configuration
        config = SandboxConfig(
            memory_limit_mb=256,
            execution_timeout_seconds=5,
            enable_network=False
        )
        
        # Execute the code safely
        print("Executing code in sandbox...")
        result = code_execution_in_sandbox(
            code=code,
            df=df,
            llm_service_func=mock_code_generation,
            original_prompt=query,
            enable_retry=True,
            config=config
        )
        
        print("Execution successful!")
        return result
        
    except SandboxExecutionError as e:
        print(f"Execution error: {e}")
        return {
            "answer": f"Sorry, I encountered an error while analyzing your data: {str(e)}",
            "data": {},
            "chart_type": "none",
            "confidence": 0.0,
            "error": str(e)
        }
    except SchemaValidationError as e:
        print(f"Schema validation error: {e}")
        return {
            "answer": f"Sorry, the analysis result didn't match the expected format: {str(e)}",
            "data": {},
            "chart_type": "none",
            "confidence": 0.0,
            "error": str(e)
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "answer": f"Sorry, an unexpected error occurred: {str(e)}",
            "data": {},
            "chart_type": "none",
            "confidence": 0.0,
            "error": str(e)
        }

def main():
    """Run the example."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'sales': [100, 150, 200, 120, 80, 250, 300, 180, 220, 190],
        'gross': [25, 40, 50, 30, 15, 60, 75, 45, 55, 47],
        'lead_source': ['Web', 'Referral', 'Web', 'Walk-in', 'Web', 
                      'Referral', 'Web', 'Walk-in', 'Referral', 'Web']
    })
    
    # Sample queries to demonstrate different analyses
    queries = [
        "Give me a summary of the sales data",
        "Analyze sales by lead source",
        "What's the profit margin by lead source?",
        "This query will cause an intentional error",
    ]
    
    # Process each query
    for i, query in enumerate(queries):
        print("\n" + "="*80)
        print(f"QUERY {i+1}: {query}")
        print("="*80)
        
        result = process_user_query(query, df)
        
        print("\nRESULT:")
        print(f"Answer: {result['answer']}")
        print(f"Chart Type: {result['chart_type']}")
        print(f"Confidence: {result['confidence']}")
        print("Data:")
        print(json.dumps(result['data'], indent=2))
        
        if 'metadata' in result:
            print("Metadata:")
            print(json.dumps(result['metadata'], indent=2))
        
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print("\n" + "="*80)
    print("Example complete!")

if __name__ == "__main__":
    main()