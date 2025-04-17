"""
Validate specific business queries against the enhanced prompt system.

This script tests the queries mentioned in the task to evaluate the formatting,
accuracy, and usefulness of the generated insights.
"""

import sys
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.insight_conversation_enhanced import ConversationManager

def create_sample_data():
    """Create realistic sample automotive sales data."""
    # Create a sample DataFrame with automotive sales data
    data = {
        'SaleDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'SalesRepName': ['Alice Johnson', 'Bob Smith', 'Charlie Davis', 'Diana Martinez'] * 25,
        'VehicleType': ['SUV', 'Sedan', 'Truck', 'Compact', 'SUV'] * 20,
        'LeadSource': ['Internet', 'Walk-in', 'Referral', 'Phone', 'Return Customer'] * 20,
        'FrontGross': [round(1000 + 2000 * pd.np.random.random(), 2) for _ in range(100)],
        'BackGross': [round(500 + 1500 * pd.np.random.random(), 2) for _ in range(100)],
        'TotalGross': [0] * 100,  # Will be calculated
        'VIN': [f'VIN{i:05d}' for i in range(1, 101)],
        'Make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan'] * 20,
        'Model': ['Camry', 'Civic', 'F-150', 'Silverado', 'Altima'] * 20,
        'Year': [2020, 2021, 2022, 2023] * 25,
        'ListPrice': [round(20000 + 30000 * pd.np.random.random(), 2) for _ in range(100)],
        'SalePrice': [round(18000 + 28000 * pd.np.random.random(), 2) for _ in range(100)],
        'Days_on_Lot': [int(pd.np.random.randint(1, 120)) for _ in range(100)],
        'FinanceProduct': ['Extended Warranty', 'GAP Insurance', 'Prepaid Maintenance', 'None', 'Multiple'] * 20,
        'FinanceIncome': [round(0 + 1200 * pd.np.random.random(), 2) for _ in range(100)]
    }
    
    # Calculate TotalGross
    for i in range(100):
        data['TotalGross'][i] = data['FrontGross'][i] + data['BackGross'][i]
    
    # Create DataFrame
    return pd.DataFrame(data)

def evaluate_response(query, response):
    """Evaluate the quality of the response to a business query."""
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    # Print summary
    print(f"Summary: {response['summary']}")
    print()
    
    # Print value insights
    print("Value Insights:")
    for idx, insight in enumerate(response['value_insights'], 1):
        print(f"  {idx}. {insight}")
    print()
    
    # Print actionable flags
    print("Actionable Flags:")
    for idx, flag in enumerate(response['actionable_flags'], 1):
        print(f"  {idx}. {flag}")
    print()
    
    # Print confidence level
    print(f"Confidence: {response['confidence']}")
    print("-" * 80)
    
    # Evaluate formatting (markdown presence)
    markdown_count = 0
    if "**" in response['summary']:
        markdown_count += 1
    
    for insight in response['value_insights']:
        if "**" in insight:
            markdown_count += 1
    
    print(f"Markdown Usage: {markdown_count} instances")
    
    # Evaluate numeric content
    has_numbers = False
    if any(c.isdigit() for c in response['summary']):
        has_numbers = True
    
    for insight in response['value_insights']:
        if any(c.isdigit() for c in insight):
            has_numbers = True
            break
    
    print(f"Contains Numeric Data: {'Yes' if has_numbers else 'No'}")
    
    # Evaluate actionability
    action_verbs = ["consider", "review", "analyze", "implement", "focus", "increase", 
                    "decrease", "improve", "examine", "prioritize", "allocate", "investigate"]
    
    has_action_verbs = False
    for flag in response['actionable_flags']:
        if any(verb in flag.lower() for verb in action_verbs):
            has_action_verbs = True
            break
    
    print(f"Contains Action Verbs: {'Yes' if has_action_verbs else 'No'}")
    print("\n")
    
    return {
        "markdown_usage": markdown_count,
        "has_numbers": has_numbers,
        "has_action_verbs": has_action_verbs
    }

def main():
    """Run the validation tests on business queries."""
    print("\n=== Validating Business Queries with Enhanced Prompt ===\n")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.validated_data = create_sample_data()
    
    # Initialize conversation manager with mock mode
    # In a real environment, set use_mock=False and ensure API keys are configured
    conv_manager = ConversationManager(use_mock=True)
    
    # Create validation context
    validation_context = {
        'data_shape': st.session_state.validated_data.shape,
        'columns': st.session_state.validated_data.columns.tolist(),
        'lead_source_breakdown': st.session_state.validated_data['LeadSource'].value_counts().to_dict()
    }
    
    # Test queries from the task
    test_queries = [
        "Which sales rep closed the most deals last month?",
        "What was the average front gross by lead source?",
        "Which day had the highest volume of sales?"
    ]
    
    results = []
    for query in test_queries:
        response = conv_manager.generate_insight(query, validation_context)
        eval_result = evaluate_response(query, response)
        results.append({
            "query": query,
            "response": response,
            "evaluation": eval_result
        })
    
    # Summarize overall results
    print("\n=== Summary of Query Validations ===\n")
    for idx, result in enumerate(results, 1):
        query = result["query"]
        eval_result = result["evaluation"]
        print(f"{idx}. Query: '{query}'")
        print(f"   Markdown Usage: {eval_result['markdown_usage']} instances")
        print(f"   Contains Numeric Data: {'Yes' if eval_result['has_numbers'] else 'No'}")
        print(f"   Contains Action Verbs: {'Yes' if eval_result['has_action_verbs'] else 'No'}")
        print()
    
    print("Validation complete.")

if __name__ == "__main__":
    main()
