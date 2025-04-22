import pandas as pd
import numpy as np
import logging
import sys
import os
import re
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from watchdog_ai.insights.insight_conversation import ConversationManager
from watchdog_ai.insights.utils import validate_numeric_columns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create a test DataFrame with correct data types."""
    data = {
        'lead_source': ['Website', 'Referral', 'Walk-in', 'Website', 'Referral'],
        'profit': ['$1,000.00', '$2,500.00', '$3,000.00', '$1,500.00', '$2,000.00'],
        'sales_rep_name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'Alice Brown', 'Charlie Davis'],
        'sold_price': ['$25,000.00', '$35,000.00', '$40,000.00', '$30,000.00', '$32,000.00'],
        'vehicle_make': ['Toyota', 'Honda', 'Ford', 'Toyota', 'Honda'],
        'days_to_close': ['15', '20', '10', '25', '18']
    }
    return pd.DataFrame(data)

def test_numeric_conversion():
    """Test that numeric columns are properly converted."""
    df = create_test_data()
    df_converted = validate_numeric_columns(df)
    
    # Check that profit column is numeric and sums correctly
    assert pd.api.types.is_numeric_dtype(df_converted['profit']), "Profit column should be numeric"
    assert df_converted['profit'].sum() == 10000.0, f"Expected profit sum of 10000.0, got {df_converted['profit'].sum()}"
    
    # Check that sold_price column is numeric and sums correctly
    assert pd.api.types.is_numeric_dtype(df_converted['sold_price']), "Sold price column should be numeric"
    assert df_converted['sold_price'].sum() == 162000.0, f"Expected sold_price sum of 162000.0, got {df_converted['sold_price'].sum()}"
    
    # Check that days_to_close column is numeric and averages correctly
    assert pd.api.types.is_numeric_dtype(df_converted['days_to_close']), "Days to close column should be numeric"
    assert df_converted['days_to_close'].mean() == 17.6, f"Expected days_to_close mean of 17.6, got {df_converted['days_to_close'].mean()}"
    
    logger.info("All numeric conversion tests passed!")

def test_query_independence():
    """Test that different queries produce distinct and appropriate results."""
    df = create_test_data()
    conversation_manager = ConversationManager()
    
    # Test queries that should produce different results
    queries = [
        "What is the average days to close a sale?",
        "Which vehicle make has the highest average selling price?",
        "What was the total profit across all vehicle sales?"
    ]
    
    results = []
    for query in queries:
        result = conversation_manager.process_query(query, df)
        results.append(result)
        logger.info(f"Query: {query}")
        logger.info(f"Result: {result}\n")
    
    # Verify that results are distinct
    assert len(set(results)) == len(results), "All queries should produce distinct results"
    
    # Verify that results contain expected information
    assert any("17.6" in str(r) for r in results), "Should find average days to close"
    assert any("Ford" in str(r) for r in results), "Should identify Ford as highest average price"
    assert any("10000" in str(r) for r in results), "Should calculate total profit"
    
    logger.info("All query independence tests passed!")

def test_vehicle_make_pattern():
    """Test the pattern matching for vehicle make queries."""
    pattern = r"(?:which|what)\s+(?:vehicle\s+)?(?:make|brand|model)(?:\s+has|\s+is)?(?:\s+the)?(?:\s+highest|\s+lowest)?(?:\s+average)?(?:\s+selling\s+)?(?:price|value|sales)?"
    
    test_queries = [
        "Which vehicle make has the highest average selling price?",
        "What make has the lowest price?",
        "Which brand has the most sales?",
        "What vehicle model has the highest value?",
        "Which make has the best performance?"
    ]
    
    for query in test_queries:
        assert re.search(pattern, query.lower()), f"Pattern should match query: {query}"
    
    logger.info("All vehicle make pattern tests passed!")

if __name__ == "__main__":
    logger.info("Starting tests...")
    test_numeric_conversion()
    test_query_independence()
    test_vehicle_make_pattern()
    logger.info("All tests completed successfully!") 