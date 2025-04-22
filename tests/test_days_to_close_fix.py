"""
Manual test script for days_to_close fix.
"""

import pandas as pd
import re
from datetime import datetime

# Setup test data with a known Karen Davis record
test_data = pd.DataFrame({
    'sales_rep': ['Karen Davis', 'John Smith', 'Michael Johnson', 'Sarah Williams'],
    'days_to_close': [9, 15, 7, 12],
    'gross_profit': [1200, 950, 1500, 1100],
    'sale_count': [3, 2, 4, 2],
    'revenue': [35000, 28000, 42000, 31000],
    'lead_source': ['Website', 'CarGurus', 'Referral', 'Website'],
    'date': [datetime.now() for _ in range(4)]  # Just use current date for all
})

print("Test data created with Karen Davis record:")
print(test_data[test_data['sales_rep'] == 'Karen Davis'])
print("\n" + "="*50 + "\n")

# Define extraction patterns
sales_rep_pattern = re.compile(r"(?:(?:for|by|about|from|is)\s+['\"](.*?)['\"]\??)|(?:['\"](.*?)['\"]'?s)", re.IGNORECASE)
metric_pattern = re.compile(r'(?:what\s+(?:is|are|was|were)\s+the\s+)?(days?[\s_-]to[\s_-]close|closing\s+time|time\s+to\s+close|closing\s+days?|sale\s+duration|how\s+(?:long|many)\s+days?(?:\s+(?:does|did)\s+it\s+take)?)', re.IGNORECASE)

# Test extraction from different query formats
test_queries = [
    "What are the days_to_close for the sale made by 'Karen Davis'?",
    "What is 'Karen Davis's days to close?",
    "Tell me about 'Karen Davis' days to close",
    "How many days does it take for 'Karen Davis' to close a deal?",
    "What is the closing time for 'Karen Davis'?"
]

print("Testing query extraction patterns:")
for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Extract sales rep name
    rep_match = sales_rep_pattern.search(query)
    if rep_match:
        rep_name = next((group for group in rep_match.groups() if group is not None), None)
        print(f"  Sales Rep: {rep_name}")
    else:
        print("  No sales rep found")
    
    # Extract metric type
    metric_match = metric_pattern.search(query)
    if metric_match:
        metric_type = metric_match.group(1)
        print(f"  Metric: {metric_type}")
    else:
        print("  No metric found")
    
    # Filter data and extract value
    if rep_match:
        rep_name = next((group for group in rep_match.groups() if group is not None), None)
        if rep_name:
            # Case-insensitive filter
            filtered_data = test_data[test_data['sales_rep'].str.lower() == rep_name.lower()]
            if not filtered_data.empty:
                days_value = filtered_data['days_to_close'].mean()
                print(f"  Days to Close Value: {days_value}")
            else:
                print("  No matching data found")

print("\n" + "="*50 + "\n")
print("Test complete - verify that all queries correctly extract 'Karen Davis' and 'days_to_close' with value 9")