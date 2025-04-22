"""
Test fixtures for end-to-end tests.
"""

import pandas as pd

def get_mock_sales_report_data():
    """Get mock sales report data."""
    return pd.DataFrame({
        "Lead Source": ["NeoIdentity", "AutoTrader", "CarGurus"],
        "Sales Count": [4, 2, 1],
        "Total Gross": [15000, 8000, 3000]
    })