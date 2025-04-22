"""
Mock Nova Act sales data for testing.
"""

import pandas as pd
from datetime import datetime, timedelta

def create_mock_sales_data():
    """Create mock sales data matching Nova Act format."""
    
    # Create base data
    data = {
        'LeadSource': [
            'NeoIdentity', 'NeoIdentity', 'NeoIdentity', 'NeoIdentity',  # 4 sales
            'CarGurus', 'CarGurus', 'CarGurus',  # 3 sales
            'Website', 'Website', 'Website', 'Website', 'Website',  # 5 sales
            'Walk-in', 'Walk-in', 'Walk-in',  # 3 sales
            'Phone', 'Phone', 'Phone'  # 3 sales
        ],
        'TotalGross': [
            2800, 2400, 2900, 2300,  # NeoIdentity avg: $2,600
            2000, 1800, 2200,  # CarGurus avg: $2,000
            1500, 1800, 1600, 1400, 1700,  # Website avg: $1,600
            1900, 2100, 1800,  # Walk-in avg: $1,933
            1600, 1400, 1800  # Phone avg: $1,600
        ],
        'SalePrice': [
            25000, 22000, 28000, 24000,
            21000, 20000, 23000,
            19000, 21000, 20000, 18000, 22000,
            21000, 23000, 20000,
            19000, 18000, 21000
        ]
    }
    
    # Add dates (last 30 days)
    today = datetime.now()
    data['SaleDate'] = [
        today - timedelta(days=x) for x in range(len(data['LeadSource']))
    ]
    
    # Add VINs
    data['VIN'] = [
        f'1HGCM82633A{str(i).zfill(6)}' for i in range(len(data['LeadSource']))
    ]
    
    return pd.DataFrame(data)

def get_expected_metrics():
    """Get expected metrics for validation."""
    return {
        'total_sales': 18,
        'total_gross': 33800,
        'avg_gross': 1877.78,
        'top_source': 'Website',
        'top_source_sales': 5,
        'top_source_percentage': 27.78,  # 5/18 * 100
        'highest_avg_source': 'NeoIdentity',
        'highest_avg_gross': 2600.00
    }