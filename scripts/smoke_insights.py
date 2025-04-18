#!/usr/bin/env python
"""
Smoke test script for Watchdog AI's core insights.

This script loads sample data and runs the key insight functions
to verify they are working correctly.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
import random

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smoke_insights")

# Import functions to test
try:
    from src.insights.core_insights import (
        get_sales_rep_performance,
        get_inventory_aging_alerts,
        compute_sales_performance,
        compute_inventory_anomalies
    )
    from src.utils.term_normalizer import TermNormalizer
except ImportError as e:
    logger.error(f"Failed to import insight functions: {e}")
    sys.exit(1)

def create_sample_sales_data(records=100):
    """
    Create sample sales data for testing.
    
    Args:
        records: Number of sample records to generate
        
    Returns:
        pandas DataFrame with synthetic sales data
    """
    # Generate dates over the last 180 days
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range(180)]
    
    # Create sales reps
    sales_reps = ["Alice Smith", "Bob Jones", "Charlie Brown", 
                  "David Miller", "Emily Wilson", "Frank Thomas"]
    
    # Create sample data
    data = {
        'SalesRepName': [random.choice(sales_reps) for _ in range(records)],
        'TotalGross': [random.randint(-1000, 5000) for _ in range(records)],
        'SaleDate': [random.choice(dates) for _ in range(records)],
        'StockNumber': [f'S{i:04d}' for i in range(1, records + 1)]
    }
    
    return pd.DataFrame(data)

def create_sample_inventory_data(records=50):
    """
    Create sample inventory data for testing.
    
    Args:
        records: Number of sample records to generate
        
    Returns:
        pandas DataFrame with synthetic inventory data
    """
    # Vehicle makes and models
    vehicle_models = {
        'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander'],
        'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot'],
        'Ford': ['F-150', 'Explorer', 'Escape', 'Edge'],
        'Chevrolet': ['Silverado', 'Malibu', 'Equinox', 'Tahoe']
    }
    
    makes = []
    models = []
    for _ in range(records):
        make = random.choice(list(vehicle_models.keys()))
        model = random.choice(vehicle_models[make])
        makes.append(make)
        models.append(model)
    
    # Create sample data
    data = {
        'VIN': [f'VIN{i:04d}' for i in range(1, records + 1)],
        'DaysInInventory': [random.randint(5, 150) for _ in range(records)],
        'Make': makes,
        'Model': models,
        'ListPrice': [random.randint(15000, 50000) for _ in range(records)]
    }
    
    return pd.DataFrame(data)

def run_smoke_test():
    """
    Run smoke tests for all insight functions.
    """
    logger.info("Starting smoke test for Watchdog AI insights...")
    
    # Create sample data
    logger.info("Generating sample sales data...")
    sales_df = create_sample_sales_data()
    
    logger.info("Generating sample inventory data...")
    inventory_df = create_sample_inventory_data()
    
    # Test 1: Sales Rep Performance
    logger.info("Testing get_sales_rep_performance...")
    try:
        sales_perf_df = get_sales_rep_performance(sales_df)
        logger.info(f"Sales performance success: Found {len(sales_perf_df)} sales reps")
        print("\nSales Rep Performance Summary:")
        print(sales_perf_df)
    except Exception as e:
        logger.error(f"Sales performance test failed: {e}")
    
    # Test 2: Inventory Aging Alerts
    logger.info("Testing get_inventory_aging_alerts...")
    try:
        inventory_alerts = get_inventory_aging_alerts(inventory_df, threshold_days=30)
        logger.info(f"Inventory alerts success: Found {len(inventory_alerts)} alerts")
        print("\nInventory Aging Alerts Summary:")
        print(inventory_alerts.head())
    except Exception as e:
        logger.error(f"Inventory alerts test failed: {e}")
    
    # Test 3: Full Sales Performance
    logger.info("Testing compute_sales_performance...")
    try:
        sales_insight = compute_sales_performance(sales_df)
        rep_count = len(sales_insight.get('rep_metrics', []))
        logger.info(f"Full sales insight success: Analyzed {rep_count} sales reps")
        print("\nFull Sales Performance Insight:")
        for key, value in sales_insight.items():
            if key != 'rep_metrics' and key != 'insights' and key != 'time_based':
                print(f"{key}: {value}")
        print(f"Generated {len(sales_insight.get('insights', []))} insights")
    except Exception as e:
        logger.error(f"Full sales insight test failed: {e}")
    
    # Test 4: Full Inventory Anomalies
    logger.info("Testing compute_inventory_anomalies...")
    try:
        inventory_insight = compute_inventory_anomalies(inventory_df)
        outlier_count = len(inventory_insight.get('outliers', []))
        logger.info(f"Full inventory insight success: Found {outlier_count} outliers")
        print("\nFull Inventory Anomalies Insight:")
        for key, value in inventory_insight.items():
            if key != 'outliers' and key != 'insights' and key != 'all_outliers' and key != 'model_metrics':
                print(f"{key}: {value}")
        print(f"Generated {len(inventory_insight.get('insights', []))} insights")
    except Exception as e:
        logger.error(f"Full inventory insight test failed: {e}")
    
    logger.info("Smoke test complete.")

if __name__ == "__main__":
    run_smoke_test()