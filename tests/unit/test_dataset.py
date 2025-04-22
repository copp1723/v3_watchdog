#!/usr/bin/env python
"""
Simple test script to load the dataset and verify it can be analyzed correctly.
"""

import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_currency(s):
    """Clean currency string to numeric value."""
    if isinstance(s, str):
        return float(s.replace('$', '').replace(',', ''))
    return float(s)

def clean_and_convert_df(df):
    """Convert all currency and numeric columns to proper format"""
    df_clean = df.copy()
    
    # Convert currency columns (those with $ sign)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if this column contains currency
            sample = df[col].astype(str).iloc[0]
            if '$' in sample:
                df_clean[col] = df[col].astype(str).str.replace('$', '', regex=False) \
                                      .str.replace(',', '', regex=False) \
                                      .astype(float)
    
    # Convert specific columns we know should be numeric
    numeric_cols = ['listing_price', 'sold_price', 'profit', 'expense', 'days_to_close']
    for col in numeric_cols:
        if col in df.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
            except:
                logger.warning(f"Could not convert {col} to numeric")
    
    return df_clean

def main():
    """Load dataset and run basic analysis to verify data"""
    try:
        # Load dataset
        dataset_path = "/Users/joshcopp/Downloads/watchdog dummy data - Sheet2.csv"
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at: {dataset_path}")
            return
        
        logger.info(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        logger.info(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display raw data
        print("\n=== Raw Data Sample ===")
        print(df.head())
        
        # Display column information 
        print("\n=== Column Info ===")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
        
        # Clean and convert data
        df_clean = clean_and_convert_df(df)
        logger.info("Dataset cleaned and converted")
        
        # Display cleaned data
        print("\n=== Cleaned Data Sample ===")
        print(df_clean.head())
        
        # Basic statistics
        print("\n=== Basic Statistics ===")
        print(f"Total vehicles: {len(df_clean)}")
        print(f"Total profit: ${df_clean['profit'].sum():,.2f}")
        print(f"Average selling price: ${df_clean['sold_price'].mean():,.2f}")
        print(f"Average days to close: {df_clean['days_to_close'].mean():.2f}")
        
        # Lead source summary
        print("\n=== Lead Source Summary ===")
        lead_summary = df_clean.groupby('lead_source').agg({
            'lead_source': 'count',
            'profit': 'sum',
            'sold_price': 'mean',
            'days_to_close': 'mean'
        }).rename(columns={'lead_source': 'count'})
        print(lead_summary)
        
        # Sales rep summary
        print("\n=== Sales Rep Summary ===")
        sales_rep_summary = df_clean.groupby('sales_rep_name').agg({
            'profit': 'sum',
            'expense': 'sum'
        }).sort_values('profit', ascending=False)
        print(sales_rep_summary)
        
        # Vehicle make summary
        print("\n=== Vehicle Make Summary ===")
        make_summary = df_clean.groupby('vehicle_make').agg({
            'vehicle_make': 'count',
            'sold_price': 'mean',
            'profit': 'mean'
        }).rename(columns={'vehicle_make': 'count'})
        print(make_summary)
        
        # Year summary
        print("\n=== Vehicle Year Summary ===")
        year_summary = df_clean.groupby('vehicle_year').agg({
            'vehicle_year': 'count',
            'profit': 'mean'
        }).rename(columns={'vehicle_year': 'count'})
        print(year_summary)
        
        logger.info("Dataset analysis complete")
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()