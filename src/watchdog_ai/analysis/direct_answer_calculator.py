#!/usr/bin/env python
"""
Direct calculator for Watchdog AI test questions with exact answer calculations.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_and_convert_df(df):
    """Convert all currency and numeric columns to proper format"""
    df_clean = df.copy()
    
    # Convert currency columns (those with $ sign)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if this column contains currency
if df[col].astype(str).str.contains(r'\$').any():
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

def question_1(df):
    """Which lead source produced the most sales?"""
    lead_counts = df['lead_source'].value_counts()
    top_lead = lead_counts.idxmax()
    count = lead_counts.max()
    return f"{top_lead} produced the most sales with {count} vehicles sold."

def question_2(df):
    """What was the total profit across all vehicle sales?"""
    total_profit = df['profit'].sum()
    return f"The total profit across all vehicle sales is ${total_profit:,.0f}."

def question_3(df):
    """Who is the top performing sales representative based on total profit?"""
    sales_profit = df.groupby('sales_rep_name')['profit'].sum().sort_values(ascending=False)
    top_rep = sales_profit.index[0]
    top_profit = sales_profit.iloc[0]
    return f"{top_rep} is the top performing sales rep with ${top_profit:,.0f} in total profit."

def question_4(df):
    """What is the average number of days it takes to close a sale?"""
    avg_days = df['days_to_close'].mean()
    return f"The average number of days to close a sale is {avg_days:.2f} days."

def question_5(df):
    """Which vehicle make has the highest average selling price?"""
    avg_prices = df.groupby('vehicle_make')['sold_price'].mean().sort_values(ascending=False)
    top_make = avg_prices.index[0]
    top_price = avg_prices.iloc[0]
    return f"{top_make} has the highest average selling price at ${top_price:,.0f}."

def question_6(df):
    """Which lead source generated the highest average profit for vehicle sales?"""
    avg_profit = df.groupby('lead_source')['profit'].mean().sort_values(ascending=False)
    top_source = avg_profit.index[0]
    top_avg_profit = avg_profit.iloc[0]
    return f"{top_source} generated the highest average profit at ${top_avg_profit:,.0f} per sale."

def question_7(df):
    """What is the total profit made by each sales representative?"""
    sales_profit = df.groupby('sales_rep_name')['profit'].sum().sort_values(ascending=False)
    result = ", ".join([f"{rep}: ${profit:,.0f}" for rep, profit in sales_profit.items()])
    return result + "."

def question_8(df):
    """How many vehicles were sold by each vehicle make?"""
    make_counts = df['vehicle_make'].value_counts().sort_values(ascending=False)
    result = ", ".join([f"{make}: {count}" for make, count in make_counts.items()])
    return result + "."

def question_9(df):
    """Which vehicle model took the longest to close, and how many days did it take?"""
    longest_idx = df['days_to_close'].idxmax()
    model = df.loc[longest_idx, 'vehicle_model']
    make = df.loc[longest_idx, 'vehicle_make']
    days = df.loc[longest_idx, 'days_to_close']
    return f"{make} {model} took the longest to close at {days} days."

def question_10(df):
    """What is the average days to close for sales from NeoIdentity leads?"""
    neo_days = df[df['lead_source'] == 'NeoIdentity']['days_to_close'].mean()
    return f"Sales from NeoIdentity leads took an average of {neo_days:.0f} days to close."

def question_11(df):
    """Which sales rep had the highest total expenses, and what was the amount?"""
    expenses = df.groupby('sales_rep_name')['expense'].sum().sort_values(ascending=False)
    top_rep = expenses.index[0]
    top_expense = expenses.iloc[0]
    return f"{top_rep} had the highest total expenses at ${top_expense:,.0f}."

def question_12(df):
    """What is the profit margin (profit/sold_price) for each vehicle sold in 2022?"""
    # Filter to 2022 vehicles and calculate profit margin
    df_2022 = df[df['vehicle_year'] == 2022].copy()
    df_2022['profit_margin'] = (df_2022['profit'] / df_2022['sold_price']) * 100
    
    # Format each vehicle's profit margin
    margins = []
    for _, row in df_2022.iterrows():
        margins.append(f"{row['vehicle_year']} {row['vehicle_make']} {row['vehicle_model']}: {row['profit_margin']:.2f}%")
    
    # Calculate average
    avg_margin = df_2022['profit_margin'].mean()
    result = ", ".join(margins)
    return f"{result}. Average profit margin for 2022 vehicles: {avg_margin:.2f}%."

def question_13(df):
    """Which lead source had the most sales for vehicles priced above $50,000 (listing price)?"""
    expensive = df[df['listing_price'] > 50000]
    if expensive.empty:
        return "No vehicles were priced above $50,000."
    
    lead_counts = expensive['lead_source'].value_counts()
    if lead_counts.max() == 1 and len(lead_counts[lead_counts == 1]) > 1:
        # Multiple sources tied with 1 each
        tied_sources = ", ".join(lead_counts[lead_counts == 1].index.tolist())
        return f"{tied_sources} each had 1 sale for vehicles priced above $50,000."
    else:
        top_source = lead_counts.idxmax()
        count = lead_counts.max()
        return f"{top_source} had the most sales for vehicles priced above $50,000 with {count} sales."

def question_14(df):
    """How does the average profit vary by vehicle year?"""
    avg_profit_by_year = df.groupby('vehicle_year')['profit'].mean().sort_values(ascending=False)
    result = ", ".join([f"{year}: ${profit:,.0f}" for year, profit in avg_profit_by_year.items()])
    return result + "."

def question_15(df):
    """Which combination of vehicle make and model had the highest single profit, and who was the sales rep?"""
    highest_idx = df['profit'].idxmax()
    make = df.loc[highest_idx, 'vehicle_make']
    model = df.loc[highest_idx, 'vehicle_model']
    year = df.loc[highest_idx, 'vehicle_year']
    profit = df.loc[highest_idx, 'profit']
    rep = df.loc[highest_idx, 'sales_rep_name']
    return f"The {year} {make} {model} had the highest single profit of ${profit:,.0f}, sold by {rep}."

def main():
    """Calculate answers to all test questions using direct data processing."""
    try:
        # Load and prepare dataset
        dataset_path = "/Users/joshcopp/Downloads/watchdog dummy data - Sheet2.csv"
        logger.info(f"Loading dataset from {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        df = clean_and_convert_df(df)
        
        logger.info(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Define questions and their corresponding functions
        questions = [
            (1, "Which lead source produced the most sales?", question_1),
            (2, "What was the total profit across all vehicle sales?", question_2),
            (3, "Who is the top performing sales representative based on total profit?", question_3),
            (4, "What is the average number of days it takes to close a sale?", question_4),
            (5, "Which vehicle make has the highest average selling price?", question_5),
            (6, "Which lead source generated the highest average profit for vehicle sales?", question_6),
            (7, "What is the total profit made by each sales representative?", question_7),
            (8, "How many vehicles were sold by each vehicle make?", question_8),
            (9, "Which vehicle model took the longest to close, and how many days did it take?", question_9),
            (10, "What is the average days to close for sales from NeoIdentity leads?", question_10),
            (11, "Which sales rep had the highest total expenses, and what was the amount?", question_11),
            (12, "What is the profit margin (profit/sold_price) for each vehicle sold in 2022?", question_12),
            (13, "Which lead source had the most sales for vehicles priced above $50,000 (listing price)?", question_13),
            (14, "How does the average profit vary by vehicle year?", question_14),
            (15, "Which combination of vehicle make and model had the highest single profit, and who was the sales rep?", question_15)
        ]
        
        # Calculate and print answers
        print("\n=== Direct Calculation Results ===\n")
        for num, question, func in questions:
            try:
                answer = func(df)
                print(f"Q{num}: {question}")
                print(f"A: {answer}")
                print()
            except Exception as e:
                logger.error(f"Error calculating answer for question {num}: {str(e)}", exc_info=True)
                print(f"Q{num}: {question}")
                print(f"ERROR: {str(e)}")
                print()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()