#!/usr/bin/env python
"""
Comprehensive Watchdog AI accuracy testing system.
This script tests Watchdog AI against a set of predefined questions and validates the answers.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
import sys
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Watchdog AI components
try:
    from src.watchdog_ai.insights.insight_conversation import ConversationManager
    WATCHDOG_IMPORTS_AVAILABLE = True
except ImportError:
    WATCHDOG_IMPORTS_AVAILABLE = False
    logger.warning("Watchdog AI components not available. Running in mock mode.")

# Define test questions with computation functions
class TestQuestion:
    def __init__(self, id: int, question: str, compute_fn, expected_answer: Optional[str] = None):
        """
        Initialize a test question with its computation function.
        
        Args:
            id: Question identifier number
            question: The question text
            compute_fn: Function that computes the answer from a DataFrame
            expected_answer: Optional pre-computed expected answer
        """
        self.id = id
        self.question = question
        self.compute_fn = compute_fn
        self.expected_answer = expected_answer
    
    def compute_answer(self, df: pd.DataFrame) -> str:
        """Compute the answer using the provided function."""
        try:
            return self.compute_fn(df)
        except Exception as e:
            logger.error(f"Error computing answer for question {self.id}: {str(e)}")
            return f"ERROR: {str(e)}"

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

# Define computation functions for each question
def q1_compute(df):
    """Which lead source produced the most sales?"""
    lead_counts = df['lead_source'].value_counts()
    top_lead = lead_counts.idxmax()
    count = lead_counts.max()
    return f"{top_lead} produced the most sales with {count} vehicles sold."

def q2_compute(df):
    """What was the total profit across all vehicle sales?"""
    total_profit = df['profit'].sum()
    return f"The total profit across all vehicle sales is ${total_profit:,.0f}."

def q3_compute(df):
    """Who is the top performing sales representative based on total profit?"""
    sales_profit = df.groupby('sales_rep_name')['profit'].sum().sort_values(ascending=False)
    top_rep = sales_profit.index[0]
    top_profit = sales_profit.iloc[0]
    return f"{top_rep} is the top performing sales rep with ${top_profit:,.0f} in total profit."

def q4_compute(df):
    """What is the average number of days it takes to close a sale?"""
    avg_days = df['days_to_close'].mean()
    return f"The average number of days to close a sale is {avg_days:.2f} days."

def q5_compute(df):
    """Which vehicle make has the highest average selling price?"""
    avg_prices = df.groupby('vehicle_make')['sold_price'].mean().sort_values(ascending=False)
    top_make = avg_prices.index[0]
    top_price = avg_prices.iloc[0]
    return f"{top_make} has the highest average selling price at ${top_price:,.0f}."

def q6_compute(df):
    """Which lead source generated the highest average profit for vehicle sales?"""
    avg_profit = df.groupby('lead_source')['profit'].mean().sort_values(ascending=False)
    top_source = avg_profit.index[0]
    top_avg_profit = avg_profit.iloc[0]
    return f"{top_source} generated the highest average profit at ${top_avg_profit:,.0f} per sale."

def q7_compute(df):
    """What is the total profit made by each sales representative?"""
    sales_profit = df.groupby('sales_rep_name')['profit'].sum().sort_values(ascending=False)
    result = ", ".join([f"{rep}: ${profit:,.0f}" for rep, profit in sales_profit.items()])
    return result + "."

def q8_compute(df):
    """How many vehicles were sold by each vehicle make?"""
    make_counts = df['vehicle_make'].value_counts().sort_values(ascending=False)
    result = ", ".join([f"{make}: {count}" for make, count in make_counts.items()])
    return result + "."

def q9_compute(df):
    """Which vehicle model took the longest to close, and how many days did it take?"""
    longest_idx = df['days_to_close'].idxmax()
    model = df.loc[longest_idx, 'vehicle_model']
    make = df.loc[longest_idx, 'vehicle_make']
    days = df.loc[longest_idx, 'days_to_close']
    return f"{make} {model} took the longest to close at {days} days."

def q10_compute(df):
    """What is the average days to close for sales from NeoIdentity leads?"""
    neo_days = df[df['lead_source'] == 'NeoIdentity']['days_to_close'].mean()
    return f"Sales from NeoIdentity leads took an average of {neo_days:.0f} days to close."

def q11_compute(df):
    """Which sales rep had the highest total expenses, and what was the amount?"""
    expenses = df.groupby('sales_rep_name')['expense'].sum().sort_values(ascending=False)
    top_rep = expenses.index[0]
    top_expense = expenses.iloc[0]
    return f"{top_rep} had the highest total expenses at ${top_expense:,.0f}."

def q12_compute(df):
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

def q13_compute(df):
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

def q14_compute(df):
    """How does the average profit vary by vehicle year?"""
    avg_profit_by_year = df.groupby('vehicle_year')['profit'].mean().sort_values(ascending=False)
    result = ", ".join([f"{year}: ${profit:,.0f}" for year, profit in avg_profit_by_year.items()])
    return result + "."

def q15_compute(df):
    """Which combination of vehicle make and model had the highest single profit, and who was the sales rep?"""
    highest_idx = df['profit'].idxmax()
    make = df.loc[highest_idx, 'vehicle_make']
    model = df.loc[highest_idx, 'vehicle_model']
    year = df.loc[highest_idx, 'vehicle_year']
    profit = df.loc[highest_idx, 'profit']
    rep = df.loc[highest_idx, 'sales_rep_name']
    return f"The {year} {make} {model} had the highest single profit of ${profit:,.0f}, sold by {rep}."

def normalize_text(text):
    """Normalize text for fuzzy matching."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common stopwords and variations
    text = re.sub(r'\b(the|a|an|is|was|were|had|has|have|with|at|by|for|in|of)\b', '', text)
    # Normalize number formats
    text = re.sub(r'\$?([\d,]+)\.?0+', r'\1', text)
    return text

def is_answer_correct(expected, actual):
    """Compare expected and actual answers, allowing for formatting differences."""
    # Normalize both answers
    normalized_expected = normalize_text(expected)
    normalized_actual = normalize_text(actual)
    
    # Check if all key information from expected is in actual
    return normalized_expected in normalized_actual

class WatchdogTester:
    """Tester class for Watchdog AI."""
    
    def __init__(self, dataset_path, use_mock=False):
        """
        Initialize the tester.
        
        Args:
            dataset_path: Path to the test dataset CSV
            use_mock: Whether to use mock mode instead of actual Watchdog AI
        """
        self.dataset_path = dataset_path
        self.use_mock = use_mock
        self.results = []
        
        # Initialize Watchdog AI components if available
        if WATCHDOG_IMPORTS_AVAILABLE and not use_mock:
            self.conversation_manager = ConversationManager(use_mock=False)
        else:
            self.conversation_manager = None
            logger.warning("Running in mock mode - will only calculate expected answers")
        
        # Define test questions
        self.test_questions = [
            TestQuestion(1, "Which lead source produced the most sales?", q1_compute),
            TestQuestion(2, "What was the total profit across all vehicle sales?", q2_compute),
            TestQuestion(3, "Who is the top performing sales representative based on total profit?", q3_compute),
            TestQuestion(4, "What is the average number of days it takes to close a sale?", q4_compute),
            TestQuestion(5, "Which vehicle make has the highest average selling price?", q5_compute),
            TestQuestion(6, "Which lead source generated the highest average profit for vehicle sales?", q6_compute),
            TestQuestion(7, "What is the total profit made by each sales representative?", q7_compute),
            TestQuestion(8, "How many vehicles were sold by each vehicle make?", q8_compute),
            TestQuestion(9, "Which vehicle model took the longest to close, and how many days did it take?", q9_compute),
            TestQuestion(10, "What is the average days to close for sales from NeoIdentity leads?", q10_compute),
            TestQuestion(11, "Which sales rep had the highest total expenses, and what was the amount?", q11_compute),
            TestQuestion(12, "What is the profit margin (profit/sold_price) for each vehicle sold in 2022?", q12_compute),
            TestQuestion(13, "Which lead source had the most sales for vehicles priced above $50,000 (listing price)?", q13_compute),
            TestQuestion(14, "How does the average profit vary by vehicle year?", q14_compute),
            TestQuestion(15, "Which combination of vehicle make and model had the highest single profit, and who was the sales rep?", q15_compute)
        ]
    
    def load_dataset(self):
        """Load and preprocess the dataset."""
        try:
            # Load CSV file
            df = pd.read_csv(self.dataset_path)
            
            # Process numeric columns
            df = clean_and_convert_df(df)
            
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
            raise
    
    def run_tests(self, single_question_id=None):
        """
        Run all tests or a single test if specified.
        
        Args:
            single_question_id: Optional ID of a single question to test
        """
        try:
            # Load dataset
            logger.info(f"Loading dataset from {self.dataset_path}")
            df = self.load_dataset()
            logger.info(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
            
            # Filter questions if single_question_id is specified
            questions_to_test = [q for q in self.test_questions if single_question_id is None or q.id == single_question_id]
            
            for question in questions_to_test:
                logger.info(f"Processing question {question.id}: {question.question}")
                
                # Calculate expected answer
                expected_answer = question.compute_answer(df)
                logger.info(f"Expected answer: {expected_answer}")
                
                # Get answer from Watchdog AI if available
                if self.conversation_manager is not None:
                    try:
                        start_time = time.time()
                        response = self.conversation_manager.process_query(question.question, df)
                        processing_time = time.time() - start_time
                        
                        # Extract actual answer from response
                        actual_answer = response.get("summary", "")
                        logger.info(f"Actual answer: {actual_answer}")
                        
                        # Check if answer is correct
                        is_correct = is_answer_correct(expected_answer, actual_answer)
                        result_status = "PASS" if is_correct else "FAIL"
                        
                        # Save result
                        self.results.append({
                            "id": question.id,
                            "question": question.question,
                            "expected_answer": expected_answer,
                            "actual_answer": actual_answer,
                            "is_correct": is_correct,
                            "status": result_status,
                            "processing_time": processing_time,
                            "full_response": response
                        })
                        
                        logger.info(f"Result: {result_status} (Time: {processing_time:.2f}s)")
                        
                    except Exception as e:
                        logger.error(f"Error getting answer from Watchdog AI: {str(e)}", exc_info=True)
                        self.results.append({
                            "id": question.id,
                            "question": question.question,
                            "expected_answer": expected_answer,
                            "actual_answer": f"ERROR: {str(e)}",
                            "is_correct": False,
                            "status": "ERROR",
                            "processing_time": 0,
                            "full_response": None
                        })
                else:
                    # In mock mode, just record the expected answer
                    self.results.append({
                        "id": question.id,
                        "question": question.question,
                        "expected_answer": expected_answer,
                        "actual_answer": "MOCK MODE - No actual answer",
                        "is_correct": False,
                        "status": "MOCK",
                        "processing_time": 0,
                        "full_response": None
                    })
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}", exc_info=True)
            raise
    
    def generate_report(self, output_file="watchdog_test_report.md"):
        """Generate a markdown report of the test results."""
        if not self.results:
            logger.warning("No results to report. Run tests first.")
            return
        
        try:
            # Calculate summary statistics
            total = len(self.results)
            passed = sum(1 for r in self.results if r["status"] == "PASS")
            failed = sum(1 for r in self.results if r["status"] == "FAIL")
            errors = sum(1 for r in self.results if r["status"] == "ERROR")
            
            # Create report
            with open(output_file, 'w') as f:
                f.write("# Watchdog AI Accuracy Test Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write summary
                f.write("## Summary\n\n")
                f.write(f"- Dataset: `{os.path.basename(self.dataset_path)}`\n")
                f.write(f"- Total Questions: {total}\n")
                f.write(f"- Passed: {passed} ({passed/total*100:.1f}%)\n")
                f.write(f"- Failed: {failed} ({failed/total*100:.1f}%)\n")
                f.write(f"- Errors: {errors} ({errors/total*100:.1f}%)\n\n")
                
                # Write detailed results
                f.write("## Detailed Results\n\n")
                
                for result in sorted(self.results, key=lambda x: x["id"]):
                    status_emoji = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
                    
                    f.write(f"### Question {result['id']}: {result['question']} {status_emoji}\n\n")
                    f.write(f"**Status:** {result['status']}\n\n")
                    
                    if "processing_time" in result and result["processing_time"] > 0:
                        f.write(f"**Processing Time:** {result['processing_time']:.2f} seconds\n\n")
                    
                    f.write(f"**Expected Answer:**\n{result['expected_answer']}\n\n")
                    f.write(f"**Actual Answer:**\n{result['actual_answer']}\n\n")
                    
                    # Include analysis for failed questions
                    if result["status"] == "FAIL":
                        f.write("**Analysis:**\n")
                        f.write("Key differences between expected and actual answers:\n\n")
                        
                        # Highlight missing info in the actual answer
                        expected_normalized = normalize_text(result['expected_answer'])
                        actual_normalized = normalize_text(result['actual_answer'])
                        
                        # Extract keywords from expected answer
                        expected_keywords = set(re.findall(r'\b\w+\b', expected_normalized))
                        actual_keywords = set(re.findall(r'\b\w+\b', actual_normalized))
                        
                        missing_keywords = expected_keywords - actual_keywords
                        if missing_keywords:
                            f.write(f"Missing keywords in actual answer: {', '.join(missing_keywords)}\n\n")
                    
                    f.write("---\n\n")
                
                # Write recommendations
                f.write("## Recommendations\n\n")
                
                if failed > 0 or errors > 0:
                    f.write("Based on the test results, the following improvements should be considered:\n\n")
                    
                    if failed > 0:
                        failed_questions = [r for r in self.results if r["status"] == "FAIL"]
                        f.write("### Failed Questions Analysis\n\n")
                        
                        for failed in failed_questions:
                            f.write(f"- **Question {failed['id']}**: Ensure the model correctly understands and calculates {failed['question'].lower()}\n")
                    
                    if errors > 0:
                        f.write("\n### Error Analysis\n\n")
                        f.write("- Fix errors in the processing pipeline to handle all question types reliably\n")
                        f.write("- Add better error handling and fallback mechanisms\n")
                else:
                    f.write("✅ All tests passed! No immediate improvements needed.\n")
            
            logger.info(f"Report generated at {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            raise

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test Watchdog AI accuracy on predefined questions")
    parser.add_argument("--dataset", type=str, default="/Users/joshcopp/Downloads/watchdog dummy data - Sheet2.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--output", type=str, default="watchdog_test_report.md",
                        help="Path for the output report file")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (calculate expected answers only)")
    parser.add_argument("--question", type=int, help="Test a single question by ID (1-15)")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = WatchdogTester(args.dataset, use_mock=args.mock)
        
        # Run tests
        logger.info("Starting Watchdog AI accuracy tests")
        results = tester.run_tests(single_question_id=args.question)
        
        # Generate report
        report_file = tester.generate_report(args.output)
        
        # Print summary to console
        if not args.mock:
            total = len(results)
            passed = sum(1 for r in results if r["status"] == "PASS")
            print(f"\nTest Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
            print(f"Detailed report saved to: {report_file}")
        else:
            print(f"\nMock mode: Expected answers saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())