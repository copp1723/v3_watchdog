#!/usr/bin/env python
"""
Test script for Watchdog AI questions with predefined answers to validate accuracy.
"""

import pandas as pd
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Watchdog AI components
from watchdog_ai.insights.insight_conversation import ConversationManager
from watchdog_ai.utils.adaptive_schema import AdaptiveSchema
from watchdog_ai.insights.utils import validate_numeric_columns

# Define test questions and expected answers
TEST_QUESTIONS = [
    {
        "question": "Which lead source produced the most sales?",
        "expected_answer": "NeoIdentity produced the most sales with 4 vehicles sold."
    },
    {
        "question": "What was the total profit across all vehicle sales?",
        "expected_answer": "The total profit across all vehicle sales is $30,900."
    },
    {
        "question": "Who is the top performing sales representative based on total profit?",
        "expected_answer": "Jane Smith is the top performing sales rep with $7,800 in total profit."
    },
    {
        "question": "What is the average number of days it takes to close a sale?",
        "expected_answer": "The average number of days to close a sale is 19.38 days."
    },
    {
        "question": "Which vehicle make has the highest average selling price?",
        "expected_answer": "GMC has the highest average selling price at $46,000."
    },
    {
        "question": "Which lead source generated the highest average profit for vehicle sales?",
        "expected_answer": "NeoIdentity generated the highest average profit at $1,950 per sale."
    },
    {
        "question": "What is the total profit made by each sales representative?",
        "expected_answer": "Jane Smith: $7,800, John Doe: $5,400, Mike Wilson: $5,000, Tom Roberts: $4,000, Alex Johnson: $2,700, David Miller: $2,400, Sam Brown: $1,700, Laura Garcia: $1,500, Karen Davis: $1,200, Maria Lopez: $1,100."
    },
    {
        "question": "How many vehicles were sold by each vehicle make?",
        "expected_answer": "GMC: 4, Honda: 3, Toyota: 3, Nissan: 2, Buick: 2, Jeep: 2, Chevrolet: 1, Ford: 1."
    },
    {
        "question": "Which vehicle model took the longest to close, and how many days did it take?",
        "expected_answer": "Ford F-150 took the longest to close at 99 days."
    },
    {
        "question": "What is the average days to close for sales from NeoIdentity leads?",
        "expected_answer": "Sales from NeoIdentity leads took an average of 19 days to close."
    },
    {
        "question": "Which sales rep had the highest total expenses, and what was the amount?",
        "expected_answer": "John Doe had the highest total expenses at $2,250."
    },
    {
        "question": "What is the profit margin (profit/sold_price) for each vehicle sold in 2022?",
        "expected_answer": "2022 GMC Sierra: 5.98%, 2022 GMC Yukon: 5.74%, 2022 Nissan Rogue: 4.58%, 2022 Ford F-150: 3.44%. Average profit margin for 2022 vehicles: 4.93%."
    },
    {
        "question": "Which lead source had the most sales for vehicles priced above $50,000 (listing price)?",
        "expected_answer": "NeoIdentity, AutoTrader, and GM Financial each had 1 sale for vehicles priced above $50,000."
    },
    {
        "question": "How does the average profit vary by vehicle year?",
        "expected_answer": "2023: $3,100, 2022: $2,325, 2021: $1,225, 2020: $1,250, 2019: $1,567, 2018: $1,867, 2017: $900."
    },
    {
        "question": "Which combination of vehicle make and model had the highest single profit, and who was the sales rep?",
        "expected_answer": "The 2022 GMC Sierra had the highest single profit of $3,200, sold by John Doe."
    }
]

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert numeric columns that may be formatted as strings.
    """
    # Clone the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # Process numeric columns with dollar signs or commas
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert currency strings to numeric
            try:
                # Remove $ and commas, then convert to numeric
                df_clean[col] = df[col].str.replace('$', '', regex=False) \
                                       .str.replace(',', '', regex=False) \
                                       .replace('', None) \
                                       .astype(float)
                logger.info(f"Converted column {col} to numeric")
            except:
                # If conversion fails, keep as is
                pass
    
    return df_clean

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Process numeric columns
    df = clean_numeric_columns(df)
    
    # Validate numeric columns
    df = validate_numeric_columns(df)
    
    return df

def test_questions(df: pd.DataFrame, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Test each question against the Watchdog AI system and compare with expected answers.
    """
    results = []
    conversation_manager = ConversationManager(use_mock=False)
    
    for idx, test_case in enumerate(questions):
        question = test_case["question"]
        expected = test_case["expected_answer"]
        
        logger.info(f"Testing question {idx+1}/{len(questions)}: {question}")
        
        try:
            # Process query using Watchdog AI
            response = conversation_manager.process_query(question, df)
            
            # Extract summary from response
            actual_answer = response.get("summary", "")
            
            # Calculate similarity or match
            is_match = expected.lower() in actual_answer.lower()
            
            # Log result
            status = "PASS" if is_match else "FAIL"
            logger.info(f"Result: {status}")
            
            # Store detailed results
            results.append({
                "question": question,
                "expected": expected,
                "actual": actual_answer,
                "result": status,
                "full_response": response
            })
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            results.append({
                "question": question,
                "expected": expected,
                "actual": f"ERROR: {str(e)}",
                "result": "ERROR",
                "full_response": None
            })
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save test results to a JSON file.
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "passed": sum(1 for r in results if r["result"] == "PASS"),
        "failed": sum(1 for r in results if r["result"] == "FAIL"),
        "errors": sum(1 for r in results if r["result"] == "ERROR"),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def generate_markdown_report(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Generate a markdown report of the test results.
    """
    passed = sum(1 for r in results if r["result"] == "PASS")
    failed = sum(1 for r in results if r["result"] == "FAIL")
    errors = sum(1 for r in results if r["result"] == "ERROR")
    total = len(results)
    
    with open(output_file, 'w') as f:
        f.write("# Watchdog AI Test Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total Questions: {total}\n")
        f.write(f"- Passed: {passed} ({passed/total*100:.1f}%)\n")
        f.write(f"- Failed: {failed} ({failed/total*100:.1f}%)\n")
        f.write(f"- Errors: {errors} ({errors/total*100:.1f}%)\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for idx, result in enumerate(results):
            status_emoji = "✅" if result["result"] == "PASS" else "❌" if result["result"] == "FAIL" else "⚠️"
            f.write(f"### {idx+1}. {result['question']} {status_emoji}\n\n")
            f.write(f"**Expected Answer:** {result['expected']}\n\n")
            f.write(f"**Actual Answer:** {result['actual']}\n\n")
            f.write(f"**Result:** {result['result']}\n\n")
            f.write("---\n\n")
    
    logger.info(f"Markdown report saved to {output_file}")

def main():
    """
    Main function to run the tests.
    """
    try:
        # Dataset path
        dataset_path = "/Users/joshcopp/Downloads/watchdog dummy data - Sheet2.csv"
        
        # Output paths
        results_path = "test_results.json"
        report_path = "test_report.md"
        
        # Load the dataset
        logger.info(f"Loading dataset from {dataset_path}")
        df = load_dataset(dataset_path)
        logger.info(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        
        # Run tests
        logger.info(f"Running {len(TEST_QUESTIONS)} test questions")
        results = test_questions(df, TEST_QUESTIONS)
        
        # Save results
        save_results(results, results_path)
        generate_markdown_report(results, report_path)
        
        # Display summary
        passed = sum(1 for r in results if r["result"] == "PASS")
        logger.info(f"Test completed: {passed}/{len(results)} passed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()