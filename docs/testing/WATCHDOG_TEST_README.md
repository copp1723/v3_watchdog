# Watchdog AI Testing Suite

This directory contains a comprehensive testing suite for validating the accuracy of Watchdog AI with the provided test dataset. The suite includes tools for testing, analyzing, and improving the system's responses to the predefined test questions.

## Test Dataset

The test dataset is located at: `/Users/joshcopp/Downloads/watchdog dummy data - Sheet2.csv`

This dataset contains car sales data with the following columns:
- lead_source: Source of the sales lead
- listing_price: Original price of the vehicle
- sold_price: Final selling price
- profit: Profit made on the sale
- expense: Expenses associated with the sale
- sales_rep_name: Name of the sales representative
- vehicle_year: Year of the vehicle
- vehicle_make: Make of the vehicle
- vehicle_model: Model of the vehicle
- days_to_close: Number of days to close the sale

## Test Questions

The system should be able to accurately answer the following 15 questions:

1. Which lead source produced the most sales?
2. What was the total profit across all vehicle sales?
3. Who is the top performing sales representative based on total profit?
4. What is the average number of days it takes to close a sale?
5. Which vehicle make has the highest average selling price?
6. Which lead source generated the highest average profit for vehicle sales?
7. What is the total profit made by each sales representative?
8. How many vehicles were sold by each vehicle make?
9. Which vehicle model took the longest to close, and how many days did it take?
10. What is the average days to close for sales from NeoIdentity leads?
11. Which sales rep had the highest total expenses, and what was the amount?
12. What is the profit margin (profit/sold_price) for each vehicle sold in 2022?
13. Which lead source had the most sales for vehicles priced above $50,000 (listing price)?
14. How does the average profit vary by vehicle year?
15. Which combination of vehicle make and model had the highest single profit, and who was the sales rep?

## Testing Scripts

### 1. `test_dataset.py`

This script loads and verifies the test dataset, performing basic analysis to ensure the data is properly loaded and can be analyzed correctly.

**Usage:**
```bash
python test_dataset.py
```

### 2. `direct_answer_calculator.py`

This script directly calculates the correct answers to all test questions using pure data analysis methods, without using the Watchdog AI system. This establishes the ground truth for our tests.

**Usage:**
```bash
python direct_answer_calculator.py
```

### 3. `test_watchdog_questions.py`

This script tests the Watchdog AI system against the test questions and compares the results with the expected answers.

**Usage:**
```bash
python test_watchdog_questions.py
```

### 4. `watchdog_accuracy_tester.py`

This is a comprehensive testing tool with command-line arguments for more flexible testing. It can run in mock mode (just calculating expected answers) or test the actual Watchdog AI system.

**Usage:**
```bash
# Test all questions
python watchdog_accuracy_tester.py

# Test a specific question
python watchdog_accuracy_tester.py --question 5

# Run in mock mode (calculate expected answers only)
python watchdog_accuracy_tester.py --mock

# Specify custom dataset path
python watchdog_accuracy_tester.py --dataset /path/to/dataset.csv

# Custom output report path
python watchdog_accuracy_tester.py --output custom_report.md
```

### 5. `prompt_optimizer.py`

This script generates optimized prompts for the LLM engine to improve accuracy on the test questions. It creates an optimized version of the LLM engine and a guide for prompt improvements.

**Usage:**
```bash
python prompt_optimizer.py

# Custom output paths
python prompt_optimizer.py --llm-output custom_llm_engine.py --guide-output custom_guide.md
```

## Usage Instructions

To validate and improve Watchdog AI accuracy:

1. First, verify the dataset can be loaded correctly:
   ```bash
   python test_dataset.py
   ```

2. Calculate the correct answers for all test questions:
   ```bash
   python direct_answer_calculator.py
   ```

3. Run the accuracy tester to see how well Watchdog AI performs:
   ```bash
   python watchdog_accuracy_tester.py
   ```

4. Generate optimized prompts to improve accuracy:
   ```bash
   python prompt_optimizer.py
   ```

5. Implement the optimized LLM engine:
   - Backup the original LLM engine: `cp src/watchdog_ai/llm/llm_engine.py src/watchdog_ai/llm/llm_engine.py.bak`
   - Replace with optimized version: `cp src/watchdog_ai/llm/llm_engine_optimized.py src/watchdog_ai/llm/llm_engine.py`

6. Run the Watchdog AI application with the optimized LLM engine:
   ```bash
   ./run.sh
   ```

7. Verify improvements by running the accuracy tester again:
   ```bash
   python watchdog_accuracy_tester.py
   ```

## Files Generated

- `test_results.json`: Detailed JSON test results
- `test_report.md`: Markdown report of test results
- `watchdog_test_report.md`: Comprehensive test report from the accuracy tester
- `watchdog_prompt_guide.md`: Guide for prompt optimization
- `src/watchdog_ai/llm/llm_engine_optimized.py`: Optimized LLM engine with special handling for test questions

## Troubleshooting

If you encounter issues:

1. Check that the dataset path is correct
2. Ensure all required dependencies are installed
3. Check that the Watchdog AI system is properly configured
4. Look for error messages in the test output and logs

For detailed logs, add the `--log-level=DEBUG` flag to the Python commands.