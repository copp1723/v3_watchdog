import pytest
import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path to allow imports like src.insight_conversation
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.insight_conversation import ConversationManager
from src.validators.validator_service import process_uploaded_file
from src.validators.validation_profile import get_available_profiles

# --- Test Configuration ---
# Use a known good test file with relevant columns
TEST_DATA_FILE = "test_data_complete.csv" 
TEST_PROFILE_ID = "default" # Use the default profile for context
PROFILES_DIR = "profiles"
# Set to False to run actual LLM calls (requires API key env var)
# Set to True to use mock responses
USE_MOCK_LLM = True 

# --- High-Value Dealership Questions ---
PROMPT_BANK = [
    "What is the average gross profit per deal?",
    "Which LeadSource generated the most deals?",
    "Identify vehicles with negative gross profit.",
    "Are there any duplicate VINs in the dataset? If so, list them.",
    "What is the distribution of sales by Vehicle Year?",
    "Which SalesRepName had the highest total gross profit?",
    "Provide a summary of deals with missing VIN numbers.",
    "What percentage of deals are missing a LeadSource?",
    "Calculate the average selling price for Acura RDX models.",
    "Are there any deals with an APR significantly higher than 15%?"
    # Add more questions as needed
]

# --- Test Fixture for ConversationManager ---
@pytest.fixture(scope="module")
def setup_conversation_manager():
    """Sets up the ConversationManager and loads test data."""
    # Load the validation profile
    profiles = get_available_profiles(PROFILES_DIR)
    selected_profile = next((p for p in profiles if p.id == TEST_PROFILE_ID), None)
    if not selected_profile:
        pytest.fail(f"Test profile '{TEST_PROFILE_ID}' not found in '{PROFILES_DIR}'.")

    # Process the test data file to get context
    validation_context = {}
    processed_df = None
    report_df = None
    summary = None
    
    if not os.path.exists(TEST_DATA_FILE):
         pytest.fail(f"Test data file not found: {TEST_DATA_FILE}")

    try:
        with open(TEST_DATA_FILE, 'rb') as f:
            # Simulate file upload object
            from io import BytesIO
            file_like = BytesIO(f.read())
            file_like.name = TEST_DATA_FILE
            processed_df, summary, report_df = process_uploaded_file(
                file_like, 
                selected_profile,
                auto_clean=False # Keep flags for context
            )
        
        if processed_df is None:
            pytest.fail(f"Failed to process test data file: {summary.get('message')}")
            
        validation_context = {
            'validated_data': processed_df, # Use the potentially cleaned one
            'validation_summary': summary,
            'validation_report': report_df # Use the one with flags
        }
    except Exception as e:
        pytest.fail(f"Error setting up test data: {e}")

    # Initialize ConversationManager
    # Force mock setting based on test config
    manager = ConversationManager(use_mock=USE_MOCK_LLM)
    
    return manager, validation_context

# --- Test Function ---
# Parameterize the test to run for each prompt in the bank
@pytest.mark.parametrize("prompt", PROMPT_BANK)
def test_insight_generation_prompt_bank(setup_conversation_manager, prompt):
    """Tests insight generation for a given prompt from the bank."""
    manager, validation_context = setup_conversation_manager
    
    print(f"\n--- Testing Prompt: {prompt} ---")
    
    # Generate insight
    response = None
    try:
        response = manager.generate_insight(
            prompt,
            validation_context=validation_context,
            add_to_history=False # Don't pollute session state during tests
        )
        assert response is not None, "generate_insight returned None"
        print(f"Response Received (JSON):\n{json.dumps(response, indent=2)}")
        
        # Basic Assertions (can be expanded)
        assert "summary" in response, "Response missing 'summary' key"
        assert isinstance(response["summary"], str), "'summary' is not a string"
        assert "confidence" in response, "Response missing 'confidence' key"
        assert response["confidence"] in ["high", "medium", "low"], "Invalid 'confidence' value"
        
        # TODO: Manual Evaluation
        # After running pytest, review the logged JSON responses.
        # Mark each response as ✅ (accurate), ❌ (off-base), or ⚠️ (vague/incomplete).
        # Based on evaluation, tweak the system prompt in ConversationManager._build_enriched_prompt
        # Aim for >= 80% accuracy (✅).
        print(f"Manual Evaluation: [ ] Accurate  [ ] Off-base  [ ] Vague") 
        
    except Exception as e:
        pytest.fail(f"Error during insight generation for prompt '{prompt}': {e}")

# --- To Run This Test --- 
# 1. Ensure required environment variables are set (e.g., OPENAI_API_KEY if USE_MOCK_LLM=False)
# 2. Navigate to the project root directory in your terminal.
# 3. Run pytest: `pytest tests/test_prompt_bank.py -s` 
#    (-s flag shows print statements for easier evaluation) 