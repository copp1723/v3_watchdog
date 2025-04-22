import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from io import BytesIO
import streamlit as st # Import streamlit itself for potential session state access

# Assuming process_uploaded_file is the entry point we test via
# It internally calls LLMEngine, applies mapping, etc.
from src.validators.validator_service import process_uploaded_file, render_data_validation_interface
from src.validators.validator_service import ValidatorService # Import ValidatorService for render function

# Mock the LLMEngine class completely where it's used in validator_service
@patch('src.validators.validator_service.LLMEngine')
class TestDataUploadLLMMapping(unittest.TestCase):

    def setUp(self):
        """Set up common test data and state."""
        # Initialize Streamlit session state for each test
        st.session_state.clear() 
        self.sample_df_no_clarify = pd.DataFrame({
            "profit": [1000, 2500],
            "lead_source": ["Website", "Referral"],
            "sold_date": ["2023-01-15", "2023-02-20"]
        })
        # Create a mock file object
        output = BytesIO()
        self.sample_df_no_clarify.to_csv(output, index=False)
        output.seek(0)
        self.mock_file_no_clarify = MagicMock()
        self.mock_file_no_clarify.name = "test_no_clarify.csv"
        self.mock_file_no_clarify.read.return_value = output.getvalue()
        self.mock_file_no_clarify.seek.return_value = 0 # Mock seek method
        self.mock_file_no_clarify.tell.return_value = len(output.getvalue()) # Mock tell method

        self.sample_df_clarify = pd.DataFrame({
            "trade_value": [5000, 7000],
            "post_code": ["90210", "10001"],
            "vin_number": ["VIN1", "VIN2"]
        })
        output_clarify = BytesIO()
        self.sample_df_clarify.to_csv(output_clarify, index=False)
        output_clarify.seek(0)
        self.mock_file_clarify = MagicMock()
        self.mock_file_clarify.name = "test_clarify.csv"
        self.mock_file_clarify.read.return_value = output_clarify.getvalue()
        self.mock_file_clarify.seek.return_value = 0
        self.mock_file_clarify.tell.return_value = len(output_clarify.getvalue())
        
        # Mock ValidatorService needed for render_data_validation_interface
        self.mock_validator_service = MagicMock(spec=ValidatorService)
        # Mock the validate_dataframe method to return basic info
        def mock_validate_df(df):
            return df, {"total_records": len(df), "total_issues": 0, "percentage_clean": 100.0}
        self.mock_validator_service.validate_dataframe.side_effect = mock_validate_df

    def test_scenario_1_no_clarifications(self, MockLLMEngine):
        """E2E Test: Upload data with clear mapping, no clarifications needed."""
        # --- Arrange ---
        # Configure the mock LLMEngine instance and its method return value
        mock_engine_instance = MockLLMEngine.return_value
        mock_llm_response = {
            "mapping": {
                "TransactionInformation": {
                    "TotalGross": {"column": "profit", "confidence": 0.98},
                    "SaleDate": {"column": "sold_date", "confidence": 0.95}
                },
                "SalesProcessInformation": {
                    "LeadSource": {"column": "lead_source", "confidence": 1.00}
                }
                # Assume other canonical fields map to null implicitly or explicitly
            },
            "clarifications": [],
            "unmapped_columns": []
        }
        mock_engine_instance.map_columns_jeopardy.return_value = mock_llm_response

        # --- Act ---
        # Call the processing function which uses the mocked LLMEngine
        processed_df, summary, report, schema_info = process_uploaded_file(
            file=self.mock_file_no_clarify
        )

        # --- Assert ---
        # 1. LLM Engine was called correctly
        MockLLMEngine.assert_called_once() # Check if constructor was called
        mock_engine_instance.map_columns_jeopardy.assert_called_once_with(['profit', 'lead_source', 'sold_date'])
        
        # 2. Processing status indicates success (no clarification needed)
        self.assertEqual(summary.get('status'), 'success')
        self.assertEqual(summary.get('llm_mapping_clarifications', []), [])
        self.assertNotIn("original_llm_mapping", st.session_state, "Original mapping should not persist if no clarifications")

        # 3. DataFrame columns are correctly renamed
        self.assertIsNotNone(processed_df)
        expected_columns = ['TotalGross', 'LeadSource', 'SaleDate']
        self.assertListEqual(processed_df.columns.tolist(), expected_columns)
        
        # 4. Schema info reflects renamed columns
        self.assertIsNotNone(schema_info)
        self.assertListEqual(list(schema_info.keys()), expected_columns)

    def test_scenario_2_with_clarifications(self, MockLLMEngine):
        """E2E Test: Upload data needing clarifications, user confirms."""
        # --- Arrange --- 
        mock_engine_instance = MockLLMEngine.return_value
        clarification_item = {
            "column": "trade_value",
            "question": "Does 'trade_value' map to SalePrice or TradeInValue?",
            "options": ["SalePrice", "TradeInValue"]
        }
        mock_llm_response = {
            "mapping": {
                "TransactionInformation": {
                    # Low confidence for trade_value triggers clarification
                    "TradeInValue": {"column": "trade_value", "confidence": 0.6},
                    "SalePrice": {"column": "trade_value", "confidence": 0.5} 
                },
                 "CustomerInformation": {
                    "CustomerZip": {"column": "post_code", "confidence": 0.99}
                 },
                 "VehicleInformation": {
                     "VIN": {"column": "vin_number", "confidence": 0.99}
                 }
            },
            "clarifications": [clarification_item],
            "unmapped_columns": []
        }
        mock_engine_instance.map_columns_jeopardy.return_value = mock_llm_response

        # --- Act 1: Initial processing - should require clarification ---
        processed_df_initial, summary_initial, _, _ = process_uploaded_file(
            file=self.mock_file_clarify
        )
        
        # --- Assert 1: Check clarification state ---
        self.assertEqual(summary_initial.get('status'), 'needs_clarification')
        self.assertEqual(summary_initial.get('llm_mapping_clarifications'), [clarification_item])
        self.assertIn("original_llm_mapping", st.session_state, "Original mapping should be stored")
        self.assertEqual(st.session_state["original_llm_mapping"], mock_llm_response)
        # Initial DF columns should still be original names before confirmation
        self.assertListEqual(processed_df_initial.columns.tolist(), ["trade_value", "post_code", "vin_number"])

        # --- Act 2: Simulate UI confirmation --- 
        # Set user choice in session state (as if user clicked radio button)
        st.session_state.clarification_choices = {
            "trade_value": "TradeInValue" # User chooses TradeInValue
        }
        # Simulate the button click/rerun by calling render_data_validation_interface
        # We need to mock the streamlit UI elements within render_data_validation_interface
        # to avoid actual rendering and capture the state changes.
        
        # Mock streamlit elements used *during* the confirmation logic within render_data_validation_interface
        with patch('streamlit.success') as mock_st_success, \
             patch('streamlit.write') as mock_st_write, \
             patch('streamlit.info') as mock_st_info, \
             patch('streamlit.rerun') as mock_st_rerun: # Capture rerun

            # Call the render function - this should trigger the confirmation logic
            # We pass the state *after* initial processing (df_initial, summary_initial)
            # The function *mutates* df_initial and session_state based on choices
            final_df, cleaning_done = render_data_validation_interface(
                df=processed_df_initial, # Pass the df from the initial run
                validator=self.mock_validator_service, # Use the mock service
                summary=summary_initial, # Pass the summary from the initial run
                on_continue=None # No need for callback in this test focus
            )

        # --- Assert 2: Check state after confirmation --- 
        # 1. Rerun should have been called to update UI state
        mock_st_rerun.assert_called_once()
        # 2. Success message shown
        mock_st_success.assert_any_call("Columns remapped successfully based on your confirmation!")
        # 3. Clarification state should be resolved and cleared
        self.assertTrue(st.session_state.get("clarifications_resolved"))
        self.assertNotIn("clarification_choices", st.session_state)
        self.assertNotIn("original_llm_mapping", st.session_state)
        
        # 4. The DataFrame passed to render_data_validation_interface (processed_df_initial)
        #    should now have been renamed inplace according to the choice.
        #    Note: render_data_validation_interface returns the df *before* potential cleaning confirmation.
        expected_columns_after_confirm = ['TradeInValue', 'CustomerZip', 'VIN']
        self.assertListEqual(processed_df_initial.columns.tolist(), expected_columns_after_confirm)
        # Also check the df returned by the function (should be the same at this stage)
        self.assertListEqual(final_df.columns.tolist(), expected_columns_after_confirm)
        self.assertFalse(cleaning_done)

    def test_scenario_3_unmapped_columns(self, MockLLMEngine):
        """E2E Test: Upload data with columns that remain unmapped by the LLM."""
        # --- Arrange ---
        # Sample data with potentially unmappable columns
        sample_df_unmapped = pd.DataFrame({
            "vehicle_vin": ["VIN123"],
            "sale_amt": [30000],
            "CarnowCars.com": [1], # Potential lead source value
            "some_random_id": ["XYZ789"] # Likely unmappable 
        })
        output_unmapped = BytesIO()
        sample_df_unmapped.to_csv(output_unmapped, index=False)
        output_unmapped.seek(0)
        mock_file_unmapped = MagicMock()
        mock_file_unmapped.name = "test_unmapped.csv"
        mock_file_unmapped.read.return_value = output_unmapped.getvalue()
        mock_file_unmapped.seek.return_value = 0
        mock_file_unmapped.tell.return_value = len(output_unmapped.getvalue())
        
        # Configure mock LLM response identifying unmapped columns
        mock_engine_instance = MockLLMEngine.return_value
        mock_llm_response = {
            "mapping": {
                "VehicleInformation": {
                    "VIN": {"column": "vehicle_vin", "confidence": 0.99}
                },
                "TransactionInformation": {
                    "SalePrice": {"column": "sale_amt", "confidence": 0.95}
                }
            },
            "clarifications": [],
            "unmapped_columns": [
                {
                    "column": "CarnowCars.com",
                    "potential_category": "LeadSource",
                    "notes": "This looks like a specific lead-source value, not a header."
                },
                 {
                    "column": "some_random_id",
                    "potential_category": None,
                    "notes": "Column name is unrecognized and does not match known patterns."
                 } 
            ]
        }
        mock_engine_instance.map_columns_jeopardy.return_value = mock_llm_response

        # --- Act ---
        processed_df, summary, report, schema_info = process_uploaded_file(
            file=mock_file_unmapped
        )

        # --- Assert ---
        # 1. LLM Engine was called correctly
        MockLLMEngine.assert_called() # Called at least once (could be >1 if previous tests ran)
        mock_engine_instance.map_columns_jeopardy.assert_called_with(['vehicle_vin', 'sale_amt', 'CarnowCars.com', 'some_random_id'])
        
        # 2. Processing status indicates success
        self.assertEqual(summary.get('status'), 'success')
        self.assertEqual(summary.get('llm_mapping_clarifications', []), []) # No clarifications
        
        # 3. Unmapped columns are correctly reported in the summary
        expected_unmapped = mock_llm_response["unmapped_columns"]
        # Note: The logic in process_uploaded_file might add other implicitly ignored cols. 
        # We check if the *explicitly* unmapped ones from LLM are present.
        reported_unmapped = summary.get('llm_mapping_unmapped', [])
        # Check that each expected unmapped item exists in the reported list
        for expected_item in expected_unmapped:
             self.assertTrue(any(item['column'] == expected_item['column'] for item in reported_unmapped),
                             f"Expected unmapped column '{expected_item['column']}' not found in summary")
        # Check that explicitly unmapped columns were not accidentally added to schema_info with canonical names
        unmapped_col_names = {item['column'] for item in expected_unmapped}
        for col in unmapped_col_names:
             self.assertNotIn(col, schema_info, f"Explicitly unmapped column '{col}' should not be in final schema info keys")

        # 4. DataFrame columns include mapped *and* original unmapped names 
        #    (Current logic renames in place but doesn't drop unmapped cols)
        self.assertIsNotNone(processed_df)
        expected_final_columns = ['VIN', 'SalePrice', 'CarnowCars.com', 'some_random_id']
        self.assertCountEqual(processed_df.columns.tolist(), expected_final_columns, "Columns should be mapped + original unmapped")
        
        # 5. Schema info reflects the final columns (mapped + unmapped original)
        self.assertIsNotNone(schema_info)
        self.assertCountEqual(list(schema_info.keys()), expected_final_columns)

    def test_scenario_4_user_selects_unmapped(self, MockLLMEngine):
        """E2E Test: Clarification needed, user selects 'Unmapped'."""
        # --- Arrange --- 
        # Reuse sample data from scenario 2
        mock_engine_instance = MockLLMEngine.return_value
        clarification_item = {
            "column": "trade_value",
            "question": "Does 'trade_value' map to SalePrice or TradeInValue?",
            "options": ["SalePrice", "TradeInValue"]
        }
        mock_llm_response = {
            "mapping": {
                "TransactionInformation": {
                    "TradeInValue": {"column": "trade_value", "confidence": 0.6},
                    "SalePrice": {"column": "trade_value", "confidence": 0.5} 
                },
                 "CustomerInformation": {
                    "CustomerZip": {"column": "post_code", "confidence": 0.99}
                 },
                 "VehicleInformation": {
                     "VIN": {"column": "vin_number", "confidence": 0.99}
                 }
            },
            "clarifications": [clarification_item],
            "unmapped_columns": []
        }
        mock_engine_instance.map_columns_jeopardy.return_value = mock_llm_response

        # --- Act 1: Initial processing ---
        processed_df_initial, summary_initial, _, _ = process_uploaded_file(
            file=self.mock_file_clarify
        )
        
        # --- Assert 1: Check clarification state ---
        self.assertEqual(summary_initial.get('status'), 'needs_clarification')
        self.assertIn("original_llm_mapping", st.session_state)

        # --- Act 2: Simulate user choosing 'Unmapped' ---
        st.session_state.clarification_choices = {
            "trade_value": "Unmapped" # User chooses Unmapped
        }
        
        # Mock streamlit UI calls during confirmation
        with patch('streamlit.success') as mock_st_success, \
             patch('streamlit.write') as mock_st_write, \
             patch('streamlit.info') as mock_st_info, \
             patch('streamlit.rerun') as mock_st_rerun:

            # Call render function to trigger confirmation logic
            final_df, cleaning_done = render_data_validation_interface(
                df=processed_df_initial.copy(), # Pass a copy to check inplace rename later
                validator=self.mock_validator_service,
                summary=summary_initial,
                on_continue=None 
            )

        # --- Assert 2: Check state after confirmation --- 
        # 1. Rerun called, state cleared
        mock_st_rerun.assert_called_once()
        self.assertTrue(st.session_state.get("clarifications_resolved"))
        self.assertNotIn("clarification_choices", st.session_state)
        self.assertNotIn("original_llm_mapping", st.session_state)
        
        # 2. The DataFrame should have other columns mapped, but 'trade_value' remains original
        expected_columns_after_unmap = ['trade_value', 'CustomerZip', 'VIN'] # trade_value keeps original name
        self.assertListEqual(processed_df_initial.columns.tolist(), expected_columns_after_unmap)
        self.assertListEqual(final_df.columns.tolist(), expected_columns_after_unmap)
        self.assertFalse(cleaning_done)

    def test_scenario_5_session_state_corruption(self, MockLLMEngine):
        """E2E Test: Clarification needed, but original mapping is missing from session state."""
        # --- Arrange --- 
        # Reuse setup from scenario 2
        mock_engine_instance = MockLLMEngine.return_value
        clarification_item = {
            "column": "trade_value",
            "question": "Does 'trade_value' map to SalePrice or TradeInValue?",
            "options": ["SalePrice", "TradeInValue"]
        }
        mock_llm_response = {
            "mapping": {
                "TransactionInformation": {
                    "TradeInValue": {"column": "trade_value", "confidence": 0.6}
                }
            },
            "clarifications": [clarification_item],
            "unmapped_columns": []
        }
        mock_engine_instance.map_columns_jeopardy.return_value = mock_llm_response

        # --- Act 1: Initial processing - should set session state ---
        processed_df_initial, summary_initial, _, _ = process_uploaded_file(
            file=self.mock_file_clarify
        )
        
        # --- Assert 1: Check clarification state is set ---
        self.assertEqual(summary_initial.get('status'), 'needs_clarification')
        self.assertIn("original_llm_mapping", st.session_state)

        # --- Act 2: Corrupt session state and simulate confirmation ---
        # Simulate user making a choice
        st.session_state.clarification_choices = {"trade_value": "TradeInValue"}
        # Corrupt session state by removing the mapping
        del st.session_state["original_llm_mapping"] 
        
        # Mock streamlit UI calls during confirmation
        with patch('streamlit.error') as mock_st_error, \
             patch('streamlit.rerun') as mock_st_rerun:

            # Call render function - confirmation logic should hit the error check
            final_df, cleaning_done = render_data_validation_interface(
                df=processed_df_initial.copy(), 
                validator=self.mock_validator_service,
                summary=summary_initial,
                on_continue=None 
            )

        # --- Assert 2: Check that error was handled --- 
        # 1. Specific error message should be shown via st.error
        mock_st_error.assert_called_once_with("Original mapping data not found in session state. Please re-upload the file.")
        
        # 2. Rerun should NOT have been called because confirmation failed
        mock_st_rerun.assert_not_called()
        
        # 3. Clarification state should NOT be marked as resolved
        self.assertFalse(st.session_state.get("clarifications_resolved"))
        
        # 4. DataFrame should remain unchanged from the initial processing output
        self.assertListEqual(processed_df_initial.columns.tolist(), ["trade_value", "post_code", "vin_number"])
        # The returned df should also be the unchanged one
        self.assertListEqual(final_df.columns.tolist(), ["trade_value", "post_code", "vin_number"])
        self.assertFalse(cleaning_done)
        # Session state for choices should still exist as confirmation failed
        self.assertIn("clarification_choices", st.session_state)

# Add placeholder for other scenarios
# class TestDataUploadLLMMapping(unittest.TestCase):
#     ...
#     def test_scenario_2_with_clarifications(self, MockLLMEngine):
#         pass
#     def test_scenario_3_unmapped_columns(self, MockLLMEngine):
#         pass
#     def test_scenario_4_invalid_user_input(self, MockLLMEngine):
#         pass
#     def test_scenario_5_session_state_corruption(self, MockLLMEngine):
#         pass

if __name__ == '__main__':
    unittest.main() 