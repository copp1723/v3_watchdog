import streamlit as st

# Add code to explicitly load .env file
import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print(f"[DEBUG] Loaded .env file. USE_MOCK={os.getenv('USE_MOCK')}, API KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")

# Handle OpenAI version compatibility
try:
    import openai
    openai_version = getattr(openai, "__version__", "unknown")
    print(f"[DEBUG] Current OpenAI version: {openai_version}")
    
    # If USE_MOCK is false, ensure we have the compatible version
    if os.getenv("USE_MOCK", "true").lower() == "false":
        if not openai_version.startswith("0."):
            print("[DEBUG] Detected incompatible OpenAI version. Installing openai==0.28 for compatibility...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai==0.28", "--quiet"])
            print("[DEBUG] Reloading openai module...")
            import importlib
            importlib.reload(openai)
            print(f"[DEBUG] OpenAI version after reload: {openai.__version__}")
except Exception as e:
    print(f"[ERROR] Failed to check/update OpenAI version: {e}")

# This must be the very first Streamlit command
st.set_page_config(
    page_title="Watchdog AI",
    page_icon=None,
    layout="wide"
)

# Inject custom CSS for typography and styling
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Helvetica Neue', 'Segoe UI', sans-serif;
            font-weight: 400;
            color: #D0D0D0;
        }
        
        body {
            background: linear-gradient(to top, #1e0f0f, #3b1e1e);
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: 500 !important;
        }

        .stButton>button {
            font-weight: 400 !important;
        }

        .stCheckbox label span {
            font-weight: 400 !important;
        }

        /* Override warning colors to a soft gray tone */
        .stAlert.stWarning {
            background-color: #444 !important;
            color: #EEE !important;
        }
        
        .stAlert.stWarning > div > div > div {
            color: #EEE !important;
        }

        /* Override checkbox color to a soft blue tone */
        .stCheckbox > label > div[data-testid="stCheckbox"] > div {
            background-color: #2d4059 !important;
        }

        /* Message bubble fade-in animation */
        .stMarkdown > div {
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Style for user bubble */
        .user-bubble {
            background-color: #e45258;
            color: white;
            padding: 12px;
            border-radius: 12px;
            text-align: right;
            max-width: 80%;
            margin-left: auto;
            font-weight: 600;
        }

        /* Style for AI bubble */
        .ai-bubble {
            background-color: #f4f4f4;
            color: #111;
            padding: 12px;
            border-radius: 12px;
            text-align: left;
            max-width: 80%;
            margin-right: auto;
            font-weight: 600;
        }

        /* Optional: Remove Streamlit branding footer */
        footer {visibility: hidden;}

        /* Optional: Smoother chart section */
        .element-container:has(> .stAltairChart) {
            padding-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Add the project root directory (which contains 'src') to the Python path
# This allows imports like 'from src.validators import ...' to work correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
Main Streamlit app for Watchdog AI.

This application integrates the validation profile system with the insight generation pipeline.
It provides a clean UI for uploading files, validating data, and generating insights.
"""

import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import necessary components
from src.validators import ValidationProfile, get_available_profiles # Added get_available_profiles
from src.ui.components import render_data_upload
from src.insight_conversation import ConversationManager # Keep for insights page
from src.insight_card_improved import render_insight_card # Using the improved version
from src.validators.validator_service import process_uploaded_file
from src.ui.components.flag_panel import render_flag_summary

# Removed: ValidatorService, InsightValidator, render_flag_panel (using render_flag_summary), render_conversation_history

# --- Constants and Config --- 
# TODO: Move to config file or env vars
PROFILES_DIR = os.getenv("WATCHDOG_PROFILES_DIR", "profiles")
DEFAULT_PROFILE_ID = "Default"

# --- Helper Functions --- 

def load_profile(profile_id: str, profiles_dir: str = PROFILES_DIR) -> Optional[ValidationProfile]:
    """Loads a specific ValidationProfile by ID."""
    try:
        # Assuming ValidationProfile has a class method or function to load by ID/path
        # Replace this with the actual loading mechanism (e.g., from_json, from_yaml)
        # Example placeholder:
        profiles = get_available_profiles(profiles_dir)
        profile = next((p for p in profiles if p.id == profile_id), None)
        if profile:
            return profile
        else:
            st.error(f"Could not find profile with ID: {profile_id}")
            return None
    except Exception as e:
        st.error(f"Error loading profile '{profile_id}': {e}")
        return None

def render_insight_history(history: List[Dict[str, Any]]) -> None:
    """
    Render the insight history as a chat stream with styled bubbles.
    
    Args:
        history: List of conversation history entries
    """
    # New Insight button removed from here, might be placed elsewhere like a header or sticky footer later
    
    # Render the conversation history
    if not history:
        st.info("No insights yet. Start by entering a prompt!")
        return
    
    # Render insights in reverse chronological order for chat flow
    for i, entry in enumerate(reversed(history)):
        with st.container():
            # User Prompt Bubble (Right Aligned)
            user_prompt = entry['prompt']
            st.markdown(f"""
            <div class="user-bubble">
              {user_prompt}
            </div>
            """, unsafe_allow_html=True)
            
            # Optional timestamp in small text
            if 'timestamp' in entry:
                timestamp = entry['timestamp']
                # Convert to more readable format if it's ISO format
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%I:%M %p")  # Format as "1:23 PM"
                except:
                    pass  # Keep original format if conversion fails
                    
                st.caption(f"{timestamp}")
            
            # AI Response Bubble (Left Aligned)
            if 'response' in entry and 'summary' in entry['response']:
                ai_response = entry['response']['summary']
                st.markdown(f"""
                <div class="ai-bubble">
                  {ai_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Render the full insight card below (chart, recommendation, etc.)
                render_insight_card(entry['response'], show_buttons=False, card_index=i)
            
            # Add regeneration button for each insight
            cols = st.columns(4)
            with cols[0]:
                if st.button("Regenerate", key=f"regenerate_{i}"):
                    st.session_state.regenerate_insight = True
                    st.session_state.regenerate_index = len(history) - 1 - i  # Calculate correct index
                    st.rerun()
            
            # Add spacing between conversation pairs
            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

def handle_clean_data(cleaned_df: pd.DataFrame, applied_actions: Dict[str, Any]):
    """Callback function to handle data after cleaning actions are applied in the flag panel."""
    st.session_state.validated_data = cleaned_df
    # Store a summary of what was done (optional)
    if 'cleaning_summary' not in st.session_state:
        st.session_state.cleaning_summary = []
    
    summary_entry = {
        "timestamp": datetime.now().isoformat(),
        "actions": applied_actions
    }
    st.session_state.cleaning_summary.append(summary_entry)
    
    # Log to console or display message
    st.toast("Cleaning applied. Data updated in session state.")
    print(f"Cleaning applied at {summary_entry['timestamp']}:")
    for action, details in applied_actions.items():
        print(f"- {action}: {details}")

    # Automatically proceed to insights after cleaning
    st.session_state.page = "insights"
    st.rerun() # Use st.rerun instead of experimental_rerun

def main():
    """Main Streamlit app for Watchdog AI."""
    # Initialize session state variables
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = ""
    if "clear_prompt_input_flag" not in st.session_state:
        st.session_state.clear_prompt_input_flag = False
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "regenerate_insight" not in st.session_state:
        st.session_state.regenerate_insight = False
    if "regenerate_index" not in st.session_state:
        st.session_state.regenerate_index = None
    if 'page' not in st.session_state:
        st.session_state.page = "upload"  # Default page
    if 'validated_data' not in st.session_state:
        st.session_state.validated_data = None
    if 'validation_summary' not in st.session_state:
        st.session_state.validation_summary = None
    if 'validation_report' not in st.session_state:
        st.session_state.validation_report = None
    if 'selected_profile_id' not in st.session_state:
        st.session_state.selected_profile_id = DEFAULT_PROFILE_ID
    
    # Initialize session state for cleaning summary
    if 'cleaning_summary' not in st.session_state:
        st.session_state.cleaning_summary = []
    
    # --- Load Available Profiles --- 
    available_profiles = []
    try:
        available_profiles = get_available_profiles(PROFILES_DIR)
    except Exception as e:
        st.sidebar.error(f"Error loading validation profiles from {PROFILES_DIR}: {e}")

    # Sidebar
    with st.sidebar:
        st.header("Options")
        
        # Auto-clean checkbox (kept visible, label updated)
        # Note: Styling checkboxes directly with markdown is limited
        st.markdown('<style>.stCheckbox label span { font-size: 0.95em; }</style>', unsafe_allow_html=True) # Attempt to slightly reduce label size
        auto_clean = st.checkbox("Auto-clean data on upload", value=True, key="auto_clean_checkbox")
        
        # Advanced Settings Expander for Profile Selection
        with st.expander("Advanced Settings"):
            # Profile Selection Dropdown (moved inside expander)
            profile_options = {"None": "No Profile"} # Start with 'None' option
            profile_options.update({p.id: p.name for p in available_profiles}) # Add loaded profiles
            
            # Get the current profile ID from session state, defaulting to DEFAULT_PROFILE_ID
            current_profile_id = st.session_state.get('selected_profile_id', DEFAULT_PROFILE_ID)
            
            # Safely determine if the current_profile_id exists in the options
            if current_profile_id not in profile_options:
                # If not found, default to "None"
                st.warning(f"Profile '{current_profile_id}' not found. Using 'None' instead.")
                current_profile_id = "None"
                # Update session state
                st.session_state.selected_profile_id = current_profile_id
            
            selected_id = st.selectbox(
                "Validation Profile (Optional)", # Updated label
                options=list(profile_options.keys()),
                index=list(profile_options.keys()).index(current_profile_id),
                format_func=lambda x: profile_options[x], # Display names
                key='selected_profile_id' # Use session state key
            )
            st.caption("Choose a profile to apply custom data validation rules.") # Added caption
            
        # Removed Load Sample Data section

        # Only show insights navigation when validation is complete
        if st.session_state.validated_data is not None and st.session_state.page != "insights":
            if st.button("Go to Insights"):
                st.session_state.page = "insights"
                st.rerun() # Rerun to switch page
        
        # Simplified LLM indicator
        if 'conversation_manager' in st.session_state:
            cm = st.session_state.conversation_manager
            if not cm.use_mock:
                provider_name = cm.llm_provider.capitalize()
                st.success(f"Using {provider_name} API") # Removed emoji
    
    # Main content
    # --- Page Routing & Content --- 
    current_page = st.session_state.get('page', 'upload')

    # Removed sample data loading logic

    # Determine if data is ready for validation/insights
    data_loaded = st.session_state.get('validated_data') is not None

    # Redirect to upload if no data is loaded, unless already there
    if not data_loaded and current_page != 'upload':
        st.session_state.page = "upload"
        current_page = "upload" # Update for current script run
        # st.rerun() # Avoid immediate rerun here, let the structure handle it

    if current_page == "upload":
        st.header("1. Upload Data")
        st.markdown("Drag and drop your file to begin.")
        uploaded_file = render_data_upload(simplified=True)  # We'll add simplified mode parameter
        
        # Add the warning here if no profile is selected but without emoji
        if st.session_state.selected_profile_id == "None":
            st.warning("No validation profile selected in the sidebar. Data will be loaded but not analyzed for quality issues.")
        
        if uploaded_file is not None:
            # --- Load the selected profile object --- 
            selected_profile_object = None
            if selected_id != "None": # Check if a profile was selected
                selected_profile_object = load_profile(selected_id, PROFILES_DIR)
                if selected_profile_object is None:
                    # Error already shown by load_profile, prevent processing
                    st.stop()
            
            # --- Process File --- 
            with st.spinner("Processing file..."):
                # Call process_uploaded_file with the loaded object (or None)
                cleaned_df, summary, report = process_uploaded_file(
                    uploaded_file, 
                    selected_profile=selected_profile_object, # Pass object
                    auto_clean=auto_clean 
                )

                if cleaned_df is not None: # Check if processing was successful
                    st.session_state.validated_data = cleaned_df
                    st.session_state.validation_summary = summary 
                    st.session_state.validation_report = report

                    st.success(f"File processed successfully (Profile: {summary.get('profile_used', 'N/A')}).")
                    st.session_state.page = "validation" 
                    st.rerun() # Rerun to switch page
                else:
                    st.error(f"Error processing file: {summary.get('message', 'Unknown error')}")
                    # Reset state on error
                    st.session_state.validated_data = None
                    st.session_state.validation_summary = None
                    st.session_state.validation_report = None
                    # Keep selected profile ID, don't reset page immediately
    
    elif current_page == "validation":
        st.header("2. Data Validation")
        # Ensure data exists before rendering panel
        selected_profile_object = None
        if selected_id != "None":
            selected_profile_object = load_profile(selected_id, PROFILES_DIR)
        if st.session_state.validated_data is not None:
            # Warn if expected columns are missing
            if selected_profile_object:
                expected = set()
                for rule in selected_profile_object.get_enabled_rules():
                    expected.update(rule.column_mapping.values())
                actual = set(st.session_state.validated_data.columns)
                missing = expected - actual
                if missing:
                    st.warning(f"Missing expected columns for validation: {missing}")
            render_flag_summary(
                st.session_state.validated_data, 
                on_clean_click=handle_clean_data # Pass the callback here
            )
        else:
            st.warning("No data validated yet. Please upload a file first.")
            if st.button("Go to Upload"):
                st.session_state.page = "upload"
                st.rerun()
    
    elif current_page == "insights":
        st.header("Insights Generation")
        
        # --- Render Chat History --- 
        # This container will hold the chat stream
        chat_container = st.container()
        with chat_container:
             # Render insight history (now styled as chat)
             render_insight_history(st.session_state.conversation_history)
        
        # --- Spacer to push input to bottom (adjust height as needed) ---
        # This is a simple way; a more robust solution might involve CSS
        st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)
        
        # --- Chat Input Area (Moved to bottom) ---
        # Check if we need to clear the prompt input (logic remains)
        if st.session_state.clear_prompt_input_flag:
            st.session_state.prompt_input = ""
            st.session_state.clear_prompt_input_flag = False
        
        # Chat input area with styling
        st.markdown("""
        <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #2d2d2d; padding: 20px; border-top: 1px solid #444; z-index: 1000;">
            <div style="max-width: 1200px; margin: 0 auto;">
                <div style="display: flex; align-items: center;">
                    <div style="flex-grow: 1; margin-right: 10px;">
                        <!-- Input field will be placed here by Streamlit -->
                    </div>
                    <div style="width: 60px;">
                        <!-- Button will be placed here by Streamlit -->
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Input row using columns
        input_col1, input_col2 = st.columns([5, 1])
        with input_col1:
            # Render the prompt input field with custom styling
            prompt = st.text_input(
                "Type your question...",
                key="prompt_input",
                placeholder="Ask a question about your data...",
                label_visibility="collapsed"
            )
        with input_col2:
             send_button = st.button("Send", key="send_button", use_container_width=True)

        # Get validation context (logic remains the same)
        validation_context = {}
        if 'validation_summary' in st.session_state and st.session_state.validation_summary:
            validation_context = {
                'validation_summary': st.session_state.validation_summary,
                'validation_report': st.session_state.validation_report if 'validation_report' in st.session_state else None
            }
        
        # --- Insight Generation Logic ---
        # Triggered by the new send button
        if send_button and prompt:
            st.session_state.current_prompt = prompt # Store the prompt that was submitted
            with st.spinner("Generating insight..."):
                cm = st.session_state.conversation_manager
                # Debug prints remain
                print(f"[DEBUG] Final check before generate_insight - cm.use_mock: {getattr(cm, 'use_mock', 'N/A')}")
                print(f"[DEBUG] Final check before generate_insight - cm.llm_provider: {getattr(cm, 'llm_provider', 'N/A')}")
                
                response = cm.generate_insight(
                    st.session_state.current_prompt,
                    validation_context=validation_context # Pass context
                )
            # Clear the prompt *after* successful generation
            st.session_state.current_prompt = None 
            # Flag to clear the input field itself on rerun
            st.session_state.clear_prompt_input_flag = True 
            st.rerun() # Rerun to update history & clear input

        # --- Insight Regeneration Logic (Triggered by flags, but buttons removed from history view) ---
        # This logic might need a new trigger if regeneration is desired from a context bar later
        elif st.session_state.get('regenerate_insight', False) and st.session_state.get('regenerate_index') is not None:
             # ... (existing regeneration logic remains, but won't be triggered by buttons in chat history) ...
             # Reset flags after processing
             st.session_state.regenerate_insight = False
             st.session_state.regenerate_index = None
             st.rerun()
             
        # Removed rendering of history from here as it's now above the input
        # Removed cleaning summary expander from here, might be placed elsewhere


if __name__ == "__main__":
    main()