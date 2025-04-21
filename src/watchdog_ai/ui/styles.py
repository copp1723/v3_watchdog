"""
Shared styles for the Watchdog AI UI.

This file provides utility functions for working with the CSS theme variables
defined in src/watchdog_ai/ui/styles/theme.css and applied via the light-theme class.

IMPORTANT: Most styles should be in CSS files that use CSS variables. This file should
only contain utility functions for Streamlit components that need programmatic styling.
"""

import os
import logging

logger = logging.getLogger(__name__)

def load_css_file(file_path):
    """Load a CSS file content."""
    try:
        with open(file_path) as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not load CSS file {file_path}: {e}")
        return ""

def load_theme_css():
    """Load all CSS files from the styles directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    styles_dir = os.path.join(current_dir, "styles")
    
    css_files = ["theme.css", "main.css", "insight_card.css"]
    css_output = ""
    
    for css_file in css_files:
        css_path = os.path.join(styles_dir, css_file)
        css_output += load_css_file(css_path)
    
    return css_output

def apply_theme(theme="dark"):
    """Apply theme CSS to the Streamlit app."""
    css = load_theme_css()
    theme_class = "light-theme" if theme == "light" else ""
    
    # Return script to apply theme class to body
    return f"""
        <style>
            {css}
        </style>
        <script>
            (function() {{
                document.body.className = '{theme_class}';
            }})();
        </script>
    """

def get_streamlit_styles():
    """Get styles for Streamlit-specific components that can't be targeted with CSS."""
    return """
        <style>
            /* Streamlit component overrides */
            .stApp {
                background-color: var(--bg-page) !important;
                color: var(--fg-primary) !important;
                font-family: var(--font-sans) !important;
            }
            
            /* Upload button styling */
            .stFileUploader > div > button {
                background: var(--btn-gradient) !important;
                color: var(--fg-primary) !important;
                border-radius: var(--border-radius) !important;
                border: none !important;
                font-family: var(--font-sans) !important;
                transition: all 0.2s ease !important;
            }

            .stFileUploader > div > button:hover {
                background: var(--btn-gradient-hover) !important;
                transform: translateY(-1px);
            }

            /* Input field styling */
            .stTextInput > div > div > input {
                background-color: var(--bg-input) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--fg-primary) !important;
                border-radius: var(--border-radius) !important;
                font-family: var(--font-sans) !important;
            }

            .stTextInput > div > div > input::placeholder {
                color: var(--fg-placeholder) !important;
            }

            /* Button styling */
            .stButton > button {
                background: var(--btn-gradient) !important;
                color: var(--fg-primary) !important;
                border-radius: 20px !important;
                border: none !important;
                font-family: var(--font-sans) !important;
                padding: var(--spacing-sm) var(--spacing-lg) !important;
                transition: all 0.2s ease !important;
            }

            .stButton > button:hover {
                background: var(--btn-gradient-hover) !important;
                transform: translateY(-1px);
            }

            /* Selectbox styling */
            .stSelectbox > div > div > select {
                background-color: var(--bg-input) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--fg-primary) !important;
                border-radius: var(--border-radius) !important;
                font-family: var(--font-sans) !important;
            }

            /* Progress bar styling */
            .stProgress > div > div {
                background-color: var(--accent-primary) !important;
            }

            /* Expander styling */
            .streamlit-expanderHeader {
                background-color: var(--bg-card) !important;
                color: var(--fg-primary) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--border-radius) !important;
            }
        </style>
    """

def inject_custom_css(theme="light"):
    """
    Legacy function to inject custom CSS into the Streamlit app.
    
    DEPRECATED: Use apply_theme() instead.
    
    Args:
        theme: The theme to apply ('light' or 'dark')
        
    Returns:
        CSS and JS to apply the theme
    """
    return apply_theme(theme) + get_streamlit_styles()