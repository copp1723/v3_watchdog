"""Header component for the Watchdog AI UI."""

import streamlit as st
from datetime import datetime
import os

def render_header():
    """Render the application header."""
    # Initialize theme in session state if not present
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"  # Set dark theme as default
    
    # Get theme class
    theme_class = "light-theme" if st.session_state.theme == "light" else ""
    
    # Load CSS files
    css_files = ["theme.css", "main.css", "insight_card.css"]
    css_output = ""
    
    for css_file in css_files:
        try:
            css_path = os.path.join(os.path.dirname(__file__), "..", "styles", css_file)
            with open(css_path) as f:
                css_output += f.read()
        except Exception as e:
            st.warning(f"Could not load {css_file}: {e}")
    
    # Apply theme class to the body with early CSS injection in head
    st.markdown(f"""
        <head>
            <style>
                {css_output}
            </style>
        </head>
        <script>
            (function() {{
                // Check localStorage for saved theme preference
                const savedTheme = localStorage.getItem('watchdogTheme');
                const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                
                // Initialize theme based on saved preference, OS preference, or default
                let activeTheme = '{theme_class}';
                if (savedTheme) {{
                    // Use saved theme
                    activeTheme = savedTheme === 'light' ? 'light-theme' : '';
                }} else if (prefersDark) {{
                    // Use OS preference if no saved theme
                    activeTheme = '';
                }}
                
                // Apply theme class to body element
                document.body.className = activeTheme;
                
                // Listen for messages from iframe buttons
                window.addEventListener('message', function(event) {{
                    if (event.data.type === 'themeToggle') {{
                        // Simulate click on the hidden Streamlit button
                        document.querySelector('button[data-testid="baseButton-header-theme-toggle"]').click();
                        
                        // Save theme preference to localStorage
                        const newTheme = document.body.className.includes('light-theme') ? 'light' : 'dark';
                        localStorage.setItem('watchdogTheme', newTheme);
                    }}
                }});
            }})();
        </script>
    """, unsafe_allow_html=True)
    
    # Add header-specific styles
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                width: 100%;
                padding: var(--spacing-lg) 0;
                background-color: var(--bg-card);
                border-bottom: 1px solid var(--border-color);
            }
            
            .logo-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: var(--spacing-md);
            }
            
            .title-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            
            .header-title {
                font-size: var(--font-size-xl);
                font-weight: 600;
                color: var(--fg-primary);
                margin: 0;
            }
            
            .header-subtitle {
                font-size: var(--font-size-sm);
                color: var(--fg-secondary);
                margin: 0;
            }
            
            .controls-container {
                position: absolute;
                top: var(--spacing-md);
                right: var(--spacing-md);
                display: flex;
                align-items: center;
                gap: var(--spacing-md);
            }
            
            .theme-toggle {
                background: var(--button-gradient);
                color: var(--fg-primary);
                border: none;
                border-radius: var(--border-radius);
                padding: var(--spacing-xs) var(--spacing-sm);
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .theme-toggle:hover {
                background: var(--button-gradient-hover);
                transform: translateY(-1px);
            }
            
            @media (max-width: 768px) {
                .header-container {
                    padding: var(--spacing-md) 0;
                }
                
                .controls-container {
                    position: static;
                    margin-top: var(--spacing-md);
                }
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Render header with new structure
    last_update = (
        st.session_state.get('last_sync_timestamp', 'Never')
        if hasattr(st, 'session_state') and 'last_sync_timestamp' in st.session_state
        else datetime.now().strftime("%B %d, %Y")
    )
    
    st.markdown(f"""
        <div class="header-container">
            <div class="logo-container">
                <img src="/Users/joshcopp/Desktop/v3watchdog_ai/assets/watchdog_logo.png" width="40" alt="Watchdog AI Logo">
            </div>
            <div class="title-container">
                <h1 class="header-title">Summit Motors Group</h1>
                <p class="header-subtitle">Powered by Watchdog AI</p>
            </div>
            <div class="controls-container">
                <button class="theme-toggle" onclick="window.parent.postMessage({{type: 'themeToggle'}}, '*');">
                    {('☀️' if st.session_state.theme == "dark" else '🌓')}
                </button>
                <div class="header-subtitle">
                    Data updated: {last_update}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Handle theme toggle (hidden button for Streamlit to handle the state)
    if st.button("Toggle Theme", key="header-theme-toggle", label_visibility="hidden"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()