"""
Enhanced Watchdog AI application with improved UI and chat interface.
"""

import streamlit as st
import pandas as pd
import logging
import os
import sys
import time
import base64
from datetime import datetime
from PIL import Image
import io

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import components
from ui.components.data_upload_enhanced import DataUploadManager
from ui.components.chat_interface import ChatInterface
from ui.components.system_connect import render_system_connect
from utils.errors import handle_error

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state
if 'validated_data' not in st.session_state:
    st.session_state.validated_data = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_insight' not in st.session_state:
    st.session_state.last_insight = None
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

# Set page config
st.set_page_config(
    page_title="Watchdog AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Base64 encoded placeholder logo
LOGO_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAA==
"""

# Function to get base64 encoded image
def get_base64_image(base64_string):
    return f"data:image/png;base64,{base64_string}"

# Enhanced CSS with animations and branding
st.markdown("""
<link href='https://fonts.googleapis.com/css2?family=Orbitron&display=swap' rel='stylesheet'>
<style>
    /* Modern dark theme with tech grid background */
    :root {
        --accent: #00FF88;
        --background: #1E1E1E;
        --text: #D4D4D4;
        --card-bg: #252526;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text);
        background-image: 
            linear-gradient(rgba(0, 255, 136, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 136, 0.05) 1px, transparent 1px);
        background-size: 20px 20px;
        background-attachment: fixed;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 50% 50%, transparent 0%, var(--background) 100%);
        pointer-events: none;
        z-index: 1;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        animation: fadeIn 0.5s ease;
        background-color: rgba(30, 30, 30, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid var(--accent);
        backdrop-filter: blur(5px);
        position: relative;
        z-index: 2;
    }
    
    .logo-container {
        position: relative;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .logo-container::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 2px solid var(--accent);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        70% {
            transform: scale(1.2);
            opacity: 0.7;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Chat message styling */
    .chat-message {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        animation: slideIn 0.3s ease;
    }
    
    .chat-message.user {
        border-left: 3px solid var(--accent);
    }
    
    .chat-message.ai {
        border-left: 3px solid #0066cc;
    }
    
    /* Upload area styling */
    .upload-container {
        padding: 2rem;
        border: 2px dashed var(--accent);
        border-radius: 8px;
        text-align: center;
        background: var(--card-bg);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .upload-container:hover {
        border-color: #00CC66;
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--accent);
        color: var(--background);
        border: none;
        transition: all 0.2s ease;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #00CC66;
        transform: translateY(-1px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0;
            transform: translateY(10px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading spinner */
    .stSpinner {
        border: 4px solid var(--accent);
        border-top: 4px solid transparent;
        animation: spin 1s linear infinite, radar-pulse 2s ease-in-out infinite;
    }
    
    @keyframes spin { 
        100% { transform: rotate(360deg); } 
    }
    
    @keyframes radar-pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(0, 255, 136, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent);
        border-radius: 3px;
    }
    
    /* Section transitions */
    .section-transition {
        animation: fadeIn 0.5s ease;
    }

    /* Logo specific CSS */
    .logo {
        margin-bottom: 16px; /* Spacing below logo */
    }
    .logo img {
        width: 100px !important; /* Default width */
        height: auto !important;
    }

    @media (max-width: 600px) {
      .logo img {
        width: 60px !important; /* Smaller width on mobile */
      }
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    """Render the enhanced header with branding."""
    try:
        logo_path = os.path.join(project_root, "assets", "watchdog_logo.png")
        logo = Image.open(logo_path)
        # Wrap st.image in HTML for CSS targeting
        st.markdown(
            f'<div class="logo"><img src="data:image/png;base64,{image_to_base64(logo)}" alt="Watchdog AI logo"></div>',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Error: watchdog_logo.png not found in assets folder.")
        # Fallback or placeholder if needed
        st.markdown(
            '<div class="logo"><span style="font-size: 40px;">🐾</span></div>',
            unsafe_allow_html=True
        )
        
    # Existing header content
    st.markdown("""
        <div class="header-container">
            <div>
                <h1 style="color: #00FF88; font-family: 'Orbitron', sans-serif; margin: 0; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);">
                    Watchdog AI
                </h1>
                <p style="color: #D4D4D4; margin: 0;">Your AI-powered sales insights assistant</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Helper function to convert PIL Image to base64
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    """Main application function with integrated UI."""
    start_time = time.time()
    
    # Render header
    render_header()
    
    # Create a container for the main content
    main_container = st.container()
    
    with main_container:
        # Render data upload section
        st.markdown("<div class='section-transition'>", unsafe_allow_html=True)
        DataUploadManager().render_upload_section()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Check if data is uploaded and validated
        if st.session_state.validated_data is not None:
            # Show a success message and transition to chat
            st.success("✅ Data uploaded successfully! You can now ask questions about your data.")
            
            # Add a small delay for the transition effect
            time.sleep(0.5)
            
            # Render chat interface
            st.markdown("<div class='section-transition'>", unsafe_allow_html=True)
            ChatInterface().render_chat_interface()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Show a placeholder for the chat interface
            st.markdown("""
                <div style="background-color: rgba(37, 37, 38, 0.7); padding: 2rem; border-radius: 8px; text-align: center; margin-top: 2rem;">
                    <h3 style="color: #00FF88; font-family: 'Orbitron', sans-serif;">Ask About Your Data</h3>
                    <p>Upload your data above to start getting insights.</p>
                    <div style="font-size: 3rem; margin: 1rem 0;">🔍</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Add a sidebar for system connect
    with st.sidebar:
        st.markdown("<h3 style='color: #00FF88; font-family: Orbitron, sans-serif;'>Connect Systems</h3>", unsafe_allow_html=True)
        render_system_connect()
    
    # Log UI load time
    load_time = time.time() - start_time
    logger.debug(f"UI loaded in {load_time:.2f} seconds")

if __name__ == "__main__":
    main()
