"""
Header component for Watchdog AI application.

This module provides a modern, responsive header with theme integration,
navigation links, and notification indicators.
"""

import streamlit as st
from typing import List, Dict, Optional, Callable
from datetime import datetime

from watchdog_ai.ui.utils.ui_theme import Theme, ColorMode, ColorSystem, Typography, Spacing


def render_logo() -> None:
    """Render the Watchdog AI logo and title."""
    # Get current theme mode
    theme_mode = st.session_state.get("theme", "light")
    
    # Determine logo color based on theme (white for dark mode, dark for light mode)
    logo_color = "white" if theme_mode == "dark" else "#1E3A8A"
    
    # Create logo and title with styling
    st.markdown(f"""
        <div class="logo-container">
            <div class="logo-icon">🔍</div>
            <div class="logo-text" style="color: {logo_color};">
                <span class="logo-name">Watchdog</span>
                <span class="logo-accent" style="color: {ColorSystem.get_primary()};">AI</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Render the theme toggle switch."""
    # Get current theme
    current_theme = st.session_state.get("theme", "light")
    is_dark = current_theme == "dark"
    
    # Create theme toggle with custom styling
    toggle_id = "theme-toggle"
    
    # Define toggle appearance based on current theme
    toggle_bg = ColorSystem.NEUTRAL.get(700 if is_dark else 200)
    toggle_dot_color = ColorSystem.get_primary(500)
    toggle_dot_position = "18px" if is_dark else "2px"
    
    # Create the toggle HTML
    st.markdown(f"""
        <div class="theme-toggle-container">
            <label for="{toggle_id}" class="theme-toggle-label">
                <span class="toggle-icon">{'🌙' if is_dark else '☀️'}</span>
            </label>
            <div class="theme-toggle-wrapper">
                <div class="theme-toggle" 
                     style="background-color: {toggle_bg};"
                     onclick="toggleTheme()">
                    <div class="toggle-dot" 
                         style="transform: translateX({toggle_dot_position}); background-color: {toggle_dot_color};"></div>
                </div>
            </div>
        </div>
        
        <script>
            function toggleTheme() {{
                const themeKey = "theme";
                const currentTheme = localStorage.getItem(themeKey) || "light";
                const newTheme = currentTheme === "light" ? "dark" : "light";
                
                // Update localStorage
                localStorage.setItem(themeKey, newTheme);
                
                // Update session state via a hidden form
                const formElement = document.createElement('form');
                formElement.method = 'POST';
                formElement.style.display = 'none';
                
                const inputElement = document.createElement('input');
                inputElement.name = "theme-toggle-submit";
                inputElement.value = newTheme;
                formElement.appendChild(inputElement);
                
                document.body.appendChild(formElement);
                formElement.submit();
                document.body.removeChild(formElement);
            }}
        </script>
    """, unsafe_allow_html=True)
    
    # Handle form submissions for theme toggle
    if st.session_state.get("theme-toggle-submit"):
        new_theme = st.session_state["theme-toggle-submit"]
        st.session_state["theme"] = new_theme
        # Clear the form submission value
        st.session_state["theme-toggle-submit"] = None
        # Force rerun to apply theme changes
        st.rerun()


def render_notification_indicator(count: int = 0) -> None:
    """
    Render notification indicator with badge showing count.
    
    Args:
        count: Number of notifications to display
    """
    # Only show count if there are notifications
    badge_html = f"""
        <div class="notification-badge">{count}</div>
    """ if count > 0 else ""
    
    # Create notification bell with badge
    st.markdown(f"""
        <div class="notification-container">
            <div class="notification-icon">🔔</div>
            {badge_html}
        </div>
    """, unsafe_allow_html=True)


def render_navigation_links(links: List[Dict[str, str]] = None) -> None:
    """
    Render horizontal navigation links.
    
    Args:
        links: List of dictionaries with 'label' and 'url' keys
    """
    if links is None:
        links = [
            {"label": "Insights", "url": "#insights"},
            {"label": "Dashboard", "url": "#dashboard"},
            {"label": "Reports", "url": "#reports"},
            {"label": "Help", "url": "#help"}
        ]
    
    # Create navigation bar
    st.markdown("""
        <div class="nav-links">
    """, unsafe_allow_html=True)
    
    # Add each navigation link
    for link in links:
        st.markdown(f"""
            <a href="{link['url']}" class="nav-link">{link['label']}</a>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    """, unsafe_allow_html=True)


def render_user_menu(user_name: str = None, user_email: str = None) -> None:
    """
    Render user profile menu.
    
    Args:
        user_name: Name of the current user
        user_email: Email of the current user
    """
    # Default user info if not provided
    user_name = user_name or "Demo User"
    user_email = user_email or "user@example.com"
    
    # Create user profile avatar (first letter of name)
    avatar_text = user_name[0].upper() if user_name else "U"
    
    # Create user menu
    st.markdown(f"""
        <div class="user-menu">
            <div class="user-avatar">{avatar_text}</div>
            <div class="user-info">
                <div class="user-name">{user_name}</div>
                <div class="user-email">{user_email}</div>
            </div>
            <div class="user-dropdown-icon">▼</div>
        </div>
    """, unsafe_allow_html=True)


def render_header_css(theme: Theme) -> None:
    """
    Render CSS styles for header component.
    
    Args:
        theme: Current theme instance
    """
    # Get theme colors
    bg_color = ColorSystem.get_background(theme.get_color_mode())
    text_color = ColorSystem.get_text(theme.get_color_mode())
    primary_color = ColorSystem.get_primary()
    border_color = ColorSystem.NEUTRAL.get(200 if theme.get_color_mode() == ColorMode.LIGHT else 700)
    
    # Create header-specific CSS
    st.markdown(f"""
    <style>
        /* Header Container */
        .header-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: {Spacing.MD} {Spacing.LG};
            background-color: {bg_color};
            border-bottom: 1px solid {border_color};
            margin-bottom: {Spacing.LG};
        }}
        
        /* Logo Styling */
        .logo-container {{
            display: flex;
            align-items: center;
            gap: {Spacing.XS};
        }}
        
        .logo-icon {{
            font-size: 24px;
        }}
        
        .logo-text {{
            font-weight: {Typography.WEIGHT_BOLD};
            font-size: {Typography.SIZE_XL};
        }}
        
        .logo-name {{
            margin-right: 2px;
        }}
        
        /* Navigation Links */
        .nav-links {{
            display: flex;
            gap: {Spacing.LG};
        }}
        
        .nav-link {{
            color: {text_color};
            text-decoration: none;
            font-weight: {Typography.WEIGHT_MEDIUM};
            padding-bottom: 3px;
            border-bottom: 2px solid transparent;
            transition: border-color 0.2s ease;
        }}
        
        .nav-link:hover {{
            border-bottom: 2px solid {primary_color};
        }}
        
        /* Theme Toggle */
        .theme-toggle-container {{
            display: flex;
            align-items: center;
            gap: {Spacing.XS};
        }}
        
        .theme-toggle-wrapper {{
            position: relative;
        }}
        
        .theme-toggle {{
            width: 40px;
            height: 22px;
            border-radius: 11px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }}
        
        .toggle-dot {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            position: relative;
            top: 2px;
            transition: transform 0.3s ease, background-color 0.3s ease;
        }}
        
        .toggle-icon {{
            font-size: 16px;
        }}
        
        /* Notification Styling */
        .notification-container {{
            position: relative;
            cursor: pointer;
        }}
        
        .notification-icon {{
            font-size: 20px;
        }}
        
        .notification-badge {{
            position: absolute;
            top: -8px;
            right: -8px;
            background-color: {ColorSystem.ALERT.get(500)};
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: {Typography.WEIGHT_BOLD};
        }}
        
        /* User Menu */
        .user-menu {{
            display: flex;
            align-items: center;
            gap: {Spacing.XS};
            cursor: pointer;
            padding: {Spacing.XS} {Spacing.SM};
            border-radius: 8px;
            transition: background-color 0.2s ease;
        }}
        
        .user-menu:hover {{
            background-color: rgba(0, 0, 0, 0.05);
        }}
        
        .user-avatar {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: {primary_color};
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: {Typography.WEIGHT_BOLD};
        }}
        
        .user-info {{
            display: none;
        }}
        
        .user-name {{
            font-weight: {Typography.WEIGHT_MEDIUM};
            font-size: 14px;
        }}
        
        .user-email {{
            font-size: 12px;
            color: {ColorSystem.NEUTRAL.get(500)};
        }}
        
        .user-dropdown-icon {{
            font-size: 10px;
            color: {ColorSystem.NEUTRAL.get(500)};
        }}
        
        /* Responsive adjustments */
        @media (min-width: 768px) {{
            .user-info {{
                display: block;
            }}
        }}
        
        /* Fix Streamlit components spacing */
        [data-testid="stVerticalBlock"] > div:has(.header-container) {{
            padding-bottom: 0 !important;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_header(
    show_navigation: bool = True, 
    show_user_menu: bool = True,
    notification_count: int = 0
) -> None:
    """
    Render the complete header component.
    
    Args:
        show_navigation: Whether to display navigation links
        show_user_menu: Whether to display user menu
        notification_count: Number of unread notifications to display
    """
    # Create theme instance
    theme_mode = st.session_state.get("theme", "light")
    theme = Theme(mode=ColorMode.DARK if theme_mode == "dark" else ColorMode.LIGHT)
    
    # Render header CSS
    render_header_css(theme)
    
    # Use columns for header layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Main header container
    st.markdown("""
    <div class="header-container">
        <div class="header-left">
    """, unsafe_allow_html=True)
    
    # Left section - Logo
    render_logo()
    
    st.markdown("""
        </div>
        <div class="header-center">
    """, unsafe_allow_html=True)
    
    # Center section - Navigation
    if show_navigation:
        render_navigation_links()
    
    st.markdown("""
        </div>
        <div class="header-right">
    """, unsafe_allow_html=True)
    
    # Right section - Theme toggle, notifications, user menu
    render_theme_toggle()
    render_notification_indicator(notification_count)
    
    if show_user_menu:
        render_user_menu()
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

