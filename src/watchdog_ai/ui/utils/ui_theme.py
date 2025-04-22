"""
UI theming system for Watchdog AI application.

This module provides a comprehensive theming system including:
1. Color palettes for light and dark themes
2. Spacing scales for consistent layouts
3. Typography settings
4. Component-specific styling
5. Helper functions for applying theme-consistent styling
"""

import streamlit as st
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

# ==============================================================================
# Color System
# ==============================================================================

class ColorScale:
    """Color scale with variations from lightest to darkest."""
    
    def __init__(self, colors: Dict[int, str]):
        """
        Initialize a color scale with a mapping of weight to hex values.
        
        Args:
            colors: Dictionary mapping weight (100-900) to hex color values
        """
        self.colors = colors
    
    def get(self, weight: int = 500) -> str:
        """Get color hex value by weight."""
        return self.colors.get(weight, self.colors.get(500, "#000000"))


class ColorMode(Enum):
    """Supported color modes."""
    LIGHT = "light"
    DARK = "dark"


class ColorSystem:
    """Complete color system with primary, secondary, and neutral palettes."""
    
    # Primary Blue palette
    PRIMARY = ColorScale({
        50: "#EFF6FF",
        100: "#DBEAFE",
        200: "#BFDBFE",
        300: "#93C5FD",
        400: "#60A5FA",
        500: "#3B82F6",  # Primary brand color
        600: "#2563EB",
        700: "#1D4ED8",
        800: "#1E40AF",
        900: "#1E3A8A",
    })
    
    # Secondary Green palette
    SECONDARY = ColorScale({
        50: "#ECFDF5",
        100: "#D1FAE5",
        200: "#A7F3D0",
        300: "#6EE7B7",
        400: "#34D399",
        500: "#10B981",  # Secondary brand color
        600: "#059669",
        700: "#047857",
        800: "#065F46",
        900: "#064E3B",
    })
    
    # Alert Red palette
    ALERT = ColorScale({
        50: "#FEF2F2",
        100: "#FEE2E2",
        200: "#FECACA",
        300: "#FCA5A5",
        400: "#F87171",
        500: "#EF4444",
        600: "#DC2626",
        700: "#B91C1C",
        800: "#991B1B",
        900: "#7F1D1D",
    })
    
    # Warning Yellow palette
    WARNING = ColorScale({
        50: "#FFFBEB",
        100: "#FEF3C7",
        200: "#FDE68A",
        300: "#FCD34D",
        400: "#FBBF24",
        500: "#F59E0B",
        600: "#D97706",
        700: "#B45309",
        800: "#92400E",
        900: "#78350F",
    })
    
    # Neutral Gray palette
    NEUTRAL = ColorScale({
        50: "#F9FAFB",
        100: "#F3F4F6",
        200: "#E5E7EB",
        300: "#D1D5DB",
        400: "#9CA3AF",
        500: "#6B7280",
        600: "#4B5563",
        700: "#374151",
        800: "#1F2937",
        900: "#111827",
    })
    
    @classmethod
    def get_background(cls, mode: ColorMode) -> str:
        """Get background color for the current theme mode."""
        return cls.NEUTRAL.get(50 if mode == ColorMode.LIGHT else 900)
    
    @classmethod
    def get_text(cls, mode: ColorMode) -> str:
        """Get text color for the current theme mode."""
        return cls.NEUTRAL.get(900 if mode == ColorMode.LIGHT else 50)
    
    @classmethod
    def get_primary(cls, weight: int = 500) -> str:
        """Get primary color with specified weight."""
        return cls.PRIMARY.get(weight)
    
    @classmethod
    def get_secondary(cls, weight: int = 500) -> str:
        """Get secondary color with specified weight."""
        return cls.SECONDARY.get(weight)
    
    @classmethod
    def get_neutral(cls, weight: int = 500) -> str:
        """Get neutral color with specified weight."""
        return cls.NEUTRAL.get(weight)


# ==============================================================================
# Spacing System
# ==============================================================================

class Spacing:
    """Spacing constants for consistent layout."""
    
    UNIT = 0.25  # Base spacing unit in rem (4px at typical browser settings)
    
    @classmethod
    def get(cls, multiplier: int = 1) -> str:
        """Get spacing value as rem string, based on multiplier of the base unit."""
        return f"{cls.UNIT * multiplier}rem"
    
    # Common spacing values
    XXS = get.__func__(1)    # 0.25rem - 4px
    XS = get.__func__(2)     # 0.5rem - 8px
    SM = get.__func__(3)     # 0.75rem - 12px
    MD = get.__func__(4)     # 1rem - 16px
    LG = get.__func__(6)     # 1.5rem - 24px
    XL = get.__func__(8)     # 2rem - 32px
    XXL = get.__func__(12)   # 3rem - 48px
    
    # Layout spacing
    LAYOUT_XS = get.__func__(8)    # 2rem - 32px
    LAYOUT_SM = get.__func__(12)   # 3rem - 48px
    LAYOUT_MD = get.__func__(16)   # 4rem - 64px
    LAYOUT_LG = get.__func__(24)   # 6rem - 96px
    LAYOUT_XL = get.__func__(32)   # 8rem - 128px


# ==============================================================================
# Typography System
# ==============================================================================

class Typography:
    """Typography settings and helpers."""
    
    # Font families
    SANS = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
    MONO = '"SF Mono", SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace'
    
    # Font weights
    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700
    
    # Font sizes (in rem)
    SIZE_XS = "0.75rem"    # 12px
    SIZE_SM = "0.875rem"   # 14px
    SIZE_MD = "1rem"       # 16px
    SIZE_LG = "1.125rem"   # 18px
    SIZE_XL = "1.25rem"    # 20px
    SIZE_2XL = "1.5rem"    # 24px
    SIZE_3XL = "1.875rem"  # 30px
    SIZE_4XL = "2.25rem"   # 36px
    
    # Line heights
    LINE_HEIGHT_TIGHT = 1.2
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.75
    
    @classmethod
    def get_font_css(cls, size: str, weight: int = WEIGHT_NORMAL, 
                    family: str = SANS, line_height: float = LINE_HEIGHT_NORMAL) -> str:
        """Get CSS font declaration."""
        return f"font: {weight} {size}/{line_height} {family};"


# ==============================================================================
# Theme Component
# ==============================================================================

class Theme:
    """Main theme manager class."""
    
    def __init__(self, mode: ColorMode = ColorMode.LIGHT):
        """Initialize theme with specified color mode."""
        self.mode = mode
    
    @property
    def is_dark_mode(self) -> bool:
        """Check if theme is in dark mode."""
        return self.mode == ColorMode.DARK
    
    def get_color_mode(self) -> ColorMode:
        """Get current color mode from session state or default."""
        theme_str = st.session_state.get("theme", "light")
        return ColorMode.DARK if theme_str == "dark" else ColorMode.LIGHT
    
    def toggle_theme(self) -> None:
        """Toggle between light and dark theme."""
        if self.mode == ColorMode.LIGHT:
            st.session_state["theme"] = "dark"
            self.mode = ColorMode.DARK
        else:
            st.session_state["theme"] = "light"
            self.mode = ColorMode.LIGHT
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get theme configuration for Streamlit."""
        return {
            "primaryColor": ColorSystem.get_primary(),
            "backgroundColor": ColorSystem.get_background(self.mode),
            "secondaryBackgroundColor": ColorSystem.NEUTRAL.get(100 if self.mode == ColorMode.LIGHT else 800),
            "textColor": ColorSystem.get_text(self.mode),
            "font": Typography.SANS,
        }
    
    def get_global_css(self) -> str:
        """Generate global CSS based on current theme."""
        background = ColorSystem.get_background(self.mode)
        text_color = ColorSystem.get_text(self.mode)
        border_color = ColorSystem.NEUTRAL.get(200 if self.mode == ColorMode.LIGHT else 700)
        
        return f"""
        <style>
            :root {{
                --color-primary: {ColorSystem.get_primary()};
                --color-primary-light: {ColorSystem.get_primary(400)};
                --color-primary-dark: {ColorSystem.get_primary(600)};
                --color-secondary: {ColorSystem.get_secondary()};
                --color-secondary-light: {ColorSystem.get_secondary(400)};
                --color-secondary-dark: {ColorSystem.get_secondary(600)};
                --color-background: {background};
                --color-text: {text_color};
                --color-border: {border_color};
                --space-unit: {Spacing.UNIT}rem;
                --font-sans: {Typography.SANS};
                --font-mono: {Typography.MONO};
            }}
            
            /* Base styles */
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: {Spacing.MD};
            }}
            
            /* Chat interface specific styles */
            .chat-container {{
                background-color: {'var(--color-background)'};
                border: 1px solid var(--color-border);
                border-radius: 8px;
                padding: {Spacing.MD};
                margin-bottom: {Spacing.LG};
            }}
            
            .chat-message {{
                padding: {Spacing.SM} {Spacing.MD};
                margin-bottom: {Spacing.SM};
                border-radius: 8px;
                max-width: 80%;
            }}
            
            .chat-message-user {{
                background-color: {'var(--color-primary-light)'};
                color: white;
                margin-left: auto;
                border-top-right-radius: 0;
            }}
            
            .chat-message-assistant {{
                background-color: {ColorSystem.NEUTRAL.get(200 if self.mode == ColorMode.LIGHT else 700)};
                color: {'var(--color-text)'};
                margin-right: auto;
                border-top-left-radius: 0;
            }}
            
            /* Insight card styles */
            .insight-card {{
                border: 1px solid var(--color-border);
                border-radius: 8px;
                padding: {Spacing.MD};
                margin-bottom: {Spacing.LG};
                background-color: {ColorSystem.NEUTRAL.get(50 if self.mode == ColorMode.LIGHT else 800)};
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            
            .insight-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            
            .insight-card-title {{
                {Typography.get_font_css(Typography.SIZE_LG, Typography.WEIGHT_SEMIBOLD)}
                color: {'var(--color-text)'};
                margin-bottom: {Spacing.XS};
            }}
            
            .insight-card-metric {{
                {Typography.get_font_css(Typography.SIZE_2XL, Typography.WEIGHT_BOLD)}
                color: {'var(--color-primary)'};
                margin-bottom: {Spacing.SM};
            }}
            
            /* Welcome panel styles */
            .welcome-container {{
                text-align: center;
                padding: {Spacing.LAYOUT_SM};
                margin-bottom: {Spacing.LAYOUT_XS};
                background: linear-gradient(135deg, {ColorSystem.get_primary(400)} 0%, {ColorSystem.get_secondary(400)} 100%);
                color: white;
                border-radius: 12px;
            }}
            
            .welcome-title {{
                {Typography.get_font_css(Typography.SIZE_3XL, Typography.WEIGHT_BOLD)}
                margin-bottom: {Spacing.MD};
            }}
            
            .welcome-subtitle {{
                {Typography.get_font_css(Typography.SIZE_LG, Typography.WEIGHT_NORMAL)}
                opacity: 0.9;
            }}
            
            /* Loading states */
            .loading-animation {{
                opacity: 0.7;
                transition: opacity 0.3s ease-in-out;
            }}
            
            /* Responsive adjustments */
            @media (max-width: 768px) {{
                .chat-message {{
                    max-width: 90%;
                }}
                
                .welcome-container {{
                    padding: {Spacing.LG};
                }}
                
                .welcome-title {{
                    {Typography.get_font_css(Typography.SIZE_2XL, Typography.WEIGHT_BOLD)}
                }}
                
                .welcome-subtitle {{
                    {Typography.get_font_css(Typography.SIZE_MD, Typography.WEIGHT_NORMAL)}
                }}
            }}
        </style>
        """


# ==============================================================================
# Component Style Helpers
# ==============================================================================

def apply_theme_to_page() -> None:
    """Apply the current theme to the Streamlit page by injecting CSS."""
    mode_str = st.session_state.get("theme", "light")
    mode = ColorMode.DARK if mode_str == "dark" else ColorMode.LIGHT
    theme = Theme(mode=mode)
    
    # Apply global CSS
    st.markdown(theme.get_global_css(), unsafe_allow_html=True)
    
    # Configure Streamlit theme
    st.set_page_config(**theme.get_streamlit_config())

