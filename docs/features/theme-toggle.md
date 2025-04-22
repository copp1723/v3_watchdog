# Theme Toggle System

## Getting Started

The Watchdog AI application includes a theme toggle system that allows users to switch between light and dark modes. This feature enhances user experience by providing visual comfort based on preference or ambient lighting conditions. The theme toggle is implemented as a button in the user interface that toggles between light mode (default) and dark mode.

Theme toggling is particularly useful for applications with high usage rates or those used in varying lighting conditions. The Watchdog AI implementation is built on top of Streamlit's theming capabilities and uses CSS variables to ensure consistent styling across the application.

## Implementation

The theme toggle system relies on Streamlit's built-in theming infrastructure and is accessed via a button with the test ID `darkModeButton`. Here's how the theme toggle is tested in our visual regression test suite:

```python
def test_dark_mode(self):
    """Test the dark mode appearance."""
    # Wait for the theme toggle button to be visible
    self.page.wait_for_selector("[data-testid='darkModeButton']", state="visible")
    
    # Take screenshot in light mode (default)
    self.assert_screenshot("light_mode")
    
    # Click the dark mode toggle
    self.page.click("[data-testid='darkModeButton']")
    self.page.wait_for_timeout(1000)  # Wait for theme transition
    
    # Take screenshot in dark mode
    self.assert_screenshot("dark_mode")
    
    # Reset back to light mode
    self.page.click("[data-testid='darkModeButton']")
    self.page.wait_for_timeout(1000)  # Wait for theme transition
```

## Usage

To use the theme toggle in your application:

1. **Accessing the Theme Toggle**: The theme toggle button is located in the top navigation bar of the application and is represented by a sun/moon icon.

2. **Toggling Themes**: Click the theme toggle button to switch between light and dark modes. The application will remember your preference for future sessions.

3. **Programmatic Theme Switching**: For development or testing purposes, you can programmatically toggle the theme:

   ```python
   # Using Playwright for testing or automation
   page.click("[data-testid='darkModeButton']")
   ```

4. **Best Practices**:
   - Ensure all custom components respect the current theme by using CSS variables
   - Test both themes regularly to ensure good contrast and readability
   - Remember that screenshots and visuals may differ between themes

## Customization

The theme colors can be customized by modifying the theme configuration in the `.streamlit/config.toml` file. This allows for brand-specific color schemes while maintaining the toggle functionality.

