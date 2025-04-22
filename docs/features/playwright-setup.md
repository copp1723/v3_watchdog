# Playwright Test Setup

## Getting Started

Playwright is a powerful browser automation library developed by Microsoft that allows for reliable end-to-end testing of web applications. In the Watchdog AI project, we use Playwright with pytest to perform automated UI testing, including visual regression tests. Playwright supports multiple browsers (Chromium, Firefox, and WebKit), provides robust selectors, and has built-in auto-waiting capabilities that make tests more reliable.

Our Playwright integration allows developers to write comprehensive end-to-end tests that verify application functionality across different browsers and device sizes. This reduces manual testing effort and catches UI regressions early in the development cycle. The setup includes useful fixtures and helper methods that simplify common testing patterns.

## Installation

Playwright is included in the test dependencies of our project. You can install it along with other testing dependencies by running:

```bash
# Install the project with test dependencies
pip install -e ".[test]"

# Install the Playwright browsers
python -m playwright install
```

The test dependencies in our `setup.py` include:

```python
TEST_DEPENDENCIES = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
    "pytest-asyncio>=0.23.0",
    "pytest-playwright>=0.4.0",  # Playwright integration for pytest
    "pytest-cov>=4.1.0",
    "markdown>=3.5.0",
]
```

## Configuration

Playwright tests are configured through a `pytest.ini` or `conftest.py` file in the root of the project. Our configuration sets up:

1. **Browser Selection**: By default, tests run in Chromium, but can be configured to run in Firefox or WebKit
2. **Viewport Size**: Default viewport size for desktop testing
3. **Timeouts**: Custom timeouts for actions and navigation
4. **Screenshot Directory**: Location for storing screenshots from visual tests

Here's a sample `conftest.py` configuration for Playwright:

```python
import pytest
from playwright.sync_api import Page, BrowserContext

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Define custom browser context arguments."""
    return {
        **browser_context_args,
        "viewport": {
            "width": 1280,
            "height": 720,
        },
        "ignore_https_errors": True,
        "java_script_enabled": True,
    }

@pytest.fixture
def page(page: Page, browser_context: BrowserContext):
    """Configure the page for tests."""
    page.set_default_timeout(10000)  # 10 seconds
    page.set_default_navigation_timeout(20000)  # 20 seconds
    
    # Set up the page and return it
    yield page
```

## Usage Examples

### Basic Page Navigation and Interaction

```python
def test_login_page(page: Page):
    """Test the login functionality."""
    # Navigate to the login page
    page.goto("http://localhost:8501/login")
    
    # Fill in the login form
    page.fill("[data-testid='username']", "test_user")
    page.fill("[data-testid='password']", "test_password")
    
    # Click the login button
    page.click("[data-testid='login-button']")
    
    # Wait for navigation to complete and verify the redirect
    page.wait_for_selector(".dashboard-title")
    assert page.inner_text(".dashboard-title") == "Dashboard"
```

### Testing Responsive Design

From our visual regression tests, here's how to test responsive design:

```python
def test_mobile_responsive(self, page: Page):
    """Test that the app is responsive on mobile devices."""
    # Set mobile viewport
    page.set_viewport_size({"width": 375, "height": 812})  # iPhone X dimensions
    page.wait_for_selector(".stApp")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)  # Wait for any responsive adjustments
    
    # Take screenshot of mobile view
    page.screenshot(path="screenshots/mobile_view.png")
    
    # Reset viewport to desktop
    page.set_viewport_size({"width": 1280, "height": 800})
```

### Testing UI Components

Here's an example of testing a specific UI component:

```python
def test_navigation_menu(self, page: Page):
    """Test the navigation menu appearance and functionality."""
    # Wait for the navigation menu to be visible
    page.wait_for_selector("[data-testid='stSidebar']", state="visible")
    
    # Click on a navigation item
    page.click("text='Dashboard'")
    
    # Verify the content updated
    page.wait_for_selector(".dashboard-content")
    
    # Test collapsing the sidebar
    page.click("[data-testid='collapseSidebarButton']")
    page.wait_for_timeout(1000)  # Wait for collapse animation
```

## Best Practices

1. **Use data-testid Attributes**: Add `data-testid` attributes to your HTML elements to make selectors more reliable and resistant to CSS changes:
   ```html
   <button data-testid="submit-button">Submit</button>
   ```

2. **Wait for Elements Properly**: Use Playwright's auto-waiting functions instead of arbitrary timeouts:
   ```python
   # Good practice
   page.wait_for_selector("[data-testid='results']")
   
   # Avoid when possible
   page.wait_for_timeout(2000)  # Arbitrary timeout
   ```

3. **Handle Network Requests**: Mock or wait for network requests to complete:
   ```python
   # Wait for network idle
   page.wait_for_load_state("networkidle")
   
   # Or mock network responses
   page.route("**/api/data", lambda route: route.fulfill(json={"test": "data"}))
   ```

4. **Test in Multiple Browsers**: Configure your CI pipeline to run tests in multiple browsers to ensure cross-browser compatibility.

## Troubleshooting

1. **Tests are flaky**: Ensure you're waiting for elements properly and not relying on fixed timeouts

2. **Browser doesn't start**: Check that Playwright browsers are installed correctly:
   ```bash
   python -m playwright install
   ```

3. **Visual differences between environments**: Use a consistent browser version and consider containerization for CI environments

4. **Authentication issues**: Set up authentication state that can be reused across tests:
   ```python
   # Save authentication state
   storage = context.storage_state(path="auth.json")
   
   # Load authentication state in future tests
   browser.new_context(storage_state="auth.json")
   ```

