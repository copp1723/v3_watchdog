"""
Nova Act exceptions.
"""

class NovaActError(Exception):
    """Base exception for Nova Act errors."""
    pass

class TemplateError(NovaActError):
    """Exception raised for template-related errors."""
    pass

class LoginError(NovaActError):
    """Exception raised for login failures."""
    pass

class ReportError(NovaActError):
    """Exception raised for report fetching failures."""
    pass

class TwoFactorError(NovaActError):
    """Exception raised for 2FA-related errors."""
    pass 