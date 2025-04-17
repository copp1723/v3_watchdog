"""
Shared constants for Nova Act integration.
"""

from enum import Enum
from typing import Dict, Any

class ErrorType(Enum):
    """Error types for Nova Act operations."""
    LOGIN_FAILED = "login_failed"
    NAVIGATION_FAILED = "navigation_failed"
    DOWNLOAD_FAILED = "download_failed"
    INVALID_CREDENTIALS = "invalid_credentials"
    CAPTCHA_DETECTED = "captcha_detected"
    TWO_FACTOR_REQUIRED = "2fa_required"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN = "unknown"

class TwoFactorMethod(Enum):
    """Supported 2FA methods."""
    NONE = "none"
    SMS = "sms"
    EMAIL = "email"
    AUTHENTICATOR = "authenticator"

class FileFormat(Enum):
    """Supported file formats for report downloads."""
    CSV = "csv"
    EXCEL = "xlsx"
    PDF = "pdf"

# Default timeouts (in seconds)
TIMEOUTS = {
    "login": 30,
    "navigation": 15,
    "download": 300,  # 5 minutes for large files
    "2fa": 300,  # 5 minutes for user interaction
}

# Rate limiting settings
RATE_LIMITS = {
    "max_concurrent_sessions": 5,
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "cooldown_period": 60,  # seconds
}

# File validation settings
FILE_VALIDATION = {
    "max_size_mb": 100,
    "allowed_formats": [".csv", ".xlsx", ".xls"],
    "required_columns": {
        "sales": ["date", "amount", "customer"],
        "inventory": ["vin", "make", "model", "year"],
        "leads": ["date", "source", "status"]
    }
}

# Vendor-specific configurations
VENDOR_CONFIGS = {
    "vinsolutions": {
        "base_url": "https://{store_id}.vinsolutions.com",
        "api_version": "v2",
        "required_fields": ["store_id", "username", "password"],
        "optional_fields": ["2fa_method", "2fa_phone", "2fa_email"]
    },
    "dealersocket": {
        "base_url": "https://{region}.dealersocket.com",
        "api_version": "v1",
        "required_fields": ["region", "username", "password"],
        "optional_fields": ["2fa_method", "store_id"]
    }
}

# Security settings
SECURITY = {
    "key_rotation_days": 30,
    "min_password_length": 12,
    "require_special_chars": True,
    "hash_algorithm": "argon2",
    "encryption_algorithm": "AES-256-GCM"
}

# Logging configuration
LOGGING = {
    "level": "INFO",
    "format": "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file_path": "logs/nova_act.log",
    "max_size": 10485760,  # 10MB
    "backup_count": 5
}

# Health check settings
HEALTH_CHECK = {
    "interval": 300,  # 5 minutes
    "timeout": 10,
    "unhealthy_threshold": 3,
    "endpoints": {
        "vinsolutions": "/api/health",
        "dealersocket": "/health/status"
    }
}