"""
Secure configuration management for Watchdog AI.
Handles environment variables, secrets, and configuration validation.
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
import logging
import warnings

# Import from new path structure
from watchdog_ai.core.config.logging import get_logger

logger = get_logger(__name__)

# Confidence threshold for automatically mapping columns without clarification
MIN_CONFIDENCE_TO_AUTOMAP = 0.7

# Whether to automatically drop unmapped columns after clarification
DROP_UNMAPPED_COLUMNS = False  # Default to off

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            details: Optional details about the error
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

class SecureConfig:
    """Manages secure configuration and secrets."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize secure configuration.
        
        Args:
            env_file: Optional path to .env file
        """
        self.config_values = {}
        self.encrypted_values = {}
        self._encryption_key = None
        self._load_encryption_key()
        
        # Load configuration
        self._load_environment(env_file)
        self._validate_config()
        
        # Initialize rate limiting
        self.rate_limits = {}
        self.rate_limit_window = timedelta(minutes=5)
        self.max_requests = 100  # Default rate limit
    
    def _load_encryption_key(self):
        """Load or generate encryption key."""
        key_file = Path(".secret_key")
        if key_file.exists():
            self._encryption_key = key_file.read_bytes()
        else:
            self._encryption_key = Fernet.generate_key()
            key_file.write_bytes(self._encryption_key)
            key_file.chmod(0o600)  # Secure permissions
    
    def _load_environment(self, env_file: Optional[str] = None):
        """
        Load configuration from environment and .env file.
        
        Args:
            env_file: Optional path to .env file
        """
        # Load .env file if provided
        if env_file and Path(env_file).exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Required configuration keys and their validation functions
        required_config = {
            'OPENAI_API_KEY': self._validate_api_key,
            'USE_MOCK': lambda x: str(x).lower() in ['true', 'false', '1', '0'],
            'LOG_LEVEL': lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            'MAX_UPLOAD_SIZE_MB': lambda x: x.isdigit() and 1 <= int(x) <= 100
        }
        
        # Load and validate required configuration
        for key, validator in required_config.items():
            value = os.getenv(key)
            if value is None:
                if key == 'USE_MOCK':
                    value = 'true'  # Default to mock mode if not specified
                elif key == 'LOG_LEVEL':
                    value = 'INFO'  # Default log level
                elif key == 'MAX_UPLOAD_SIZE_MB':
                    value = '100'  # Default max upload size
                else:
                    logger.warning(f"Missing required configuration: {key}")
                    continue
            
            if not validator(value):
                raise ConfigurationError(
                    f"Invalid configuration value for {key}",
                    details={"key": key, "value": value}
                )
            
            self.config_values[key] = value
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        # Validate API keys if not in mock mode
        if not self.get_bool('USE_MOCK'):
            api_key = self.get_secret('OPENAI_API_KEY')
            if not api_key:
                raise ConfigurationError(
                    "API key required when not in mock mode",
                    details={"missing_key": "OPENAI_API_KEY"}
                )
    
    def _validate_api_key(self, key: str) -> bool:
        """
        Validate API key format.
        
        Args:
            key: API key to validate
            
        Returns:
            bool indicating if key format is valid
        """
        # Basic validation for common API key formats
        if not key:
            return False
        
        # OpenAI API key format
        if key.startswith('sk-') and len(key) > 20:
            return True
        
        # Anthropic API key format
        if key.startswith('sk-ant-') and len(key) > 20:
            return True
        
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config_values.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get boolean configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Boolean configuration value
        """
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ['true', '1', 'yes']
    
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get integer configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Integer configuration value
        """
        value = self.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Get encrypted secret value.
        
        Args:
            key: Secret key
            
        Returns:
            Decrypted secret value or None
        """
        if key in self.encrypted_values:
            f = Fernet(self._encryption_key)
            return f.decrypt(self.encrypted_values[key]).decode()
        return self.get(key)
    
    def set_secret(self, key: str, value: str):
        """
        Set encrypted secret value.
        
        Args:
            key: Secret key
            value: Secret value to encrypt
        """
        f = Fernet(self._encryption_key)
        self.encrypted_values[key] = f.encrypt(value.encode())
    
    def check_rate_limit(self, key: str) -> bool:
        """
        Check if a rate limit has been exceeded.
        
        Args:
            key: Rate limit key (e.g., IP address or API key)
            
        Returns:
            bool indicating if request should be allowed
        """
        now = datetime.now()
        
        # Clean up old entries
        self.rate_limits = {
            k: v for k, v in self.rate_limits.items()
            if v['window_start'] + self.rate_limit_window > now
        }
        
        # Get or create rate limit entry
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                'window_start': now,
                'count': 0
            }
        
        # Reset window if needed
        if self.rate_limits[key]['window_start'] + self.rate_limit_window < now:
            self.rate_limits[key] = {
                'window_start': now,
                'count': 0
            }
        
        # Increment counter and check limit
        self.rate_limits[key]['count'] += 1
        return self.rate_limits[key]['count'] <= self.max_requests
    
    def generate_request_signature(self, data: Dict[str, Any]) -> str:
        """
        Generate HMAC signature for request data.
        
        Args:
            data: Request data to sign
            
        Returns:
            Base64 encoded signature
        """
        # Sort keys for consistent ordering
        sorted_data = json.dumps(data, sort_keys=True)
        
        # Generate HMAC
        h = hmac.new(
            self._encryption_key,
            sorted_data.encode(),
            hashlib.sha256
        )
        
        return base64.b64encode(h.digest()).decode()
    
    def verify_request_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """
        Verify HMAC signature for request data.
        
        Args:
            data: Request data to verify
            signature: Expected signature
            
        Returns:
            bool indicating if signature is valid
        """
        expected = self.generate_request_signature(data)
        return hmac.compare_digest(signature, expected)


# Global configuration instance
config = SecureConfig()

