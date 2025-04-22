"""
API configuration management for LLM engine.
"""

import os
import json
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class APIConfig:
    """Manages API configuration and credentials."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize API configuration.
        
        Args:
            config_path: Optional path to config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        return os.path.join(
            os.path.expanduser("~"),
            ".watchdog_ai",
            "llm_config.json"
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config = {
            "api_key": None,
            "organization_id": None,
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 1000,
            "api_base": "https://api.openai.com/v1",
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 1
        }
        
        # Try to load from config file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                logger.warning(f"Error loading config file: {str(e)}")
        
        # Override with environment variables
        env_mapping = {
            "OPENAI_API_KEY": "api_key",
            "OPENAI_ORG_ID": "organization_id",
            "OPENAI_MODEL": "model",
            "OPENAI_TEMPERATURE": "temperature",
            "OPENAI_MAX_TOKENS": "max_tokens",
            "OPENAI_API_BASE": "api_base",
            "OPENAI_TIMEOUT": "timeout",
            "OPENAI_RETRY_COUNT": "retry_count",
            "OPENAI_RETRY_DELAY": "retry_delay"
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert numeric values
                    if config_key in ["temperature", "max_tokens", "timeout", 
                                    "retry_count", "retry_delay"]:
                        value = float(value)
                    config[config_key] = value
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {value}")
        
        return config
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False
    
    def get_api_key(self) -> Optional[str]:
        """Get API key with proper error handling."""
        api_key = self.config.get("api_key")
        if not api_key:
            logger.warning("API key not found in config or environment")
        return api_key
    
    def get_client_config(self) -> Dict[str, Any]:
        """Get configuration for API client."""
        return {
            "api_key": self.get_api_key(),
            "organization_id": self.config.get("organization_id"),
            "api_base": self.config.get("api_base"),
            "timeout": self.config.get("timeout", 30),
            "max_retries": self.config.get("retry_count", 3)
        }
    
    def get_completion_config(self) -> Dict[str, Any]:
        """Get configuration for completions API."""
        return {
            "model": self.config.get("model", "gpt-3.5-turbo"),
            "temperature": self.config.get("temperature", 0.3),
            "max_tokens": self.config.get("max_tokens", 1000)
        }
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            Boolean indicating success
        """
        try:
            # Validate updates
            for key, value in updates.items():
                if key not in self.config:
                    logger.warning(f"Unknown config key: {key}")
                    continue
                
                # Type checking
                expected_type = type(self.config[key])
                if not isinstance(value, expected_type):
                    try:
                        value = expected_type(value)
                    except:
                        logger.warning(f"Invalid type for {key}: {type(value)}")
                        continue
                
                self.config[key] = value
            
            # Save updated config
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return False

