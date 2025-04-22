"""
Engine settings configuration for LLM engine.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .api_config import APIConfig
from .prompt_config import SystemPrompts

logger = logging.getLogger(__name__)

@dataclass
class AnalysisSettings:
    """Settings for analysis components."""
    pattern_confidence_threshold: float = 0.05
    min_data_points: int = 10
    max_anomaly_percentage: float = 20.0
    correlation_threshold: float = 0.6
    seasonality_min_periods: int = 12
    trend_window_size: int = 10

@dataclass
class ValidationSettings:
    """Settings for response validation."""
    min_summary_length: int = 10
    max_summary_length: int = 500
    min_insights: int = 1
    max_insights: int = 10
    min_flags: int = 1
    max_flags: int = 5
    required_fields: list = field(default_factory=lambda: [
        "summary", "value_insights", "actionable_flags", "confidence"
    ])

@dataclass
class CacheSettings:
    """Settings for response caching."""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 1000
    persistent: bool = True
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.expanduser("~"),
        ".watchdog_ai",
        "cache"
    ))

@dataclass
class LoggingSettings:
    """Settings for engine logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = field(default_factory=lambda: os.path.join(
        os.path.expanduser("~"),
        ".watchdog_ai",
        "logs",
        "llm_engine.log"
    ))
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

class EngineSettings:
    """Manages overall LLM engine settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize engine settings.
        
        Args:
            config_path: Optional path to config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.api_config = APIConfig()
        self.system_prompts = SystemPrompts()
        
        # Load component settings
        self._load_settings()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        return os.path.join(
            os.path.expanduser("~"),
            ".watchdog_ai",
            "engine_settings.yml"
        )
        
    def _load_settings(self) -> None:
        """Load settings from file or use defaults."""
        settings = {}
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    settings = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Error loading settings: {str(e)}")
        
        # Initialize component settings
        self.analysis = AnalysisSettings(
            **settings.get('analysis', {})
        )
        self.validation = ValidationSettings(
            **settings.get('validation', {})
        )
        self.cache = CacheSettings(
            **settings.get('cache', {})
        )
        self.logging = LoggingSettings(
            **settings.get('logging', {})
        )
        
        # Set up logging
        self._configure_logging()
        
    def _configure_logging(self) -> None:
        """Configure logging based on settings."""
        try:
            # Create log directory if needed
            if self.logging.file_enabled:
                os.makedirs(os.path.dirname(self.logging.file_path), exist_ok=True)
            
            # Configure logging
            import logging.handlers
            
            root_logger = logging.getLogger('watchdog_ai.llm')
            root_logger.setLevel(getattr(logging, self.logging.level.upper()))
            
            # Create formatter
            formatter = logging.Formatter(self.logging.format)
            
            # Add file handler if enabled
            if self.logging.file_enabled:
                file_handler = logging.handlers.RotatingFileHandler(
                    self.logging.file_path,
                    maxBytes=self.logging.max_file_size,
                    backupCount=self.logging.backup_count
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            
        except Exception as e:
            logger.error(f"Error configuring logging: {str(e)}")
    
    def save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Prepare settings dict
            settings = {
                'analysis': asdict(self.analysis),
                'validation': asdict(self.validation),
                'cache': asdict(self.cache),
                'logging': asdict(self.logging),
                'last_updated': datetime.now().isoformat()
            }
            
            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(settings, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            return False
    
    def update_settings(self, 
                       analysis: Optional[Dict[str, Any]] = None,
                       validation: Optional[Dict[str, Any]] = None,
                       cache: Optional[Dict[str, Any]] = None,
                       logging: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update engine settings.
        
        Args:
            analysis: Analysis settings updates
            validation: Validation settings updates
            cache: Cache settings updates
            logging: Logging settings updates
            
        Returns:
            Boolean indicating success
        """
        try:
            if analysis:
                for key, value in analysis.items():
                    if hasattr(self.analysis, key):
                        setattr(self.analysis, key, value)
            
            if validation:
                for key, value in validation.items():
                    if hasattr(self.validation, key):
                        setattr(self.validation, key, value)
            
            if cache:
                for key, value in cache.items():
                    if hasattr(self.cache, key):
                        setattr(self.cache, key, value)
            
            if logging:
                for key, value in logging.items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
                # Reconfigure logging if settings changed
                self._configure_logging()
            
            return self.save_settings()
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return False
    
    def get_engine_config(self) -> Dict[str, Any]:
        """Get complete engine configuration."""
        return {
            'api': self.api_config.get_client_config(),
            'completion': self.api_config.get_completion_config(),
            'analysis': asdict(self.analysis),
            'validation': asdict(self.validation),
            'cache': asdict(self.cache),
            'logging': asdict(self.logging)
        }

