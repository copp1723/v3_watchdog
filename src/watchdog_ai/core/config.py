"""
Configuration management for Watchdog AI application.

This module handles loading and validation of environment variables,
management of API keys and credentials, and application settings.
"""

import os
import logging
from typing import Dict, Optional, Union, List, Any
from enum import Enum
from pathlib import Path

import dotenv
from pydantic import BaseSettings, Field, validator, SecretStr, AnyHttpUrl

# Configure logging
logger = logging.getLogger(__name__)

# Load .env file if it exists
dotenv.load_dotenv()

class EnvironmentType(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OpenAISettings(BaseSettings):
    """OpenAI API configuration settings."""
    api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    organization_id: Optional[str] = Field(None, env="OPENAI_ORGANIZATION_ID")
    model: str = Field("gpt-4", env="OPENAI_MODEL")
    temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")
    max_tokens: int = Field(4000, env="OPENAI_MAX_TOKENS")
    
    class Config:
        env_prefix = "OPENAI_"
        case_sensitive = False

class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    engine: str = Field("sqlite", env="DB_ENGINE")
    host: Optional[str] = Field(None, env="DB_HOST")
    port: Optional[int] = Field(None, env="DB_PORT")
    username: Optional[SecretStr] = Field(None, env="DB_USERNAME")
    password: Optional[SecretStr] = Field(None, env="DB_PASSWORD")
    database: str = Field("watchdog_ai.db", env="DB_DATABASE")
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False
    
    def get_connection_string(self) -> str:
        """Returns database connection string based on engine type."""
        if self.engine.lower() == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.engine.lower() in ["postgres", "postgresql"]:
            password = self.password.get_secret_value() if self.password else ""
            username = self.username.get_secret_value() if self.username else ""
            return f"postgresql://{username}:{password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database engine: {self.engine}")

class NovaActCredentials(BaseSettings):
    """Nova Act CRM API credentials."""
    api_key: SecretStr = Field(..., env="NOVA_ACT_API_KEY")
    api_url: AnyHttpUrl = Field("https://api.dealersocket.com/v1", env="NOVA_ACT_API_URL")
    username: Optional[SecretStr] = Field(None, env="NOVA_ACT_USERNAME")
    password: Optional[SecretStr] = Field(None, env="NOVA_ACT_PASSWORD")
    client_id: Optional[str] = Field(None, env="NOVA_ACT_CLIENT_ID")
    
    class Config:
        env_prefix = "NOVA_ACT_"
        case_sensitive = False

class CRMSettings(BaseSettings):
    """CRM integration settings."""
    enabled: bool = Field(False, env="CRM_ENABLED")
    provider: str = Field("nova_act", env="CRM_PROVIDER")
    sync_interval_minutes: int = Field(60, env="CRM_SYNC_INTERVAL_MINUTES")
    nova_act: Optional[NovaActCredentials] = None
    
    class Config:
        env_prefix = "CRM_"
        case_sensitive = False
    
    @validator('nova_act', always=True, pre=True)
    def load_nova_act_credentials(cls, v, values):
        if values.get("provider") == "nova_act" and values.get("enabled"):
            return NovaActCredentials()
        return v

class ApplicationSettings(BaseSettings):
    """General application settings."""
    secret_key: SecretStr = Field(..., env="SECRET_KEY")
    environment: EnvironmentType = Field(EnvironmentType.DEVELOPMENT, env="ENVIRONMENT")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    allow_signup: bool = Field(False, env="ALLOW_SIGNUP")
    max_upload_size_mb: int = Field(10, env="MAX_UPLOAD_SIZE_MB")
    data_retention_days: int = Field(90, env="DATA_RETENTION_DAYS")
    
    class Config:
        env_prefix = ""
        case_sensitive = False

class Settings(BaseSettings):
    """Main application settings container."""
    application: ApplicationSettings = ApplicationSettings()
    openai: OpenAISettings = OpenAISettings()
    database: DatabaseSettings = DatabaseSettings()
    crm: CRMSettings = CRMSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        try:
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.parent

def get_settings() -> Settings:
    """Return the application settings."""
    return Settings.from_env()

# Global settings instance
settings = get_settings()

