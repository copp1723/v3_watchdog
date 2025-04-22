"""
Custom exceptions for Watchdog AI.

This module defines application-specific exceptions that provide
more context and structured error handling throughout the application.
"""

import logging
from typing import Optional, Dict, Any, List, Union
from enum import Enum, auto

# Configure logger
logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """Error codes for categorizing exceptions."""
    # Configuration errors (1000-1999)
    CONFIG_ERROR = 1000
    ENVIRONMENT_ERROR = 1001
    SECRET_KEY_ERROR = 1002
    
    # Authentication/Authorization errors (2000-2999)
    AUTH_ERROR = 2000
    LOGIN_FAILED = 2001
    UNAUTHORIZED = 2002
    SESSION_EXPIRED = 2003
    TOKEN_INVALID = 2004
    
    # Data validation errors (3000-3999)
    VALIDATION_ERROR = 3000
    SCHEMA_ERROR = 3001
    DATA_TYPE_ERROR = 3002
    MISSING_REQUIRED_FIELD = 3003
    DATA_FORMAT_ERROR = 3004
    COLUMN_MAPPING_ERROR = 3005
    
    # API integration errors (4000-4999)
    API_ERROR = 4000
    NETWORK_ERROR = 4001
    TIMEOUT_ERROR = 4002
    API_RESPONSE_ERROR = 4003
    CRM_CONNECTION_ERROR = 4004
    OPENAI_API_ERROR = 4005
    
    # Resource limit errors (5000-5999)
    RESOURCE_LIMIT_ERROR = 5000
    API_RATE_LIMIT = 5001
    STORAGE_LIMIT_EXCEEDED = 5002
    UPLOAD_SIZE_EXCEEDED = 5003
    TOKEN_LIMIT_EXCEEDED = 5004
    
    # File operation errors (6000-6999)
    FILE_ERROR = 6000
    FILE_NOT_FOUND = 6001
    FILE_ACCESS_DENIED = 6002
    FILE_READ_ERROR = 6003
    FILE_WRITE_ERROR = 6004
    
    # Internal errors (9000-9999)
    INTERNAL_ERROR = 9000
    UNEXPECTED_ERROR = 9999

class WatchdogAIError(Exception):
    """Base exception for all Watchdog AI errors."""
    
    def __init__(
        self, 
        message: str = "An error occurred",
        error_code: ErrorCode = ErrorCode.UNEXPECTED_ERROR,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 500,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.http_status_code = http_status_code
        self.original_exception = original_exception
        
        # Log the error
        self._log_error()
        
        # Initialize the base Exception
        super().__init__(self.message)
    
    def _log_error(self) -> None:
        """Log the error with appropriate level and details."""
        log_message = f"{self.error_code.name} ({self.error_code.value}): {self.message}"
        
        # Add details if present
        if self.details:
            log_message += f" | Details: {self.details}"
        
        # Add original exception if present
        if self.original_exception:
            log_message += f" | Original exception: {str(self.original_exception)}"
        
        # Log with appropriate level based on error code
        if self.error_code.value < 2000:  # Configuration errors
            logger.error(log_message)
        elif self.error_code.value < 4000:  # Auth and validation errors
            logger.warning(log_message)
        elif self.error_code.value < 6000:  # API and resource errors
            logger.error(log_message)
        else:  # File and internal errors
            logger.critical(log_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details
        }

# Configuration Errors
class ConfigurationError(WatchdogAIError):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        error_code: ErrorCode = ErrorCode.CONFIG_ERROR,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 500,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

# Authentication/Authorization Errors
class AuthenticationError(WatchdogAIError):
    """Exception raised for authentication-related errors."""
    
    def __init__(
        self,
        message: str = "Authentication error",
        error_code: ErrorCode = ErrorCode.AUTH_ERROR,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 401,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

# Data Validation Errors
class ValidationError(WatchdogAIError):
    """Exception raised for data validation errors."""
    
    def __init__(
        self,
        message: str = "Data validation error",
        error_code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 400,
        original_exception: Optional[Exception] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None
    ):
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors
            
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

class SchemaError(ValidationError):
    """Exception raised for schema-related errors."""
    
    def __init__(
        self,
        message: str = "Schema validation error",
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 400,
        original_exception: Optional[Exception] = None,
        schema_errors: Optional[List[Dict[str, Any]]] = None
    ):
        details = details or {}
        if schema_errors:
            details["schema_errors"] = schema_errors
            
        super().__init__(
            message=message,
            error_code=ErrorCode.SCHEMA_ERROR,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

class ColumnMappingError(ValidationError):
    """Exception raised for column mapping errors."""
    
    def __init__(
        self,
        message: str = "Column mapping error",
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 400,
        original_exception: Optional[Exception] = None,
        missing_columns: Optional[List[str]] = None,
        invalid_mappings: Optional[Dict[str, str]] = None
    ):
        details = details or {}
        if missing_columns:
            details["missing_columns"] = missing_columns
        if invalid_mappings:
            details["invalid_mappings"] = invalid_mappings
            
        super().__init__(
            message=message,
            error_code=ErrorCode.COLUMN_MAPPING_ERROR,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

# API Integration Errors
class APIError(WatchdogAIError):
    """Exception raised for API-related errors."""
    
    def __init__(
        self,
        message: str = "API error",
        error_code: ErrorCode = ErrorCode.API_ERROR,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 500,
        original_exception: Optional[Exception] = None,
        api_response: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if api_response:
            details["api_response"] = api_response
            
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

class CRMConnectionError(APIError):
    """Exception raised for CRM connection errors."""
    
    def __init__(
        self,
        message: str = "CRM connection error",
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 503,
        original_exception: Optional[Exception] = None,
        provider: Optional[str] = None
    ):
        details = details or {}
        if provider:
            details["provider"] = provider
            
        super().__init__(
            message=message,
            error_code=ErrorCode.CRM_CONNECTION_ERROR,
            details=details,
            http_status_code=http_status_code,
            original_exception=original_exception
        )

class OpenAIAPIError(APIError):
    """Exception raised for OpenAI API errors."""
    
    def __init__(
        self,
        message: str = "OpenAI API error",
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 500,
        original_exception: Optional[Exception] = None,
        model: Optional[str] = None
    ):
        details = details or {}
        if model:
            details["model"] = model
            
        super().__init__(
            message=message,
            error_code=ErrorCode.OPENAI_API_ERROR,
            details=details,
            http_status_code=http_status_code,
            original_

