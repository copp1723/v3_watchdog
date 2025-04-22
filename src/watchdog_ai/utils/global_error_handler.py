"""
Global error handling decorator for Watchdog AI.

This module provides a decorator that can be applied to any function to provide
consistent error handling, logging, and user-friendly error messages.
"""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, cast, Union
import inspect
import asyncio

from src.utils.errors import (
    WatchdogError,
    ValidationError,
    ProcessingError,
    ConfigurationError,
    APIError,
    handle_error,
    format_error_for_ui
)
from src.watchdog_ai.core.config.logging import get_logger

# Set up logger
logger = get_logger(__name__, "info")

# Type variables for function signatures
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')  # Return type

def with_global_error_handling(
    friendly_message: Optional[str] = None,
    log_level: str = "error",
    reraise: bool = False,
    include_ui_feedback: bool = True
) -> Callable[[F], F]:
    """
    Decorator for global error handling across the application.
    
    Args:
        friendly_message: Optional user-friendly message to display on error
        log_level: Logging level for errors (debug, info, warning, error, critical)
        reraise: Whether to reraise the exception after handling
        include_ui_feedback: Whether to include UI feedback information
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function details for context
                module_name = func.__module__
                function_name = func.__name__
                error_location = f"{module_name}.{function_name}"
                
                # Prepare error context
                error_context = {
                    "function": error_location,
                    "args": str(args) if args else "None",
                    "kwargs": str(kwargs) if kwargs else "None",
                    "traceback": traceback.format_exc()
                }
                
                # Log the error with appropriate level
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"Error in {error_location}: {str(e)}")
                logger.debug(f"Error context: {error_context}")
                
                # Handle the error
                error_response = handle_error(
                    e, 
                    user_message=friendly_message or f"An error occurred in {error_location}"
                )
                
                # Add UI feedback if requested
                if include_ui_feedback:
                    error_response["ui_feedback"] = format_error_for_ui(error_response)
                
                # Reraise if requested
                if reraise:
                    raise
                
                return error_response
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get function details for context
                module_name = func.__module__
                function_name = func.__name__
                error_location = f"{module_name}.{function_name}"
                
                # Prepare error context
                error_context = {
                    "function": error_location,
                    "args": str(args) if args else "None",
                    "kwargs": str(kwargs) if kwargs else "None",
                    "traceback": traceback.format_exc()
                }
                
                # Log the error with appropriate level
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"Error in {error_location}: {str(e)}")
                logger.debug(f"Error context: {error_context}")
                
                # Handle the error
                error_response = handle_error(
                    e, 
                    user_message=friendly_message or f"An error occurred in {error_location}"
                )
                
                # Add UI feedback if requested
                if include_ui_feedback:
                    error_response["ui_feedback"] = format_error_for_ui(error_response)
                
                # Reraise if requested
                if reraise:
                    raise
                
                return error_response
        
        # Return the appropriate wrapper based on whether the function is async
        if is_async:
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator

def global_try_except(
    func: Optional[F] = None,
    friendly_message: Optional[str] = None,
    log_level: str = "error",
    reraise: bool = False,
    include_ui_feedback: bool = True
) -> Union[F, Callable[[F], F]]:
    """
    Alternative decorator syntax that can be used with or without arguments.
    
    This allows both:
        @global_try_except
        def my_function(): ...
        
    And:
        @global_try_except(friendly_message="Custom error")
        def my_function(): ...
    
    Args:
        func: Function to decorate (if used without parentheses)
        friendly_message: Optional user-friendly message to display on error
        log_level: Logging level for errors (debug, info, warning, error, critical)
        reraise: Whether to reraise the exception after handling
        include_ui_feedback: Whether to include UI feedback information
        
    Returns:
        Decorated function or decorator function
    """
    if func is None:
        # Used with arguments - @global_try_except(...)
        return with_global_error_handling(
            friendly_message=friendly_message,
            log_level=log_level,
            reraise=reraise,
            include_ui_feedback=include_ui_feedback
        )
    else:
        # Used without arguments - @global_try_except
        return with_global_error_handling()(func)

def handle_api_errors(
    friendly_message: Optional[str] = None,
    log_level: str = "error"
) -> Callable[[F], F]:
    """
    Specialized decorator for API endpoints.
    
    This decorator is designed specifically for API route handlers to ensure 
    consistent error responses for API clients.
    
    Args:
        friendly_message: Optional user-friendly message to display on error
        log_level: Logging level for errors (debug, info, warning, error, critical)
        
    Returns:
        Decorated function with API-specific error handling
    """
    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function details for context
                module_name = func.__module__
                function_name = func.__name__
                error_location = f"{module_name}.{function_name}"
                
                # Log the error
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"API Error in {error_location}: {str(e)}")
                logger.debug(f"API Error traceback: {traceback.format_exc()}")
                
                # Format error for API response
                error_message = friendly_message or str(e)
                error_code = getattr(e, 'error_code', 'UNKNOWN_ERROR') if isinstance(e, WatchdogError) else 'API_ERROR'
                
                # Create API error response
                return {
                    "status": "error",
                    "error_code": error_code,
                    "message": error_message,
                    "details": getattr(e, 'details', {}) if isinstance(e, WatchdogError) else {}
                }
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get function details for context
                module_name = func.__module__
                function_name = func.__name__
                error_location = f"{module_name}.{function_name}"
                
                # Log the error
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"API Error in {error_location}: {str(e)}")
                logger.debug(f"API Error traceback: {traceback.format_exc()}")
                
                # Format error for API response
                error_message = friendly_message or str(e)
                error_code = getattr(e, 'error_code', 'UNKNOWN_ERROR') if isinstance(e, WatchdogError) else 'API_ERROR'
                
                # Create API error response
                return {
                    "status": "error",
                    "error_code": error_code,
                    "message": error_message,
                    "details": getattr(e, 'details', {}) if isinstance(e, WatchdogError) else {}
                }
        
        # Return the appropriate wrapper based on whether the function is async
        if is_async:
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator

