"""
Base CRM Adapter Interface

Defines the abstract base class that all CRM adapters must implement.
This provides a consistent interface for different CRM systems.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import requests

# Configure logger
logger = logging.getLogger(__name__)


class BaseCRMAdapter(ABC):
    """
    Abstract base class for CRM system adapters.
    
    All CRM integrations must implement this interface to ensure
    consistent behavior across different CRM systems.
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize the CRM adapter.
        
        Args:
            base_url: The base URL for the CRM API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = None
        self.auth_token = None
        self.authenticated = False
        
        logger.info(f"Initialized {self.__class__.__name__} adapter for {base_url}")
    
    @abstractmethod
    def authenticate(self) -> None:
        """
        Authenticate with the CRM system.
        
        This method should set self.authenticated to True upon successful
        authentication and store any necessary tokens in self.auth_token.
        
        Raises:
            ConnectionError: If connection to CRM fails
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    def pull_sales(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Pull sales data from the CRM system.
        
        Args:
            since: Optional datetime to pull sales since a specific date
                  If None, pulls data based on adapter-specific logic
                  
        Returns:
            List of sales records as dictionaries
            
        Raises:
            ConnectionError: If connection to CRM fails
            AuthenticationError: If not authenticated
            DataFetchError: If data retrieval fails
        """
        pass
    
    @abstractmethod
    def push_insights(self, insights: List[Dict[str, Any]]) -> None:
        """
        Push insights data to the CRM system.
        
        Args:
            insights: List of insight records to push to CRM
            
        Raises:
            ConnectionError: If connection to CRM fails
            AuthenticationError: If not authenticated
            DataPushError: If data push fails
        """
        pass
    
    def _request(self, method: str, endpoint: str, 
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                retry_count: int = 3) -> Tuple[requests.Response, Any]:
        """
        Helper method to make HTTP requests to the CRM API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (will be appended to base_url)
            params: Optional query parameters
            data: Optional request data (will be JSON-encoded)
            headers: Optional request headers
            retry_count: Number of retry attempts
            
        Returns:
            Tuple of (Response object, Parsed JSON data)
            
        Raises:
            ConnectionError: If connection fails after retries
            requests.RequestException: For other request errors
        """
        if not self.authenticated:
            logger.warning("Making request without authentication")
        
        # Ensure session exists
        if self.session is None:
            self.session = requests.Session()
        
        # Build request headers
        request_headers = {'Content-Type': 'application/json'}
        if headers:
            request_headers.update(headers)
        
        # Add auth token if available
        if self.auth_token:
            request_headers['Authorization'] = f'Bearer {self.auth_token}'
        elif self.api_key:
            request_headers['X-API-Key'] = self.api_key
        
        # Prepare URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Log request (sanitize sensitive data)
        safe_headers = {k: v for k, v in request_headers.items() 
                       if k.lower() not in ('authorization', 'x-api-key')}
        logger.debug(f"Making {method} request to {url} with headers {safe_headers}")
        
        attempts = 0
        last_error = None
        
        # Retry loop
        while attempts < retry_count:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data if data else None,
                    headers=request_headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                
                # Log response status
                logger.debug(f"Response status: {response.status_code}")
                
                # Raise for error status codes
                response.raise_for_status()
                
                # Parse JSON response
                try:
                    json_data = response.json() if response.content else {}
                    return response, json_data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON from response: {response.text[:100]}...")
                    return response, {}
                    
            except requests.RequestException as e:
                attempts += 1
                last_error = e
                logger.warning(f"Request attempt {attempts} failed: {str(e)}")
                
                if attempts < retry_count:
                    logger.info(f"Retrying in 1 second... ({attempts}/{retry_count})")
                    import time
                    time.sleep(1)
                    
        # If we get here, all retries failed
        logger.error(f"All {retry_count} request attempts failed")
        raise last_error or ConnectionError(f"Failed to connect to {self.base_url}")


class AuthenticationError(Exception):
    """Exception raised when authentication with CRM system fails."""
    pass


class DataFetchError(Exception):
    """Exception raised when data retrieval from CRM system fails."""
    pass


class DataPushError(Exception):
    """Exception raised when data push to CRM system fails."""
    pass

