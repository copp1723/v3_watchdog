"""
Nova Act CRM Adapter Implementation

This module provides an implementation of the BaseCRMAdapter interface
for the Nova Act CRM system.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .base import BaseCRMAdapter, AuthenticationError, DataFetchError, DataPushError

# Configure logger
logger = logging.getLogger(__name__)

# Default mock data file path
DEFAULT_MOCK_PATH = Path("data/mock/nova_act_sales.json")

# Environment variable names
ENV_BASE_URL = "NOVA_ACT_BASE_URL"
ENV_API_KEY = "NOVA_ACT_API_KEY"
ENV_USERNAME = "NOVA_ACT_USERNAME"
ENV_PASSWORD = "NOVA_ACT_PASSWORD"


class NovaActAdapter(BaseCRMAdapter):
    """
    Nova Act CRM system adapter implementation.
    
    This adapter connects to the Nova Act CRM system API and provides
    methods to authenticate, pull sales data, and push insights.
    
    Since API documentation may not be available, this implementation
    includes fallback to mock data and stub implementations.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                username: Optional[str] = None, password: Optional[str] = None,
                timeout: int = 30, verify_ssl: bool = True, 
                mock_data_path: Optional[Union[str, Path]] = None):
        """
        Initialize the Nova Act CRM adapter.
        
        Args:
            base_url: The base URL for the Nova Act API. If None, uses NOVA_ACT_BASE_URL env var
            api_key: API key for authentication. If None, uses NOVA_ACT_API_KEY env var
            username: Username for authentication. If None, uses NOVA_ACT_USERNAME env var
            password: Password for authentication. If None, uses NOVA_ACT_PASSWORD env var
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            mock_data_path: Path to mock data file for testing. Default is data/mock/nova_act_sales.json
        """
        # Load configuration from environment variables if not provided
        self.base_url = base_url or os.environ.get(ENV_BASE_URL, "https://api.novaact.example")
        api_key = api_key or os.environ.get(ENV_API_KEY)
        
        # Initialize base class
        super().__init__(base_url=self.base_url, api_key=api_key, 
                        timeout=timeout, verify_ssl=verify_ssl)
        
        # Set additional authentication options
        self.username = username or os.environ.get(ENV_USERNAME)
        self.password = password or os.environ.get(ENV_PASSWORD)
        
        # Set mock data configuration
        self.mock_data_path = Path(mock_data_path or DEFAULT_MOCK_PATH)
        self.use_mock_data = not all([self.base_url, (self.api_key or (self.username and self.password))])
        
        if self.use_mock_data:
            logger.warning("Incomplete API configuration. Using mock data mode.")
            
            # Create directory for mock data if it doesn't exist
            self.mock_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized NovaActAdapter with base_url={self.base_url}, "
                   f"mock_mode={self.use_mock_data}")
    
    def authenticate(self) -> None:
        """
        Authenticate with the Nova Act CRM system.
        
        In mock mode, this returns a dummy token.
        In real mode, this attempts to get an authentication token via API.
        
        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection to Nova Act fails
        """
        if self.authenticated:
            logger.debug("Already authenticated")
            return
        
        logger.info("Authenticating with Nova Act CRM")
        
        if self.use_mock_data:
            # Mock authentication
            logger.info("Using mock authentication")
            self.auth_token = "mock-nova-act-token-12345"
            self.authenticated = True
            time.sleep(0.5)  # Simulate API delay
            logger.info("Mock authentication successful")
            return
        
        try:
            # Determine authentication method
            if self.api_key:
                # API key authentication
                logger.debug("Using API key authentication")
                
                # Simulate API request (to be replaced with actual API call)
                _, data = self._request(
                    method="POST",
                    endpoint="/auth/validate",
                    headers={"X-API-Key": self.api_key}
                )
                
                # Check response and extract token
                if "token" in data:
                    self.auth_token = data["token"]
                    self.authenticated = True
                    logger.info("Authentication successful with API key")
                else:
                    raise AuthenticationError("API key authentication failed: Missing token in response")
                
            elif self.username and self.password:
                # Username/password authentication
                logger.debug("Using username/password authentication")
                
                # Simulate API request (to be replaced with actual API call)
                _, data = self._request(
                    method="POST",
                    endpoint="/auth/login",
                    data={
                        "username": self.username,
                        "password": self.password
                    }
                )
                
                # Check response and extract token
                if "token" in data:
                    self.auth_token = data["token"]
                    self.authenticated = True
                    logger.info("Authentication successful with username/password")
                else:
                    raise AuthenticationError("Username/password authentication failed: Missing token in response")
                
            else:
                raise AuthenticationError("No authentication credentials provided")
                
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            self.authenticated = False
            self.auth_token = None
            raise AuthenticationError(f"Failed to authenticate with Nova Act: {str(e)}") from e
    
    def pull_sales(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Pull sales data from the Nova Act CRM system.
        
        In mock mode, returns data from mock file or generates mock data.
        In real mode, calls the Nova Act API to retrieve sales data.
        
        Args:
            since: Optional datetime to pull sales since a specific date
                  If None, pulls most recent data
        
        Returns:
            List of sales records as dictionaries
            
        Raises:
            DataFetchError: If data retrieval fails
            AuthenticationError: If not authenticated
            ConnectionError: If connection to Nova Act fails
        """
        # Ensure we're authenticated
        if not self.authenticated:
            self.authenticate()
        
        since_str = since.isoformat() if since else "all"
        logger.info(f"Pulling sales data since {since_str}")
        
        if self.use_mock_data:
            # Mock data mode - try to load from file or generate
            logger.info(f"Using mock sales data from {self.mock_data_path}")
            
            try:
                # Try to load mock data from file
                if self.mock_data_path.exists():
                    with open(self.mock_data_path, 'r') as f:
                        sales_data = json.load(f)
                        logger.info(f"Loaded {len(sales_data)} mock sales records")
                        return sales_data
                
                # Generate mock data if file doesn't exist
                logger.warning(f"Mock data file not found. Generating mock data...")
                sales_data = self._generate_mock_sales(count=25, since=since)
                
                # Save mock data for future use
                self.mock_data_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.mock_data_path, 'w') as f:
                    json.dump(sales_data, f, indent=2)
                
                logger.info(f"Generated and saved {len(sales_data)} mock sales records")
                return sales_data
                
            except Exception as e:
                logger.error(f"Error loading/generating mock data: {str(e)}")
                raise DataFetchError(f"Failed to load/generate mock sales data: {str(e)}") from e
        
        # Real API mode
        try:
            # Prepare query parameters
            params = {}
            if since:
                params["since"] = since.isoformat()
            
            # Make API request
            _, data = self._request(
                method="GET",
                endpoint="/sales",
                params=params
            )
            
            if not isinstance(data, list):
                if isinstance(data, dict) and "sales" in data and isinstance(data["sales"], list):
                    sales_data = data["sales"]
                else:
                    raise DataFetchError(f"Unexpected response format: {data}")
            else:
                sales_data = data
            
            logger.info(f"Retrieved {len(sales_data)} sales records from Nova Act")
            return sales_data
            
        except Exception as e:
            logger.error(f"Error fetching sales data: {str(e)}")
            raise DataFetchError(f"Failed to retrieve sales data from Nova Act: {str(e)}") from e
    
    def push_insights(self, insights: List[Dict[str, Any]]) -> None:
        """
        Push insights data to the Nova Act CRM system.
        
        In mock mode, this logs the insights and pretends to push them.
        In real mode, this sends the insights to the Nova Act API.
        
        Args:
            insights: List of insight records to push to CRM
            
        Raises:
            DataPushError: If data push fails
            AuthenticationError: If not authenticated
            ConnectionError: If connection to Nova Act fails
        """
        # Ensure we're authenticated
        if not self.authenticated:
            self.authenticate()
        
        if not insights:
            logger.warning("No insights provided to push")
            return
        
        logger.info(f"Pushing {len(insights)} insights to Nova Act CRM")
        
        if self.use_mock_data:
            # Mock mode - just log the insights
            logger.info("Using mock mode for pushing insights")
            
            # Log a sample of the insights
            sample_size = min(3, len(insights))
            insight_sample = insights[:sample_size]
            logger.info(f"Sample of insights being pushed (mock mode): {json.dumps(insight_sample, indent=2)}")
            
            # Simulate API delay
            time.sleep(0.5 + (0.1 * len(insights)))
            
            logger.info(f"Successfully pushed {len(insights)} insights (mock mode)")
            return
        
        # Real API mode
        try:
            # Send insights to API
            _, data = self._request(
                method="POST",
                endpoint="/insights",
                data={"insights": insights}
            )
            
            # Check response
            if "success" in data and data["success"]:
                logger.info(f"Successfully pushed {len(insights)} insights to Nova Act")
            else:
                error_msg = data.get("message", "Unknown error")
                logger.error(f"Error pushing insights: {error_msg}")
                raise DataPushError(f"Failed to push insights: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error pushing insights: {str(e)}")
            raise DataPushError(f"Failed to push insights to Nova Act: {str(e)}") from e
    
    def _generate_mock_sales(self, count: int = 20, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Generate mock sales data for testing.
        
        Args:
            count: Number of mock sales records to generate
            since: Optional datetime for the oldest sale
            
        Returns:
            List of mock sales records
        """
        import random
        from datetime import timedelta
        
        # Set base date for sales
        now = datetime.now()
        if since is None:
            since = now - timedelta(days=30)
        
        # Product data for mock generation
        products = [
            {"id": "P001", "name": "Enterprise Suite", "price": 2499.99},
            {"id": "P002", "name": "Business Analytics", "price": 999.99},
            {"id": "P003", "name": "CRM Basic", "price": 499.99},
            {"id": "P004", "name": "Mobile Integration", "price": 299.99},
            {"id": "P005", "name": "Support Package", "price": 199.99}
        ]
        
        # Customer data for mock generation
        customers = [
            {"id": "C001", "name": "Acme Corp", "contact": "John Smith"},
            {"id": "C002", "name": "TechStart Inc", "contact": "Jane Doe"},
            {"id": "C003", "name": "Global Services", "contact": "Bob Johnson"},
            {"id": "C004", "name": "Local Business", "contact": "Alice Williams"},
            {"id": "C005", "name": "MegaCorp", "contact": "Carol Brown"}
        ]
        
        # Sales representatives
        reps = [
            {"id": "SR001", "name": "Michael Scott"},
            {"id": "SR002", "name": "Dwight Schrute"},
            {"id": "SR003", "name": "Jim Halpert"},
            {"id": "SR004", "name": "Pam Beesly"}
        ]
        
        # Generate random sales data
        sales_data = []
        for i in range(count):
            # Random date between since and now
            date_range = (now - since).days
            sale_date = since + timedelta(days=random.randint(0, date_range))
            
            # Random product and quantity
            product = random.choice(products)
            quantity = random.randint(1, 5)
            
            # Random customer and sales rep
            customer = random.choice(customers)
            rep = random.choice(reps)
            
            # Generate sale record
            sale = {
                "id": f"S{1000 + i}",
                "date": sale_date.isoformat(),
                "customer": customer,
                "product": product,
                "quantity": quantity,
                "total_amount": round(product["price"] * quantity, 2),
                "sales_rep": rep,
                "status": random.choice(["completed", "pending", "invoiced"]),
                "notes": f"Mock sale generated for testing on {now.isoformat()}"
            }
            
            sales_data.append(sale)
        
        # Sort by date descending
        sales_data.sort(key=lambda x: x["date"], reverse=True)
        
        return sales_data

