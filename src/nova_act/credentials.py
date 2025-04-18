"""
Secure credential management for Nova Act integration.
"""

import os
import time
import json
import base64
import asyncio
import logging
import hashlib
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dataclasses import dataclass

from .constants import SECURITY
from .logging_config import log_error, log_info, log_warning

logger = logging.getLogger(__name__)

@dataclass
class NovaCredential:
    """Represents a set of vendor credentials."""
    vendor_id: str
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    dealer_code: Optional[str] = None
    dealer_id: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert credential to dictionary format."""
        return {
            "vendor_id": self.vendor_id,
            "username": self.username,
            "password": self.password,
            "api_key": self.api_key,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "dealer_code": self.dealer_code,
            "dealer_id": self.dealer_id,
            "environment": self.environment,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NovaCredential':
        """Create credential from dictionary format."""
        # Convert ISO format strings back to datetime objects
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and data["updated_at"]:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update credential fields."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()

# Vendor configurations
VENDOR_CONFIGS = {
    'vinsolutions': {
        'required_fields': ['username', 'password', 'api_key'],
        'optional_fields': ['dealer_id'],
        'auth_url': 'https://api.vinsolutions.com/auth',
        'base_url': 'https://api.vinsolutions.com/v1'
    },
    'dealersocket': {
        'required_fields': ['client_id', 'client_secret', 'dealer_code'],
        'optional_fields': ['environment'],
        'auth_url': 'https://auth.dealersocket.com/oauth/token',
        'base_url': 'https://api.dealersocket.com/v2'
    }
}

class CredentialManager:
    """Manages secure storage and retrieval of vendor credentials."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize the credential manager.
        
        Args:
            encryption_key: Optional encryption key. If not provided, will look for WATCHDOG_CRED_KEY env var.
        """
        # Get or generate master key
        master_key = encryption_key or os.getenv("WATCHDOG_CRED_KEY")
        if not master_key:
            master_key = self._generate_master_key()
            log_warning(
                "No encryption key provided. Generated new key.",
                "system",
                "credential_init"
            )
        # Ensure master_key is bytes
        if isinstance(master_key, str):
            master_key_bytes = master_key.encode()
        else:
            master_key_bytes = master_key
        
        # Derive encryption keys
        self.keys = self._derive_keys(master_key_bytes)
        
        # Set up storage paths
        self.storage_path = os.path.join(os.path.dirname(__file__), "storage", "credentials")
        self.metadata_path = os.path.join(os.path.dirname(__file__), "storage", "metadata")
        
        # Ensure storage directories exist
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        
        # Initialize rotation tracking
        self.last_rotation = self._load_rotation_timestamp()
        
        # Start key rotation check task
        self._start_rotation_check()
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master encryption key."""
        return base64.urlsafe_b64encode(os.urandom(32))
    
    def _derive_keys(self, master_key: bytes) -> Dict[str, bytes]:
        """Derive encryption keys from master key."""
        # Use PBKDF2 to derive multiple keys
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"NovaActCredentialManager",
            iterations=100000
        )
        
        derived_key = kdf.derive(master_key)
        
        # Create separate keys for different purposes
        keys = {
            "encryption": AESGCM.generate_key(bit_length=256),
            "authentication": derived_key[:16],
            "metadata": derived_key[16:]
        }
        
        return keys
    
    def _load_rotation_timestamp(self) -> datetime:
        """Load the last key rotation timestamp."""
        try:
            metadata_file = os.path.join(self.metadata_path, "rotation.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                return datetime.fromisoformat(data["last_rotation"])
        except Exception as e:
            log_error(e, "system", "load_rotation_timestamp")
        
        # Return current time if no timestamp found
        return datetime.now()
    
    def _save_rotation_timestamp(self):
        """Save the current rotation timestamp."""
        try:
            metadata_file = os.path.join(self.metadata_path, "rotation.json")
            with open(metadata_file, "w") as f:
                json.dump({
                    "last_rotation": datetime.now().isoformat()
                }, f)
        except Exception as e:
            log_error(e, "system", "save_rotation_timestamp")
    
    def _start_rotation_check(self):
        """Start background task to check for key rotation."""
        async def check_rotation():
            while True:
                try:
                    # Check if rotation is needed
                    days_since_rotation = (
                        datetime.now() - self.last_rotation
                    ).days
                    
                    if days_since_rotation >= SECURITY["key_rotation_days"]:
                        await self.rotate_keys()
                    
                    # Check daily
                    await asyncio.sleep(86400)  # 24 hours
                    
                except Exception as e:
                    log_error(e, "system", "rotation_check")
                    await asyncio.sleep(3600)  # Retry in 1 hour
        
        # Start the rotation check task
        asyncio.create_task(check_rotation())
    
    async def rotate_keys(self):
        """Rotate encryption keys and re-encrypt credentials."""
        try:
            # Generate new keys
            new_master_key = self._generate_master_key()
            new_keys = self._derive_keys(new_master_key)
            
            # Re-encrypt all credentials
            for vendor_id in self.list_vendors():
                # Get credentials with old keys
                creds = self.get_credentials(vendor_id)
                if creds:
                    # Store with new keys
                    self.keys = new_keys
                    # Update the credentials object with new data
                    # Inflate credentials to NovaCredential, update, and store
                    cred_obj = NovaCredential.from_dict(creds)
                    # Prepare new credential dict
                    new_creds = cred_obj.to_dict()
                    # Store with new keys
                    self.keys = new_keys
                    self.store_credentials(vendor_id, new_creds)
            
            # Update rotation timestamp
            self.last_rotation = datetime.now()
            self._save_rotation_timestamp()
            
            log_info(
                "Successfully rotated encryption keys",
                "system",
                "key_rotation"
            )
            
        except Exception as e:
            log_error(e, "system", "key_rotation")
            raise
    
    def store_credentials(self, vendor_id: str, credentials: Dict[str, Any]) -> bool:
        """
        Securely store credentials for a vendor.
        
        Args:
            vendor_id: Unique identifier for the vendor
            credentials: Dictionary containing credentials and configuration
            
        Returns:
            bool indicating success
        """
        try:
            # Validate credentials
            if not self._validate_credentials(vendor_id, credentials):
                return False
            
            # Prepare for encryption
            plaintext = json.dumps(credentials).encode()
            
            # Generate a random nonce
            nonce = os.urandom(12)
            
            # Create AESGCM cipher
            cipher = AESGCM(self.keys["encryption"])
            
            # Encrypt credentials
            encrypted_data = cipher.encrypt(
                nonce,
                plaintext,
                self.keys["authentication"]
            )
            
            # Combine nonce and encrypted data
            stored_data = {
                "nonce": base64.b64encode(nonce).decode(),
                "data": base64.b64encode(encrypted_data).decode(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store encrypted data
            file_path = os.path.join(self.storage_path, f"{vendor_id}.enc")
            with open(file_path, "w") as f:
                json.dump(stored_data, f)
            
            log_info(
                f"Stored credentials for vendor: {vendor_id}",
                vendor_id,
                "store_credentials"
            )
            return True
            
        except Exception as e:
            log_error(e, vendor_id, "store_credentials")
            return False
    
    def get_credentials(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve credentials for a vendor.
        
        Args:
            vendor_id: Unique identifier for the vendor
            
        Returns:
            Dictionary containing decrypted credentials or None if not found
        """
        try:
            # Read encrypted data
            file_path = os.path.join(self.storage_path, f"{vendor_id}.enc")
            with open(file_path, "r") as f:
                stored_data = json.load(f)
            
            # Extract components
            nonce = base64.b64decode(stored_data["nonce"])
            encrypted_data = base64.b64decode(stored_data["data"])
            
            # Create AESGCM cipher
            cipher = AESGCM(self.keys["encryption"])
            
            # Decrypt credentials
            plaintext = cipher.decrypt(
                nonce,
                encrypted_data,
                self.keys["authentication"]
            )
            
            credentials = json.loads(plaintext.decode())
            
            log_info(
                f"Retrieved credentials for vendor: {vendor_id}",
                vendor_id,
                "get_credentials"
            )
            return credentials
            
        except FileNotFoundError:
            log_warning(
                f"No credentials found for vendor: {vendor_id}",
                vendor_id,
                "get_credentials"
            )
            return None
        except Exception as e:
            log_error(e, vendor_id, "get_credentials")
            return None
    
    def update_credentials(self, vendor_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update existing credentials for a vendor.
        
        Args:
            vendor_id: Unique identifier for the vendor
            updates: Dictionary containing credential updates
            
        Returns:
            bool indicating success
        """
        try:
            # Get existing credentials
            current_creds = self.get_credentials(vendor_id)
            if not current_creds:
                return False
            
            # Update credentials
            current_creds_obj = NovaCredential.from_dict(current_creds)
            current_creds_obj.update(updates)
            updated_dict = current_creds_obj.to_dict()
            
            # Validate updated credentials
            if not self._validate_credentials(vendor_id, updated_dict):
                return False
            
            # Store updated credentials
            return self.store_credentials(vendor_id, updated_dict)
            
        except Exception as e:
            log_error(e, vendor_id, "update_credentials")
            return False
    
    def delete_credentials(self, vendor_id: str) -> bool:
        """
        Delete credentials for a vendor.
        
        Args:
            vendor_id: Unique identifier for the vendor
            
        Returns:
            bool indicating success
        """
        try:
            file_path = os.path.join(self.storage_path, f"{vendor_id}.enc")
            os.remove(file_path)
            
            log_info(
                f"Deleted credentials for vendor: {vendor_id}",
                vendor_id,
                "delete_credentials"
            )
            return True
            
        except FileNotFoundError:
            log_warning(
                f"No credentials found for vendor: {vendor_id}",
                vendor_id,
                "delete_credentials"
            )
            return False
        except Exception as e:
            log_error(e, vendor_id, "delete_credentials")
            return False
    
    def list_vendors(self) -> List[str]:
        """Get list of vendors with stored credentials."""
        try:
            vendors = []
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".enc"):
                    vendors.append(filename[:-4])  # Remove .enc extension
            return vendors
        except Exception as e:
            log_error(e, "system", "list_vendors")
            return []
    
    def _validate_credentials(self, vendor_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate credential format and required fields."""
        try:
            if vendor_id not in VENDOR_CONFIGS:
                log_warning(
                    f"Unknown vendor: {vendor_id}",
                    vendor_id,
                    "validate_credentials"
                )
                return False
            
            # Check required fields
            required_fields = VENDOR_CONFIGS[vendor_id]["required_fields"]
            for field in required_fields:
                if field not in credentials:
                    log_warning(
                        f"Missing required field: {field}",
                        vendor_id,
                        "validate_credentials"
                    )
                    return False
            
            # Validate password requirements if present
            if "password" in credentials:
                if not self._validate_password(credentials["password"]):
                    log_warning(
                        "Password does not meet requirements",
                        vendor_id,
                        "validate_credentials"
                    )
                    return False
            
            return True
            
        except Exception as e:
            log_error(e, vendor_id, "validate_credentials")
            return False
    
    def _validate_password(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < SECURITY["min_password_length"]:
            return False
        
        if SECURITY["require_special_chars"]:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True