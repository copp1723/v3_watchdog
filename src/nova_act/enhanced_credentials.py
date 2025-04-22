"""
Enhanced credential management for Nova Act integration.

Supports multiple credential storage backends including:
1. AWS Secrets Manager (primary)
2. HashiCorp Vault (optional)
3. Local encrypted file storage (fallback)
"""

import os
import time
import json
import base64
import asyncio
import logging
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .constants import SECURITY
from .logging_config import log_error, log_info, log_warning
from ..utils.aws_secrets import get_secrets_manager_instance

logger = logging.getLogger(__name__)

@dataclass
class CredentialMetadata:
    """Metadata about stored credentials."""
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    source: str = "local"  # 'local', 'aws_secrets', 'vault'
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DealerCredential:
    """Represents a set of vendor credentials for a dealership."""
    dealer_id: str
    vendor_id: str
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    dealer_code: Optional[str] = None
    environment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    active: bool = True
    
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
            "dealer_id": self.dealer_id,
            "vendor_id": self.vendor_id,
            "username": self.username,
            "password": self.password,
            "api_key": self.api_key,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "dealer_code": self.dealer_code,
            "environment": self.environment,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DealerCredential':
        """Create credential from dictionary format."""
        # Convert ISO format strings back to datetime objects
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update credential fields."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
    
    def get_secret_name(self) -> str:
        """Get the standard secret name for this credential."""
        return f"watchdog/credentials/{self.vendor_id}/{self.dealer_id}"


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
    },
    'eleads': {
        'required_fields': ['username', 'password', 'dealer_id'],
        'optional_fields': ['environment', 'client_id'],
        'auth_url': 'https://auth.eleadcrm.com/oauth2/token',
        'base_url': 'https://api.eleadcrm.com/v2'
    }
}


class EnhancedCredentialManager:
    """
    Enhanced credential manager that supports multiple storage backends.
    Primary: AWS Secrets Manager
    Fallback: Local encrypted file storage
    Optional: HashiCorp Vault (if configured)
    """
    
    def __init__(self, encryption_key: Optional[str] = None, use_aws: bool = True, use_vault: bool = False):
        """
        Initialize the enhanced credential manager.
        
        Args:
            encryption_key: Optional encryption key for local storage
            use_aws: Whether to use AWS Secrets Manager
            use_vault: Whether to use HashiCorp Vault
        """
        # Initialize credential cache
        self.credential_cache = {}
        self.metadata_cache = {}
        self.cache_ttl = SECURITY.get("credential_cache_ttl_seconds", 300)  # 5 minutes default
        
        # Initialize storage backends
        self.use_aws = use_aws
        self.use_vault = use_vault
        
        # Get AWS Secrets Manager instance if enabled
        self.aws_secrets = get_secrets_manager_instance() if use_aws else None
        
        # Initialize HashiCorp Vault if enabled
        self.vault_client = self._initialize_vault() if use_vault else None
        
        # Initialize local storage (always as fallback)
        self._initialize_local_storage(encryption_key)
        
        # Log available backends
        backends = []
        if self.aws_secrets:
            backends.append("AWS Secrets Manager")
        if self.vault_client:
            backends.append("HashiCorp Vault")
        backends.append("Local Storage")
        
        log_info(
            f"Credential manager initialized with backends: {', '.join(backends)}",
            "system", 
            "credential_init"
        )
    
    def _initialize_vault(self):
        """Initialize HashiCorp Vault client if configured."""
        try:
            import hvac
            vault_url = os.environ.get("VAULT_ADDR")
            vault_token = os.environ.get("VAULT_TOKEN")
            
            if not vault_url or not vault_token:
                log_warning(
                    "HashiCorp Vault configured but missing VAULT_ADDR or VAULT_TOKEN",
                    "system",
                    "vault_init"
                )
                return None
            
            client = hvac.Client(url=vault_url, token=vault_token)
            if client.is_authenticated():
                log_info(
                    "HashiCorp Vault connection established",
                    "system",
                    "vault_init"
                )
                return client
            else:
                log_warning(
                    "Failed to authenticate with HashiCorp Vault",
                    "system",
                    "vault_init"
                )
                return None
        except ImportError:
            log_warning(
                "hvac package not installed, HashiCorp Vault support disabled",
                "system",
                "vault_init"
            )
            return None
        except Exception as e:
            log_error(
                e,
                "system",
                "vault_init"
            )
            return None
    
    def _initialize_local_storage(self, encryption_key: Optional[str] = None):
        """Initialize local storage for credentials."""
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
            "encryption": Fernet.generate_key(),
            "authentication": derived_key[:16],
            "metadata": derived_key[16:]
        }
        
        return keys
    
    def get_credential(self, dealer_id: str, vendor_id: str) -> Optional[DealerCredential]:
        """
        Get credentials for a dealer and vendor from the best available source.
        
        Args:
            dealer_id: Dealer identifier
            vendor_id: Vendor identifier
            
        Returns:
            DealerCredential object or None if not found
        """
        cache_key = f"{dealer_id}:{vendor_id}"
        
        # Check cache first
        if cache_key in self.credential_cache:
            # Check if cache entry is still valid
            cache_entry = self.credential_cache[cache_key]
            if datetime.now() < cache_entry["expiry"]:
                # Update metadata
                if cache_key in self.metadata_cache:
                    self.metadata_cache[cache_key].access_count += 1
                    self.metadata_cache[cache_key].last_accessed = datetime.now()
                
                log_info(
                    f"Retrieved cached credentials for dealer {dealer_id}, vendor {vendor_id}",
                    dealer_id,
                    "get_credential"
                )
                return cache_entry["credential"]
        
        # Try AWS Secrets Manager first
        result = None
        source = None
        
        if self.aws_secrets:
            try:
                # Construct the secret name
                secret_name = f"watchdog/credentials/{vendor_id}/{dealer_id}"
                secret_dict = self.aws_secrets.get_secret(secret_name)
                
                if secret_dict:
                    result = DealerCredential.from_dict(secret_dict)
                    source = "aws_secrets"
                    log_info(
                        f"Retrieved credentials from AWS Secrets Manager for dealer {dealer_id}, vendor {vendor_id}",
                        dealer_id,
                        "get_credential"
                    )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "aws_get_credential"
                )
        
        # Try HashiCorp Vault if AWS failed and Vault is enabled
        if result is None and self.vault_client:
            try:
                # Construct the vault path
                vault_path = f"secret/data/watchdog/credentials/{vendor_id}/{dealer_id}"
                response = self.vault_client.secrets.kv.v2.read_secret_version(path=vault_path)
                
                if response and "data" in response and "data" in response["data"]:
                    result = DealerCredential.from_dict(response["data"]["data"])
                    source = "vault"
                    log_info(
                        f"Retrieved credentials from HashiCorp Vault for dealer {dealer_id}, vendor {vendor_id}",
                        dealer_id,
                        "get_credential"
                    )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "vault_get_credential"
                )
        
        # Fall back to local storage
        if result is None:
            try:
                local_result = self._get_local_credential(dealer_id, vendor_id)
                if local_result:
                    result = local_result
                    source = "local"
                    log_info(
                        f"Retrieved credentials from local storage for dealer {dealer_id}, vendor {vendor_id}",
                        dealer_id,
                        "get_credential"
                    )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "local_get_credential"
                )
        
        # Cache the result if found
        if result:
            self.credential_cache[cache_key] = {
                "credential": result,
                "expiry": datetime.now() + timedelta(seconds=self.cache_ttl)
            }
            
            # Update metadata
            meta = CredentialMetadata(
                created_at=result.created_at or datetime.now(),
                updated_at=result.updated_at or datetime.now(),
                access_count=1,
                last_accessed=datetime.now(),
                source=source or "unknown"
            )
            self.metadata_cache[cache_key] = meta
            
            return result
        
        log_warning(
            f"No credentials found for dealer {dealer_id}, vendor {vendor_id}",
            dealer_id,
            "get_credential"
        )
        return None
    
    def store_credential(self, credential: DealerCredential) -> bool:
        """
        Store credentials in the best available storage backend.
        
        Args:
            credential: DealerCredential object to store
            
        Returns:
            bool indicating success
        """
        # Update timestamps
        credential.updated_at = datetime.now()
        
        # Validate credentials
        if not self._validate_credential(credential):
            return False
        
        # Prepare credential dictionary
        cred_dict = credential.to_dict()
        dealer_id = credential.dealer_id
        vendor_id = credential.vendor_id
        
        # Try to store in AWS Secrets Manager first
        result = False
        source = None
        
        if self.aws_secrets:
            try:
                # Construct the secret name
                secret_name = f"watchdog/credentials/{vendor_id}/{dealer_id}"
                
                if self.aws_secrets.store_secret(secret_name, cred_dict):
                    result = True
                    source = "aws_secrets"
                    log_info(
                        f"Stored credentials in AWS Secrets Manager for dealer {dealer_id}, vendor {vendor_id}",
                        dealer_id,
                        "store_credential"
                    )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "aws_store_credential"
                )
        
        # Try HashiCorp Vault if AWS failed and Vault is enabled
        if not result and self.vault_client:
            try:
                # Construct the vault path
                vault_path = f"secret/data/watchdog/credentials/{vendor_id}/{dealer_id}"
                
                self.vault_client.secrets.kv.v2.create_or_update_secret(
                    path=vault_path,
                    secret=cred_dict
                )
                
                result = True
                source = "vault"
                log_info(
                    f"Stored credentials in HashiCorp Vault for dealer {dealer_id}, vendor {vendor_id}",
                    dealer_id,
                    "store_credential"
                )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "vault_store_credential"
                )
        
        # Fall back to local storage
        if not result:
            try:
                local_result = self._store_local_credential(credential)
                if local_result:
                    result = True
                    source = "local"
                    log_info(
                        f"Stored credentials in local storage for dealer {dealer_id}, vendor {vendor_id}",
                        dealer_id,
                        "store_credential"
                    )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "local_store_credential"
                )
        
        # Update cache if storage successful
        if result:
            cache_key = f"{dealer_id}:{vendor_id}"
            self.credential_cache[cache_key] = {
                "credential": credential,
                "expiry": datetime.now() + timedelta(seconds=self.cache_ttl)
            }
            
            # Update metadata
            meta = CredentialMetadata(
                created_at=credential.created_at,
                updated_at=credential.updated_at,
                access_count=0,
                source=source or "unknown"
            )
            self.metadata_cache[cache_key] = meta
        
        return result
    
    def delete_credential(self, dealer_id: str, vendor_id: str) -> bool:
        """
        Delete credentials from all storage backends.
        
        Args:
            dealer_id: Dealer identifier
            vendor_id: Vendor identifier
            
        Returns:
            bool indicating success
        """
        cache_key = f"{dealer_id}:{vendor_id}"
        success = False
        
        # Try AWS Secrets Manager
        if self.aws_secrets:
            try:
                # Construct the secret name
                secret_name = f"watchdog/credentials/{vendor_id}/{dealer_id}"
                
                if self.aws_secrets.delete_secret(secret_name):
                    success = True
                    log_info(
                        f"Deleted credentials from AWS Secrets Manager for dealer {dealer_id}, vendor {vendor_id}",
                        dealer_id,
                        "delete_credential"
                    )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "aws_delete_credential"
                )
        
        # Try HashiCorp Vault
        if self.vault_client:
            try:
                # Construct the vault path
                vault_path = f"secret/data/watchdog/credentials/{vendor_id}/{dealer_id}"
                
                self.vault_client.secrets.kv.v2.delete_metadata_and_all_versions(path=vault_path)
                
                success = True
                log_info(
                    f"Deleted credentials from HashiCorp Vault for dealer {dealer_id}, vendor {vendor_id}",
                    dealer_id,
                    "delete_credential"
                )
            except Exception as e:
                log_error(
                    e,
                    dealer_id,
                    "vault_delete_credential"
                )
        
        # Try local storage
        try:
            if self._delete_local_credential(dealer_id, vendor_id):
                success = True
                log_info(
                    f"Deleted credentials from local storage for dealer {dealer_id}, vendor {vendor_id}",
                    dealer_id,
                    "delete_credential"
                )
        except Exception as e:
            log_error(
                e,
                dealer_id,
                "local_delete_credential"
            )
        
        # Clear from cache
        if cache_key in self.credential_cache:
            del self.credential_cache[cache_key]
        
        if cache_key in self.metadata_cache:
            del self.metadata_cache[cache_key]
        
        return success
    
    def list_dealers(self, vendor_id: Optional[str] = None) -> List[str]:
        """
        List all dealer IDs with credentials.
        
        Args:
            vendor_id: Optional vendor ID to filter by
            
        Returns:
            List of dealer IDs
        """
        dealers = set()
        
        # Try AWS Secrets Manager
        if self.aws_secrets:
            try:
                # List secrets with "watchdog/credentials" prefix
                secrets = self.aws_secrets.list_secrets()
                
                for secret in secrets:
                    name = secret.get("name", "")
                    if name.startswith("watchdog/credentials/"):
                        parts = name.split("/")
                        if len(parts) >= 4:
                            secret_vendor_id = parts[2]
                            dealer_id = parts[3]
                            
                            if vendor_id is None or secret_vendor_id == vendor_id:
                                dealers.add(dealer_id)
            except Exception as e:
                log_error(
                    e,
                    "system",
                    "aws_list_dealers"
                )
        
        # Try HashiCorp Vault
        if self.vault_client:
            try:
                # List secrets in the vault path
                if vendor_id:
                    path = f"secret/metadata/watchdog/credentials/{vendor_id}"
                else:
                    path = "secret/metadata/watchdog/credentials"
                
                # List all subpaths
                response = self.vault_client.secrets.kv.v2.list_secrets(path=path)
                
                if response and "data" in response and "keys" in response["data"]:
                    if vendor_id:
                        # Add dealer IDs directly
                        dealers.update(response["data"]["keys"])
                    else:
                        # Need to find dealers under each vendor
                        for vendor in response["data"]["keys"]:
                            sub_path = f"secret/metadata/watchdog/credentials/{vendor}"
                            sub_response = self.vault_client.secrets.kv.v2.list_secrets(path=sub_path)
                            
                            if sub_response and "data" in sub_response and "keys" in sub_response["data"]:
                                dealers.update(sub_response["data"]["keys"])
            except Exception as e:
                log_error(
                    e,
                    "system",
                    "vault_list_dealers"
                )
        
        # Try local storage
        try:
            # Use local file patterns to find credentials
            local_dealers = self._list_local_dealers(vendor_id)
            dealers.update(local_dealers)
        except Exception as e:
            log_error(
                e,
                "system",
                "local_list_dealers"
            )
        
        return sorted(list(dealers))
    
    def list_vendors(self, dealer_id: Optional[str] = None) -> List[str]:
        """
        List all vendor IDs with credentials.
        
        Args:
            dealer_id: Optional dealer ID to filter by
            
        Returns:
            List of vendor IDs
        """
        vendors = set()
        
        # Try AWS Secrets Manager
        if self.aws_secrets:
            try:
                # List secrets with "watchdog/credentials" prefix
                secrets = self.aws_secrets.list_secrets()
                
                for secret in secrets:
                    name = secret.get("name", "")
                    if name.startswith("watchdog/credentials/"):
                        parts = name.split("/")
                        if len(parts) >= 4:
                            vendor_id = parts[2]
                            secret_dealer_id = parts[3]
                            
                            if dealer_id is None or secret_dealer_id == dealer_id:
                                vendors.add(vendor_id)
            except Exception as e:
                log_error(
                    e,
                    dealer_id or "system",
                    "aws_list_vendors"
                )
        
        # Try HashiCorp Vault
        if self.vault_client:
            try:
                if dealer_id:
                    # Need to scan all vendors to filter by dealer_id
                    path = "secret/metadata/watchdog/credentials"
                    response = self.vault_client.secrets.kv.v2.list_secrets(path=path)
                    
                    if response and "data" in response and "keys" in response["data"]:
                        for vendor in response["data"]["keys"]:
                            # Check if this vendor has the dealer_id
                            sub_path = f"secret/metadata/watchdog/credentials/{vendor}"
                            sub_response = self.vault_client.secrets.kv.v2.list_secrets(path=sub_path)
                            
                            if sub_response and "data" in sub_response and "keys" in sub_response["data"]:
                                if dealer_id in sub_response["data"]["keys"]:
                                    vendors.add(vendor)
                else:
                    # List all vendors
                    path = "secret/metadata/watchdog/credentials"
                    response = self.vault_client.secrets.kv.v2.list_secrets(path=path)
                    
                    if response and "data" in response and "keys" in response["data"]:
                        vendors.update(response["data"]["keys"])
            except Exception as e:
                log_error(
                    e,
                    dealer_id or "system",
                    "vault_list_vendors"
                )
        
        # Try local storage
        try:
            # Use local file patterns to find credentials
            local_vendors = self._list_local_vendors(dealer_id)
            vendors.update(local_vendors)
        except Exception as e:
            log_error(
                e,
                dealer_id or "system",
                "local_list_vendors"
            )
        
        return sorted(list(vendors))
    
    def _store_local_credential(self, credential: DealerCredential) -> bool:
        """Store credentials in local encrypted storage."""
        try:
            dealer_id = credential.dealer_id
            vendor_id = credential.vendor_id
            filename = f"{vendor_id}_{dealer_id}.enc"
            file_path = os.path.join(self.storage_path, filename)
            
            # Convert to dict and serialize
            cred_dict = credential.to_dict()
            data = json.dumps(cred_dict).encode()
            
            # Encrypt
            f = Fernet(self.keys["encryption"])
            encrypted_data = f.encrypt(data)
            
            # Write to file
            with open(file_path, "wb") as f:
                f.write(encrypted_data)
            
            return True
        except Exception as e:
            log_error(
                e,
                credential.dealer_id,
                "local_store_credential"
            )
            return False
    
    def _get_local_credential(self, dealer_id: str, vendor_id: str) -> Optional[DealerCredential]:
        """Get credentials from local encrypted storage."""
        try:
            filename = f"{vendor_id}_{dealer_id}.enc"
            file_path = os.path.join(self.storage_path, filename)
            
            if not os.path.exists(file_path):
                return None
            
            # Read encrypted data
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt
            f = Fernet(self.keys["encryption"])
            data = f.decrypt(encrypted_data)
            
            # Parse and convert to DealerCredential
            cred_dict = json.loads(data.decode())
            return DealerCredential.from_dict(cred_dict)
        except Exception as e:
            log_error(
                e,
                dealer_id,
                "local_get_credential"
            )
            return None
    
    def _delete_local_credential(self, dealer_id: str, vendor_id: str) -> bool:
        """Delete credentials from local encrypted storage."""
        try:
            filename = f"{vendor_id}_{dealer_id}.enc"
            file_path = os.path.join(self.storage_path, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            log_error(
                e,
                dealer_id,
                "local_delete_credential"
            )
            return False
    
    def _list_local_dealers(self, vendor_id: Optional[str] = None) -> List[str]:
        """List dealers from local encrypted storage."""
        dealers = set()
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".enc"):
                    parts = filename[:-4].split("_")
                    if len(parts) >= 2:
                        file_vendor = parts[0]
                        file_dealer = parts[1]
                        
                        if vendor_id is None or vendor_id == file_vendor:
                            dealers.add(file_dealer)
        except Exception as e:
            log_error(
                e,
                "system",
                "local_list_dealers"
            )
        return list(dealers)
    
    def _list_local_vendors(self, dealer_id: Optional[str] = None) -> List[str]:
        """List vendors from local encrypted storage."""
        vendors = set()
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".enc"):
                    parts = filename[:-4].split("_")
                    if len(parts) >= 2:
                        file_vendor = parts[0]
                        file_dealer = parts[1]
                        
                        if dealer_id is None or dealer_id == file_dealer:
                            vendors.add(file_vendor)
        except Exception as e:
            log_error(
                e,
                dealer_id or "system",
                "local_list_vendors"
            )
        return list(vendors)
    
    def _validate_credential(self, credential: DealerCredential) -> bool:
        """Validate credential format and required fields."""
        try:
            vendor_id = credential.vendor_id
            
            if vendor_id not in VENDOR_CONFIGS:
                log_warning(
                    f"Unknown vendor: {vendor_id}",
                    credential.dealer_id,
                    "validate_credential"
                )
                return False
            
            # Check required fields
            required_fields = VENDOR_CONFIGS[vendor_id]["required_fields"]
            for field in required_fields:
                value = getattr(credential, field, None)
                if value is None or value == "":
                    log_warning(
                        f"Missing required field: {field}",
                        credential.dealer_id,
                        "validate_credential"
                    )
                    return False
            
            # Validate password requirements if present
            if credential.password:
                if not self._validate_password(credential.password):
                    log_warning(
                        "Password does not meet requirements",
                        credential.dealer_id,
                        "validate_credential"
                    )
                    return False
            
            return True
            
        except Exception as e:
            log_error(
                e,
                credential.dealer_id,
                "validate_credential"
            )
            return False
    
    def _validate_password(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < SECURITY.get("min_password_length", 8):
            return False
        
        if SECURITY.get("require_special_chars", False):
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True

    def get_vendors_for_dealer(self, dealer_id: str) -> List[str]:
        """
        Get all vendors for a specific dealer.
        
        Args:
            dealer_id: Dealer identifier
            
        Returns:
            List of vendor IDs
        """
        return self.list_vendors(dealer_id)
    
    def get_dealers_for_vendor(self, vendor_id: str) -> List[str]:
        """
        Get all dealers for a specific vendor.
        
        Args:
            vendor_id: Vendor identifier
            
        Returns:
            List of dealer IDs
        """
        return self.list_dealers(vendor_id)
    
    def get_credential_metadata(self, dealer_id: str, vendor_id: str) -> Optional[CredentialMetadata]:
        """
        Get metadata for a credential.
        
        Args:
            dealer_id: Dealer identifier
            vendor_id: Vendor identifier
            
        Returns:
            CredentialMetadata or None if not found
        """
        cache_key = f"{dealer_id}:{vendor_id}"
        return self.metadata_cache.get(cache_key)


# Singleton instance
_credential_manager = None

def get_credential_manager() -> EnhancedCredentialManager:
    """
    Get the singleton credential manager instance.
    
    Returns:
        The credential manager instance
    """
    global _credential_manager
    if _credential_manager is None:
        # Check if AWS integration is enabled
        use_aws = os.getenv('USE_AWS_SECRETS', 'false').lower() in ('true', '1', 'yes')
        # Check if Vault integration is enabled
        use_vault = os.getenv('USE_VAULT', 'false').lower() in ('true', '1', 'yes')
        
        _credential_manager = EnhancedCredentialManager(
            use_aws=use_aws,
            use_vault=use_vault
        )
    return _credential_manager