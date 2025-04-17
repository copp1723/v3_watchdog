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

from .constants import SECURITY
from .logging_config import log_error, log_info, log_warning

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
        
        # Derive encryption keys
        self.keys = self._derive_keys(
            master_key if isinstance(master_key, bytes) else master_key.encode()
        )
        
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
                    self.store_credentials(vendor_id, creds)
            
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
            current_creds.update(updates)
            
            # Validate updated credentials
            if not self._validate_credentials(vendor_id, current_creds):
                return False
            
            # Store updated credentials
            return self.store_credentials(vendor_id, current_creds)
            
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

class NovaCredential:
    """
    Secure credential management for Nova Act systems.
    Handles encryption, storage, and retrieval of sensitive credentials.
    """
    
    def __init__(self, 
                storage_path: str = "config/credentials.enc", 
                master_key: Optional[str] = None,
                auto_save: bool = True):
        """
        Initialize the credential manager.
        
        Args:
            storage_path: Path to store encrypted credentials
            master_key: Master key for encryption (if None, will check environment)
            auto_save: Whether to automatically save credentials after changes
        """
        self.logger = logging.getLogger("NovaCredential")
        self.storage_path = storage_path
        self.auto_save = auto_save
        self._credentials = {}
        self._metadata = {
            "last_modified": datetime.now().isoformat(),
            "credential_count": 0,
            "version": "1.0"
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Initialize encryption key
        self._initialize_encryption(master_key)
        
        # Load credentials if file exists
        if os.path.exists(storage_path):
            self.load()
    
    def _initialize_encryption(self, master_key: Optional[str] = None):
        """Initialize encryption with the master key."""
        # Try to get key from parameter, environment, or generate a new one
        if master_key:
            self.master_key = master_key
        elif "NOVA_MASTER_KEY" in os.environ:
            self.master_key = os.environ["NOVA_MASTER_KEY"]
        else:
            # For development only - in production, a key should be provided
            self.logger.warning("No master key provided. Generating a temporary one.")
            self.master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        # Derive a key from the master key
        salt = b'NovaCredentialSalt'  # In production, this should be unique and stored
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self.cipher = Fernet(key)
    
    def add_credential(self, 
                     system_id: str, 
                     username: str, 
                     password: str, 
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Add or update credentials for a system.
        
        Args:
            system_id: Identifier for the system (e.g., 'salesforce', 'aws')
            username: Username for the system
            password: Password for the system
            metadata: Additional metadata about this credential
            
        Returns:
            bool: True if successful
        """
        # Create credential entry
        credential = {
            "username": username,
            "password": password,
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "last_used": None,
            "metadata": metadata or {}
        }
        
        # Store in the credentials dictionary
        self._credentials[system_id] = credential
        
        # Update metadata
        self._metadata["last_modified"] = datetime.now().isoformat()
        self._metadata["credential_count"] = len(self._credentials)
        
        # Save if auto_save is enabled
        if self.auto_save:
            return self.save()
        
        return True
    
    def get_credential(self, system_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve credentials for a system.
        
        Args:
            system_id: Identifier for the system
            
        Returns:
            Dict containing the credentials or None if not found
        """
        if system_id not in self._credentials:
            return None
        
        # Update last used timestamp
        self._credentials[system_id]["last_used"] = datetime.now().isoformat()
        
        # Return a copy to prevent accidental modification
        credential = dict(self._credentials[system_id])
        
        return credential
    
    def delete_credential(self, system_id: str) -> bool:
        """
        Delete credentials for a system.
        
        Args:
            system_id: Identifier for the system
            
        Returns:
            bool: True if deleted, False if not found
        """
        if system_id not in self._credentials:
            return False
        
        # Remove from dictionary
        del self._credentials[system_id]
        
        # Update metadata
        self._metadata["last_modified"] = datetime.now().isoformat()
        self._metadata["credential_count"] = len(self._credentials)
        
        # Save if auto_save is enabled
        if self.auto_save:
            return self.save()
        
        return True
    
    def list_credentials(self) -> List[Dict[str, Any]]:
        """
        List all stored credentials (without passwords).
        
        Returns:
            List of credential info dictionaries
        """
        result = []
        
        for system_id, credential in self._credentials.items():
            # Create a sanitized version without the password
            sanitized = {
                "system_id": system_id,
                "username": credential["username"],
                "created": credential["created"],
                "last_modified": credential["last_modified"],
                "last_used": credential["last_used"],
                "metadata": credential["metadata"]
            }
            result.append(sanitized)
        
        return result
    
    def save(self) -> bool:
        """
        Save credentials to encrypted storage.
        
        Returns:
            bool: True if successful
        """
        try:
            # Prepare data for encryption
            data = {
                "metadata": self._metadata,
                "credentials": self._credentials
            }
            
            # Convert to JSON and encrypt
            json_data = json.dumps(data)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            # Write to file
            with open(self.storage_path, "wb") as f:
                f.write(encrypted_data)
            
            self.logger.info(f"Saved {len(self._credentials)} credentials to {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save credentials: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        Load credentials from encrypted storage.
        
        Returns:
            bool: True if successful
        """
        try:
            # Check if file exists
            if not os.path.exists(self.storage_path):
                self.logger.warning(f"Credentials file not found: {self.storage_path}")
                return False
            
            # Read and decrypt
            with open(self.storage_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt and parse JSON
            json_data = self.cipher.decrypt(encrypted_data).decode()
            data = json.loads(json_data)
            
            # Extract data
            self._metadata = data.get("metadata", {})
            self._credentials = data.get("credentials", {})
            
            self.logger.info(f"Loaded {len(self._credentials)} credentials from {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {str(e)}")
            return False
    
    def verify_credential(self, system_id: str, test_func: callable = None) -> Dict[str, Any]:
        """
        Verify if a credential is valid by testing it.
        
        Args:
            system_id: Identifier for the system
            test_func: Function that takes (username, password) and returns (success, message)
                If None, just checks if credential exists
                
        Returns:
            Dict with verification results
        """
        result = {
            "system_id": system_id,
            "exists": False,
            "verified": False,
            "message": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if credential exists
        credential = self.get_credential(system_id)
        if not credential:
            result["message"] = f"No credentials found for {system_id}"
            return result
        
        result["exists"] = True
        
        # If no test function, just return existence
        if not test_func:
            result["message"] = "Credential exists (no verification performed)"
            return result
        
        # Test the credential
        try:
            success, message = test_func(credential["username"], credential["password"])
            result["verified"] = success
            result["message"] = message
            
            # Update metadata on the credential
            if success:
                self._credentials[system_id]["metadata"]["last_verified"] = datetime.now().isoformat()
                self._credentials[system_id]["metadata"]["verification_status"] = "success"
            else:
                self._credentials[system_id]["metadata"]["last_verification_attempt"] = datetime.now().isoformat()
                self._credentials[system_id]["metadata"]["verification_status"] = "failed"
                self._credentials[system_id]["metadata"]["last_error"] = message
            
            # Save changes if auto_save
            if self.auto_save:
                self.save()
                
        except Exception as e:
            result["verified"] = False
            result["message"] = f"Verification error: {str(e)}"
            
            # Update metadata
            self._credentials[system_id]["metadata"]["last_verification_attempt"] = datetime.now().isoformat()
            self._credentials[system_id]["metadata"]["verification_status"] = "error"
            self._credentials[system_id]["metadata"]["last_error"] = str(e)
            
            if self.auto_save:
                self.save()
        
        return result
    
    def update_credential_metadata(self, system_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for a credential.
        
        Args:
            system_id: Identifier for the system
            metadata_updates: Dictionary of metadata to update
            
        Returns:
            bool: True if successful
        """
        if system_id not in self._credentials:
            return False
        
        # Update metadata
        self._credentials[system_id]["metadata"].update(metadata_updates)
        self._credentials[system_id]["last_modified"] = datetime.now().isoformat()
        
        # Update global metadata
        self._metadata["last_modified"] = datetime.now().isoformat()
        
        # Save if auto_save is enabled
        if self.auto_save:
            return self.save()
        
        return True
    
    def export_credentials(self, password: str, output_path: str) -> bool:
        """
        Export credentials to a portable encrypted file with a specific password.
        Useful for transferring credentials between environments.
        
        Args:
            password: Password to protect the export
            output_path: Path to save the export file
            
        Returns:
            bool: True if successful
        """
        try:
            # Create key from password
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            cipher = Fernet(key)
            
            # Prepare data for export
            export_data = {
                "metadata": self._metadata,
                "credentials": self._credentials,
                "export_date": datetime.now().isoformat(),
                "export_version": "1.0"
            }
            
            # Encrypt the data
            json_data = json.dumps(export_data)
            encrypted_data = cipher.encrypt(json_data.encode())
            
            # Combine salt and encrypted data
            export_bytes = salt + encrypted_data
            
            # Write to file
            with open(output_path, "wb") as f:
                f.write(export_bytes)
            
            self.logger.info(f"Exported {len(self._credentials)} credentials to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export credentials: {str(e)}")
            return False
    
    def import_credentials(self, password: str, import_path: str, 
                         overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Import credentials from an exported file.
        
        Args:
            password: Password used to protect the export
            import_path: Path to the export file
            overwrite_existing: Whether to overwrite existing credentials
            
        Returns:
            Dict with import results
        """
        result = {
            "success": False,
            "message": "",
            "imported_count": 0,
            "skipped_count": 0,
            "error_count": 0
        }
        
        try:
            # Read the export file
            with open(import_path, "rb") as f:
                export_bytes = f.read()
            
            # Extract salt and encrypted data
            salt = export_bytes[:16]
            encrypted_data = export_bytes[16:]
            
            # Create key from password and salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            cipher = Fernet(key)
            
            # Decrypt and parse
            try:
                json_data = cipher.decrypt(encrypted_data).decode()
                import_data = json.loads(json_data)
            except Exception:
                result["message"] = "Invalid password or corrupted export file"
                return result
            
            # Process import
            imported_credentials = import_data.get("credentials", {})
            
            for system_id, credential in imported_credentials.items():
                try:
                    if system_id in self._credentials and not overwrite_existing:
                        result["skipped_count"] += 1
                        continue
                    
                    # Add the credential
                    self._credentials[system_id] = credential
                    result["imported_count"] += 1
                    
                except Exception:
                    result["error_count"] += 1
            
            # Update metadata
            self._metadata["last_modified"] = datetime.now().isoformat()
            self._metadata["credential_count"] = len(self._credentials)
            
            # Save changes
            if self.auto_save:
                self.save()
            
            result["success"] = True
            result["message"] = f"Imported {result['imported_count']} credentials"
            
            self.logger.info(f"Imported {result['imported_count']} credentials from {import_path}")
            return result
            
        except Exception as e:
            result["message"] = f"Import failed: {str(e)}"
            self.logger.error(f"Failed to import credentials: {str(e)}")
            return result
    
    def get_credential_hash(self, system_id: str) -> Optional[str]:
        """
        Get a secure hash of the credential for comparison without exposing the secret.
        
        Args:
            system_id: Identifier for the system
            
        Returns:
            str: Hash of the credential or None if not found
        """
        credential = self.get_credential(system_id)
        if not credential:
            return None
        
        # Create a hash of username + password
        hash_input = f"{credential['username']}:{credential['password']}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return hash_value
    
    def rotate_encryption_key(self, new_master_key: str) -> bool:
        """
        Rotate the encryption key for enhanced security.
        
        Args:
            new_master_key: New master key to use for encryption
            
        Returns:
            bool: True if successful
        """
        try:
            # First, save current state with current key
            self.save()
            
            # Store current data
            current_credentials = self._credentials
            current_metadata = self._metadata
            
            # Initialize with new key
            self._initialize_encryption(new_master_key)
            
            # Restore data
            self._credentials = current_credentials
            self._metadata = current_metadata
            self._metadata["last_modified"] = datetime.now().isoformat()
            
            # Save with new key
            success = self.save()
            
            if success:
                self.logger.info("Encryption key rotated successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to rotate encryption key: {str(e)}")
            return False