"""
AWS Secrets Manager integration for secure credential storage.

This module provides functionality to store and retrieve credentials 
securely using AWS Secrets Manager.
"""

import json
import base64
import boto3
import botocore
import logging
import os
from typing import Dict, Any, Optional, List, Union
from botocore.exceptions import ClientError
from datetime import datetime

from .log_utils_config import get_logger

logger = get_logger(__name__)

class SecretsManager:
    """Manages secure storage and retrieval of credentials using AWS Secrets Manager."""
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize the Secrets Manager client.
        
        Args:
            region_name: AWS region name. If None, uses environment variable or default.
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the AWS Secrets Manager client."""
        try:
            self.client = boto3.client(
                service_name='secretsmanager',
                region_name=self.region_name
            )
            logger.info(f"AWS Secrets Manager client initialized in region {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Secrets Manager client: {str(e)}")
            raise
    
    def store_secret(self, secret_name: str, secret_value: Dict[str, Any]) -> bool:
        """
        Store a secret in AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret
            secret_value: Dictionary containing secret values
            
        Returns:
            bool indicating success
        """
        try:
            # Check if secret already exists
            try:
                self.client.describe_secret(SecretId=secret_name)
                # Secret exists, update it
                self.client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(secret_value)
                )
                logger.info(f"Updated secret: {secret_name}")
            except self.client.exceptions.ResourceNotFoundException:
                # Secret doesn't exist, create it
                self.client.create_secret(
                    Name=secret_name,
                    SecretString=json.dumps(secret_value),
                    Tags=[
                        {
                            'Key': 'Application',
                            'Value': 'WatchdogAI'
                        },
                        {
                            'Key': 'Environment',
                            'Value': os.getenv('ENVIRONMENT', 'development')
                        }
                    ]
                )
                logger.info(f"Created new secret: {secret_name}")
            
            return True
            
        except ClientError as e:
            logger.error(f"Error storing secret {secret_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing secret {secret_name}: {str(e)}")
            return False
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Dictionary containing secret values or None if not found
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            
            # Parse and return the secret JSON
            if 'SecretString' in response:
                secret = json.loads(response['SecretString'])
                logger.info(f"Retrieved secret: {secret_name}")
                return secret
            elif 'SecretBinary' in response:
                # For binary secrets
                secret = base64.b64decode(response['SecretBinary'])
                logger.info(f"Retrieved binary secret: {secret_name}")
                return json.loads(secret)
            
            logger.warning(f"Secret {secret_name} found but has no value")
            return None
            
        except self.client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret {secret_name} not found")
            return None
        except ClientError as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret {secret_name}: {str(e)}")
            return None
    
    def delete_secret(self, secret_name: str, force_delete: bool = False) -> bool:
        """
        Delete a secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret
            force_delete: Whether to force deletion without recovery
            
        Returns:
            bool indicating success
        """
        try:
            if force_delete:
                self.client.delete_secret(
                    SecretId=secret_name,
                    ForceDeleteWithoutRecovery=True
                )
            else:
                self.client.delete_secret(
                    SecretId=secret_name,
                    RecoveryWindowInDays=7
                )
            
            logger.info(f"Deleted secret: {secret_name} (force_delete={force_delete})")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting secret {secret_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting secret {secret_name}: {str(e)}")
            return False
    
    def list_secrets(self, filter_tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List secrets in AWS Secrets Manager.
        
        Args:
            filter_tags: Optional dictionary of tags to filter by
            
        Returns:
            List of secrets metadata
        """
        try:
            # Initial empty list for secrets
            all_secrets = []
            
            # If filter_tags provided, create filters
            filters = []
            if filter_tags:
                for key, value in filter_tags.items():
                    filters.append({
                        'Key': 'tag-key',
                        'Values': [key]
                    })
                    filters.append({
                        'Key': 'tag-value',
                        'Values': [value]
                    })
            
            # Build the list_secrets parameters
            list_params = {}
            if filters:
                list_params['Filters'] = filters
            
            # Handle pagination in the AWS response
            paginator = self.client.get_paginator('list_secrets')
            response_iterator = paginator.paginate(**list_params)
            
            for page in response_iterator:
                all_secrets.extend(page.get('SecretList', []))
            
            # Extract relevant metadata
            result = []
            for secret in all_secrets:
                # Filter to application secrets if tags filter not provided
                if not filter_tags and not any(
                    tag.get('Key') == 'Application' and tag.get('Value') == 'WatchdogAI' 
                    for tag in secret.get('Tags', [])
                ):
                    continue
                
                result.append({
                    'name': secret.get('Name'),
                    'description': secret.get('Description', ''),
                    'last_updated': secret.get('LastChangedDate'),
                    'tags': secret.get('Tags', [])
                })
            
            logger.info(f"Listed {len(result)} secrets")
            return result
            
        except ClientError as e:
            logger.error(f"Error listing secrets: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing secrets: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if AWS Secrets Manager is available and properly configured.
        
        Returns:
            bool indicating if Secrets Manager is available
        """
        try:
            # Call list_secrets with a limit of 1 to check connectivity
            self.client.list_secrets(MaxResults=1)
            return True
        except Exception as e:
            logger.warning(f"AWS Secrets Manager not available: {str(e)}")
            return False
    
    def get_connection_details(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get connection details for a specific vendor.
        
        Args:
            vendor_id: Vendor identifier
            
        Returns:
            Dictionary with connection details or None if not found
        """
        secret_name = f"watchdog/credentials/{vendor_id}"
        return self.get_secret(secret_name)


# Create a fallback local implementation that uses files instead of AWS
class LocalSecretsManager:
    """Local filesystem-based implementation of secrets manager for development."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the local secrets manager.
        
        Args:
            storage_path: Directory to store secrets. If None, uses default.
        """
        from .encryption import get_fernet
        
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "secrets"
        )
        self.fernet = get_fernet()
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Create a metadata file for tracking all secrets
        self.metadata_path = os.path.join(self.storage_path, "metadata.json")
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w") as f:
                json.dump({"secrets": []}, f)
    
    def _secret_path(self, secret_name: str) -> str:
        """Get the file path for a secret."""
        # Replace special characters that might be problematic in filenames
        safe_name = secret_name.replace("/", "__")
        return os.path.join(self.storage_path, f"{safe_name}.enc")
    
    def store_secret(self, secret_name: str, secret_value: Dict[str, Any]) -> bool:
        """
        Store a secret in the local filesystem.
        
        Args:
            secret_name: Name of the secret
            secret_value: Dictionary containing secret values
            
        Returns:
            bool indicating success
        """
        try:
            # Encrypt the secret
            secret_data = json.dumps(secret_value).encode()
            encrypted_data = self.fernet.encrypt(secret_data)
            
            # Write to file
            with open(self._secret_path(secret_name), "wb") as f:
                f.write(encrypted_data)
            
            # Update metadata
            self._update_metadata(secret_name)
            
            logger.info(f"Stored local secret: {secret_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing local secret {secret_name}: {str(e)}")
            return False
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret from the local filesystem.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Dictionary containing secret values or None if not found
        """
        try:
            secret_path = self._secret_path(secret_name)
            if not os.path.exists(secret_path):
                logger.warning(f"Local secret {secret_name} not found")
                return None
            
            # Read and decrypt
            with open(secret_path, "rb") as f:
                encrypted_data = f.read()
            
            secret_data = self.fernet.decrypt(encrypted_data)
            secret = json.loads(secret_data.decode())
            
            logger.info(f"Retrieved local secret: {secret_name}")
            return secret
            
        except Exception as e:
            logger.error(f"Error retrieving local secret {secret_name}: {str(e)}")
            return None
    
    def delete_secret(self, secret_name: str, force_delete: bool = False) -> bool:
        """
        Delete a secret from the local filesystem.
        
        Args:
            secret_name: Name of the secret
            force_delete: Not used in local implementation
            
        Returns:
            bool indicating success
        """
        try:
            secret_path = self._secret_path(secret_name)
            if os.path.exists(secret_path):
                os.remove(secret_path)
                
                # Update metadata
                self._remove_from_metadata(secret_name)
                
                logger.info(f"Deleted local secret: {secret_name}")
                return True
            else:
                logger.warning(f"Local secret {secret_name} not found for deletion")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting local secret {secret_name}: {str(e)}")
            return False
    
    def list_secrets(self, filter_tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List secrets in the local filesystem.
        
        Args:
            filter_tags: Optional dictionary of tags to filter by (not used in local implementation)
            
        Returns:
            List of secrets metadata
        """
        try:
            # Read metadata file
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Return all secrets
            return metadata.get("secrets", [])
            
        except Exception as e:
            logger.error(f"Error listing local secrets: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if local secrets storage is available.
        
        Returns:
            bool indicating if local storage is available
        """
        return os.path.exists(self.storage_path) and os.access(self.storage_path, os.W_OK)
    
    def get_connection_details(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get connection details for a specific vendor.
        
        Args:
            vendor_id: Vendor identifier
            
        Returns:
            Dictionary with connection details or None if not found
        """
        secret_name = f"watchdog/credentials/{vendor_id}"
        return self.get_secret(secret_name)
    
    def _update_metadata(self, secret_name: str):
        """Update metadata when storing a secret."""
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Check if secret already exists in metadata
            for secret in metadata.get("secrets", []):
                if secret.get("name") == secret_name:
                    # Update timestamp
                    secret["last_updated"] = datetime.now().isoformat()
                    break
            else:
                # Secret not found, add it
                metadata.setdefault("secrets", []).append({
                    "name": secret_name,
                    "description": "",
                    "last_updated": datetime.now().isoformat(),
                    "tags": [
                        {"Key": "Application", "Value": "WatchdogAI"},
                        {"Key": "Environment", "Value": os.getenv('ENVIRONMENT', 'development')}
                    ]
                })
            
            # Write updated metadata
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating metadata for {secret_name}: {str(e)}")
    
    def _remove_from_metadata(self, secret_name: str):
        """Remove secret from metadata when deleting."""
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Filter out the deleted secret
            metadata["secrets"] = [
                secret for secret in metadata.get("secrets", [])
                if secret.get("name") != secret_name
            ]
            
            # Write updated metadata
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating metadata after deleting {secret_name}: {str(e)}")


def get_secrets_manager() -> Union[SecretsManager, LocalSecretsManager]:
    """
    Factory function to get the appropriate secrets manager based on environment.
    
    Returns:
        A secrets manager instance (AWS or local)
    """
    # Check if we're in AWS environment
    aws_enabled = os.getenv('USE_AWS_SECRETS', 'false').lower() in ('true', '1', 'yes')
    
    if aws_enabled:
        # Try to create and verify AWS Secrets Manager
        try:
            aws_sm = SecretsManager()
            if aws_sm.is_available():
                logger.info("Using AWS Secrets Manager")
                return aws_sm
            else:
                logger.warning("AWS Secrets Manager not available, falling back to local storage")
        except Exception as e:
            logger.warning(f"Error initializing AWS Secrets Manager, falling back to local storage: {str(e)}")
    
    # Fall back to local implementation
    logger.info("Using Local Secrets Manager")
    return LocalSecretsManager()


# Singleton instance
_secrets_manager_instance = None

def get_secrets_manager_instance() -> Union[SecretsManager, LocalSecretsManager]:
    """
    Get the singleton secrets manager instance.
    
    Returns:
        The secrets manager instance
    """
    global _secrets_manager_instance
    if _secrets_manager_instance is None:
        _secrets_manager_instance = get_secrets_manager()
    return _secrets_manager_instance