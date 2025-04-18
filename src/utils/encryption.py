"""
Encryption utilities for Watchdog AI.

Provides functions for encrypting and decrypting data at rest.
Uses Fernet symmetric encryption from the cryptography library.
"""

import os
import base64
import logging
from typing import Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import InvalidToken

# Configure logging
logger = logging.getLogger(__name__)

# Default location for the encryption key
DEFAULT_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'data', '.encryption_key')

# Salt for key derivation
SALT = b'watchdog_ai_salt_value'  # In production, this should be stored securely

def _get_or_create_key(key_path: Optional[str] = None) -> bytes:
    """
    Get the encryption key from file or create a new one if it doesn't exist.
    
    Args:
        key_path: Path to the encryption key file. If None, uses the default path.
        
    Returns:
        The encryption key as bytes
    """
    if key_path is None:
        key_path = DEFAULT_KEY_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    
    try:
        # Try to read the key
        if os.path.exists(key_path):
            with open(key_path, 'rb') as key_file:
                key = key_file.read()
                if len(key) == 44:  # Fernet key is 44 bytes when base64 encoded
                    return key
        
        # If key doesn't exist or is invalid, create a new one
        logger.info(f"Creating new encryption key at {key_path}")
        key = Fernet.generate_key()
        
        # Save the key with restricted permissions
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
        
        # Set appropriate permissions on the key file (POSIX systems only)
        try:
            os.chmod(key_path, 0o600)  # Owner read/write only
        except:
            logger.warning("Could not set file permissions on key file. Ensure it's protected.")
        
        return key
        
    except Exception as e:
        logger.error(f"Error handling encryption key: {str(e)}")
        # Fall back to environment variable or derived key if file access fails
        return _derive_key_from_env()


def _derive_key_from_env() -> bytes:
    """
    Derive an encryption key from an environment variable as a fallback.
    
    Returns:
        Derived encryption key
    """
    # Get the secret from environment or use a default (not recommended for production)
    secret = os.environ.get('WATCHDOG_ENCRYPTION_SECRET', 'default_secret_change_me_in_production')
    
    # Derive a key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return key


def get_fernet() -> Fernet:
    """
    Get a configured Fernet instance for encryption/decryption.
    
    Returns:
        Fernet instance
    """
    key = _get_or_create_key()
    return Fernet(key)


def encrypt_bytes(data: bytes) -> bytes:
    """
    Encrypt binary data using Fernet symmetric encryption.
    
    Args:
        data: The data to encrypt
        
    Returns:
        Encrypted data
    """
    if not data:
        return data
    
    try:
        fernet = get_fernet()
        return fernet.encrypt(data)
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        raise


def decrypt_bytes(encrypted_data: bytes) -> bytes:
    """
    Decrypt binary data using Fernet symmetric encryption.
    
    Args:
        encrypted_data: The data to decrypt
        
    Returns:
        Decrypted data
        
    Raises:
        ValueError if decryption fails
    """
    if not encrypted_data:
        return encrypted_data
    
    try:
        fernet = get_fernet()
        return fernet.decrypt(encrypted_data)
    except InvalidToken:
        logger.error("Invalid token in decryption - the data may be corrupted or tampered with")
        raise ValueError("Could not decrypt data: invalid encryption token")
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        raise


def encrypt_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Encrypt a file on disk.
    
    Args:
        input_path: Path to the file to encrypt
        output_path: Path to save the encrypted file. If None, uses input_path + '.enc'
        
    Returns:
        Path to the encrypted file
        
    Raises:
        FileNotFoundError if input file doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        output_path = input_path + '.enc'
    
    try:
        # Read the input file
        with open(input_path, 'rb') as f:
            data = f.read()
        
        # Encrypt the data
        encrypted_data = encrypt_bytes(data)
        
        # Write the encrypted data
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Encrypted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error encrypting file {input_path}: {str(e)}")
        raise


def decrypt_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Decrypt a file on disk.
    
    Args:
        input_path: Path to the encrypted file
        output_path: Path to save the decrypted file. If None, removes '.enc' suffix if present.
        
    Returns:
        Path to the decrypted file
        
    Raises:
        FileNotFoundError if input file doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        # Remove .enc suffix if present
        if input_path.endswith('.enc'):
            output_path = input_path[:-4]
        else:
            output_path = input_path + '.dec'
    
    try:
        # Read the encrypted input file
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt the data
        decrypted_data = decrypt_bytes(encrypted_data)
        
        # Write the decrypted data
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        logger.info(f"Decrypted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error decrypting file {input_path}: {str(e)}")
        raise


# Functions for working with pandas DataFrames
def read_encrypted_csv(file_path: str, **kwargs) -> 'pd.DataFrame':
    """
    Read an encrypted CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the encrypted CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame containing the decrypted data
    """
    import pandas as pd
    import io
    
    try:
        # Read and decrypt the file
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = decrypt_bytes(encrypted_data)
        
        # Convert to DataFrame
        return pd.read_csv(io.BytesIO(decrypted_data), **kwargs)
        
    except Exception as e:
        logger.error(f"Error reading encrypted CSV {file_path}: {str(e)}")
        raise


def save_encrypted_csv(df: 'pd.DataFrame', file_path: str, **kwargs) -> None:
    """
    Save a pandas DataFrame to an encrypted CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the encrypted CSV
        **kwargs: Additional arguments to pass to df.to_csv
    """
    import io
    
    try:
        # Convert DataFrame to CSV in memory
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, **kwargs)
        csv_data = csv_buffer.getvalue()
        
        # Encrypt and save
        encrypted_data = encrypt_bytes(csv_data)
        
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Saved encrypted DataFrame to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving encrypted DataFrame to {file_path}: {str(e)}")
        raise