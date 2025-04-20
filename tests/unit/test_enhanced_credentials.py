"""
Tests for the enhanced credential manager.
"""

import os
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.nova_act.enhanced_credentials import (
    EnhancedCredentialManager,
    DealerCredential,
    CredentialMetadata
)

# Fixture for a credential manager with temporary storage
@pytest.fixture
def cred_manager(tmp_path):
    """Create a credential manager with temporary storage."""
    storage_path = os.path.join(tmp_path, "storage", "credentials")
    metadata_path = os.path.join(tmp_path, "storage", "metadata")
    
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)
    
    # Initialize manager with test encryption key
    manager = EnhancedCredentialManager(
        encryption_key="test_key_for_unit_tests_only",
        use_aws=False,
        use_vault=False
    )
    
    # Override storage paths
    manager.storage_path = storage_path
    manager.metadata_path = metadata_path
    
    return manager

# Fixture for sample dealer credentials
@pytest.fixture
def sample_credentials():
    """Create sample dealer credentials."""
    return DealerCredential(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        client_id="test_client",
        client_secret="test_secret",
        dealer_code="TEST123",
        environment="production"
    )

def test_store_get_credential(cred_manager, sample_credentials):
    """Test storing and retrieving credentials."""
    # Store the credentials
    result = cred_manager._store_local_credential(sample_credentials)
    assert result is True
    
    # Retrieve the credentials
    retrieved = cred_manager._get_local_credential("test_dealer", "dealersocket")
    
    # Verify the retrieved credentials
    assert retrieved is not None
    assert retrieved.dealer_id == "test_dealer"
    assert retrieved.vendor_id == "dealersocket"
    assert retrieved.client_id == "test_client"
    assert retrieved.client_secret == "test_secret"
    assert retrieved.dealer_code == "TEST123"
    assert retrieved.environment == "production"

def test_update_credential(cred_manager, sample_credentials):
    """Test updating credentials."""
    # Store the credentials
    cred_manager._store_local_credential(sample_credentials)
    
    # Update some fields
    sample_credentials.environment = "test"
    sample_credentials.metadata = {"note": "Updated for testing"}
    
    # Store the updated credentials
    result = cred_manager._store_local_credential(sample_credentials)
    assert result is True
    
    # Retrieve the updated credentials
    retrieved = cred_manager._get_local_credential("test_dealer", "dealersocket")
    
    # Verify the updates
    assert retrieved.environment == "test"
    assert retrieved.metadata == {"note": "Updated for testing"}
    assert retrieved.updated_at > retrieved.created_at

def test_delete_credential(cred_manager, sample_credentials):
    """Test deleting credentials."""
    # Store the credentials
    cred_manager._store_local_credential(sample_credentials)
    
    # Delete the credentials
    result = cred_manager._delete_local_credential("test_dealer", "dealersocket")
    assert result is True
    
    # Verify credentials were deleted
    retrieved = cred_manager._get_local_credential("test_dealer", "dealersocket")
    assert retrieved is None

def test_list_dealers(cred_manager):
    """Test listing dealers."""
    # Store multiple credentials
    cred_manager._store_local_credential(DealerCredential(
        dealer_id="dealer1",
        vendor_id="dealersocket",
        client_id="client1",
        client_secret="secret1",
        dealer_code="D1"
    ))
    
    cred_manager._store_local_credential(DealerCredential(
        dealer_id="dealer2",
        vendor_id="dealersocket",
        client_id="client2",
        client_secret="secret2",
        dealer_code="D2"
    ))
    
    cred_manager._store_local_credential(DealerCredential(
        dealer_id="dealer1",
        vendor_id="vinsolutions",
        username="user1",
        password="pass1",
        api_key="key1"
    ))
    
    # List all dealers
    dealers = cred_manager._list_local_dealers()
    
    # Verify dealers list
    assert len(dealers) == 2
    assert "dealer1" in dealers
    assert "dealer2" in dealers
    
    # List dealers for specific vendor
    dealers_ds = cred_manager._list_local_dealers("dealersocket")
    
    # Verify filtered list
    assert len(dealers_ds) == 2
    assert "dealer1" in dealers_ds
    assert "dealer2" in dealers_ds
    
    dealers_vs = cred_manager._list_local_dealers("vinsolutions")
    
    # Verify filtered list
    assert len(dealers_vs) == 1
    assert "dealer1" in dealers_vs

def test_list_vendors(cred_manager):
    """Test listing vendors."""
    # Store multiple credentials
    cred_manager._store_local_credential(DealerCredential(
        dealer_id="dealer1",
        vendor_id="dealersocket",
        client_id="client1",
        client_secret="secret1",
        dealer_code="D1"
    ))
    
    cred_manager._store_local_credential(DealerCredential(
        dealer_id="dealer2",
        vendor_id="dealersocket",
        client_id="client2",
        client_secret="secret2",
        dealer_code="D2"
    ))
    
    cred_manager._store_local_credential(DealerCredential(
        dealer_id="dealer1",
        vendor_id="vinsolutions",
        username="user1",
        password="pass1",
        api_key="key1"
    ))
    
    # List all vendors
    vendors = cred_manager._list_local_vendors()
    
    # Verify vendors list
    assert len(vendors) == 2
    assert "dealersocket" in vendors
    assert "vinsolutions" in vendors
    
    # List vendors for specific dealer
    vendors_d1 = cred_manager._list_local_vendors("dealer1")
    
    # Verify filtered list
    assert len(vendors_d1) == 2
    assert "dealersocket" in vendors_d1
    assert "vinsolutions" in vendors_d1
    
    vendors_d2 = cred_manager._list_local_vendors("dealer2")
    
    # Verify filtered list
    assert len(vendors_d2) == 1
    assert "dealersocket" in vendors_d2

def test_credential_validation(cred_manager):
    """Test credential validation."""
    # Valid dealersocket credentials
    valid_ds = DealerCredential(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        client_id="test_client",
        client_secret="test_secret",
        dealer_code="TEST123"
    )
    
    assert cred_manager._validate_credential(valid_ds) is True
    
    # Invalid dealersocket credentials (missing required field)
    invalid_ds = DealerCredential(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        client_id="test_client",
        # Missing client_secret
        dealer_code="TEST123"
    )
    
    assert cred_manager._validate_credential(invalid_ds) is False
    
    # Valid vinsolutions credentials
    valid_vs = DealerCredential(
        dealer_id="test_dealer",
        vendor_id="vinsolutions",
        username="test_user",
        password="test_pass",
        api_key="test_key"
    )
    
    assert cred_manager._validate_credential(valid_vs) is True
    
    # Invalid vinsolutions credentials (missing required field)
    invalid_vs = DealerCredential(
        dealer_id="test_dealer",
        vendor_id="vinsolutions",
        username="test_user",
        password="test_pass",
        # Missing api_key
    )
    
    assert cred_manager._validate_credential(invalid_vs) is False

def test_credential_serialization():
    """Test credential serialization to/from dict."""
    # Create a credential
    cred = DealerCredential(
        dealer_id="test_dealer",
        vendor_id="dealersocket",
        client_id="test_client",
        client_secret="test_secret",
        dealer_code="TEST123",
        environment="production",
        metadata={"note": "Test credential"}
    )
    
    # Convert to dict
    cred_dict = cred.to_dict()
    
    # Verify dict contents
    assert cred_dict["dealer_id"] == "test_dealer"
    assert cred_dict["vendor_id"] == "dealersocket"
    assert cred_dict["client_id"] == "test_client"
    assert cred_dict["client_secret"] == "test_secret"
    assert cred_dict["dealer_code"] == "TEST123"
    assert cred_dict["environment"] == "production"
    assert cred_dict["metadata"] == {"note": "Test credential"}
    assert "created_at" in cred_dict
    assert "updated_at" in cred_dict
    
    # Convert back to credential
    new_cred = DealerCredential.from_dict(cred_dict)
    
    # Verify credential fields
    assert new_cred.dealer_id == cred.dealer_id
    assert new_cred.vendor_id == cred.vendor_id
    assert new_cred.client_id == cred.client_id
    assert new_cred.client_secret == cred.client_secret
    assert new_cred.dealer_code == cred.dealer_code
    assert new_cred.environment == cred.environment
    assert new_cred.metadata == cred.metadata
    assert new_cred.created_at is not None
    assert new_cred.updated_at is not None

@patch('src.nova_act.enhanced_credentials.get_secrets_manager_instance')
def test_credential_caching(mock_get_secrets, cred_manager, sample_credentials):
    """Test credential caching."""
    # Mock AWS Secrets Manager
    mock_secrets = MagicMock()
    mock_secrets.get_secret.return_value = sample_credentials.to_dict()
    mock_get_secrets.return_value = mock_secrets
    
    # Set flag to use AWS
    cred_manager.use_aws = True
    cred_manager.aws_secrets = mock_secrets
    
    # First call should hit AWS
    cred1 = cred_manager.get_credential("test_dealer", "dealersocket")
    
    assert cred1 is not None
    assert cred1.dealer_id == "test_dealer"
    assert cred1.vendor_id == "dealersocket"
    
    # Verify AWS was called
    mock_secrets.get_secret.assert_called_once()
    
    # Reset mock
    mock_secrets.get_secret.reset_mock()
    
    # Second call should use cache
    cred2 = cred_manager.get_credential("test_dealer", "dealersocket")
    
    assert cred2 is not None
    assert cred2.dealer_id == "test_dealer"
    assert cred2.vendor_id == "dealersocket"
    
    # Verify AWS was NOT called again
    mock_secrets.get_secret.assert_not_called()
    
    # Metadata should be updated
    cache_key = "test_dealer:dealersocket"
    assert cache_key in cred_manager.metadata_cache
    assert cred_manager.metadata_cache[cache_key].access_count > 0