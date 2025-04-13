"""
Basic tests for the identity verification system.
"""

import os
import sys
import unittest
import yaml
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.blockchain.blockchain_manager import BlockchainManager
from src.ai.ai_manager import AIManager
from src.crypto.crypto_manager import CryptoManager
from src.identity.identity_manager import IdentityManager


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests for the identity verification system."""

    def setUp(self):
        """Set up test fixtures."""
        # Load test configuration
        with open('config.example.yml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create mock managers
        self.blockchain_manager = MagicMock(spec=BlockchainManager)
        self.blockchain_manager.store_identity.return_value = "0x1234567890"
        self.blockchain_manager.verify_identity.return_value = "0x0987654321"
        
        self.crypto_manager = MagicMock(spec=CryptoManager)
        self.crypto_manager.hash_identity_data.return_value = "0xabcdef1234567890"
        self.crypto_manager.encrypt_data.return_value = "encrypted_data"
        self.crypto_manager.decrypt_data.return_value = {
            "user_id": "test_user",
            "user_data": {
                "first_name": "Test",
                "last_name": "User"
            },
            "verification_status": "pending"
        }
        
        self.ai_manager = MagicMock(spec=AIManager)
        self.ai_manager.verify_face.return_value = {
            "match": True,
            "confidence": 0.95,
            "error": None
        }
        
        # Create identity manager with mock dependencies
        self.identity_manager = IdentityManager(
            self.blockchain_manager,
            self.crypto_manager,
            self.ai_manager,
            self.config['compliance']
        )

    def test_identity_registration(self):
        """Test identity registration."""
        # Test data
        user_data = {
            "first_name": "Test",
            "last_name": "User",
            "email": "test.user@example.com",
            "date_of_birth": "1990-01-01"
        }
        
        # Register identity
        result = self.identity_manager.register_identity(user_data)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("user_id", result)
        self.assertEqual(result["verification_status"], "pending")
        
        # Verify method calls
        self.crypto_manager.hash_identity_data.assert_called_once()
        self.blockchain_manager.store_identity.assert_called_once()
        self.crypto_manager.encrypt_data.assert_called_once()

    def test_identity_verification(self):
        """Test identity verification."""
        # Create a test user
        user_data = {"first_name": "Test", "last_name": "User"}
        registration = self.identity_manager.register_identity(user_data)
        user_id = registration["user_id"]
        
        # Mock verification data
        verification_data = {
            "reference_image": b"mock_reference_image",
            "verification_image": b"mock_verification_image"
        }
        
        # Verify identity
        result = self.identity_manager.verify_identity(user_id, "facial", verification_data)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("request_id", result)
        self.assertEqual(result["status"], "completed")
        
        # Verify method calls
        self.blockchain_manager.verify_identity.assert_called_once()
        self.ai_manager.verify_face.assert_called_once()

    def test_get_identity(self):
        """Test retrieving identity data."""
        # Create a test user
        user_data = {"first_name": "Test", "last_name": "User"}
        registration = self.identity_manager.register_identity(user_data)
        user_id = registration["user_id"]
        
        # Get identity
        result = self.identity_manager.get_identity(user_id)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("identity", result)
        self.assertEqual(result["identity"]["user_id"], "test_user")
        
        # Verify method calls
        self.crypto_manager.decrypt_data.assert_called_once()


if __name__ == '__main__':
    unittest.main() 