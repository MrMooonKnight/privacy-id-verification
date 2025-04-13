"""
Identity Manager module for handling identity verification processes.
"""

import logging
import uuid
import json
import datetime
import os

logger = logging.getLogger(__name__)

class IdentityManager:
    """Manages identity verification processes and user identity data."""
    
    def __init__(self, blockchain_manager, crypto_manager, ai_manager, compliance_config):
        """
        Initialize the IdentityManager with the required dependencies.
        
        Args:
            blockchain_manager: The blockchain manager for storing identity data
            crypto_manager: The crypto manager for cryptographic operations
            ai_manager: The AI manager for verification and fraud detection
            compliance_config (dict): Configuration for compliance requirements
        """
        self.blockchain_manager = blockchain_manager
        self.crypto_manager = crypto_manager
        self.ai_manager = ai_manager
        self.compliance_config = compliance_config
        self._initialize_storage()
        logger.info("IdentityManager initialized")
    
    def _initialize_storage(self):
        """Initialize local storage for identity data."""
        # In a production environment, this would use a secure database
        # For this demo, we'll use a simple in-memory store
        self.identity_store = {}
        self.verification_requests = {}
        
        # Create a directory for temporary storage of identity documents
        os.makedirs("data/identity", exist_ok=True)
    
    def register_identity(self, user_data, biometric_data=None, document_data=None):
        """
        Register a new identity in the system.
        
        Args:
            user_data (dict): Basic user information
            biometric_data (dict, optional): Biometric data for verification
            document_data (dict, optional): Government ID document data
            
        Returns:
            dict: Registration result including user ID
        """
        try:
            # Generate a unique user ID
            user_id = str(uuid.uuid4())
            
            # Create a complete identity record
            identity = {
                "user_id": user_id,
                "user_data": user_data,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "verification_status": "pending",
                "verification_methods": []
            }
            
            # Add biometric data if provided
            if biometric_data:
                # Store biometric templates, not raw data
                # In a real system, this would properly process and secure biometric data
                identity["biometric_template"] = {
                    "type": biometric_data.get("type", "facial"),
                    "template_hash": self.crypto_manager.hash_identity_data(biometric_data)
                }
                identity["verification_methods"].append("biometric")
            
            # Add document data if provided
            if document_data:
                # Store document info, not raw data
                # In a real system, this would properly process and secure document data
                identity["document_info"] = {
                    "type": document_data.get("type", "passport"),
                    "document_hash": self.crypto_manager.hash_identity_data(document_data),
                    "issued_by": document_data.get("issued_by"),
                    "expiration_date": document_data.get("expiration_date")
                }
                identity["verification_methods"].append("document")
            
            # Hash the identity data for storage on blockchain
            identity_hash = self.crypto_manager.hash_identity_data(identity)
            
            # Set default permissions (only user can access)
            permissions = {
                "owner": user_id,
                "authorized_viewers": [],
                "expiration": (datetime.datetime.now() + datetime.timedelta(days=365)).isoformat()
            }
            
            # Store the identity hash on the blockchain
            tx_hash = self.blockchain_manager.store_identity(user_id, identity_hash, permissions)
            
            if not tx_hash:
                logger.error(f"Failed to store identity on blockchain for user {user_id}")
                return {
                    "success": False,
                    "user_id": user_id,
                    "error": "Failed to store identity on blockchain"
                }
            
            # Encrypt the identity data before storing locally
            encrypted_identity = self.crypto_manager.encrypt_data(identity)
            
            # Store the encrypted identity data
            self.identity_store[user_id] = {
                "encrypted_data": encrypted_identity,
                "permissions": permissions,
                "blockchain_tx": tx_hash
            }
            
            logger.info(f"Identity registered for user {user_id}")
            return {
                "success": True,
                "user_id": user_id,
                "tx_hash": tx_hash,
                "verification_status": "pending",
                "verification_methods": identity["verification_methods"]
            }
            
        except Exception as e:
            logger.error(f"Error registering identity: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def verify_identity(self, user_id, verification_type, verification_data):
        """
        Verify a user's identity using the specified method.
        
        Args:
            user_id (str): The user's identifier
            verification_type (str): The type of verification to perform
            verification_data (dict): Data for the verification
            
        Returns:
            dict: Verification result
        """
        try:
            # Check if the user exists
            if user_id not in self.identity_store:
                logger.warning(f"User {user_id} not found for verification")
                return {
                    "success": False,
                    "error": "User not found"
                }
            
            # Get the encrypted identity data
            encrypted_identity = self.identity_store[user_id]["encrypted_data"]
            
            # Decrypt the identity data
            identity = self.crypto_manager.decrypt_data(encrypted_identity)
            
            # Create a verification request
            request_id = str(uuid.uuid4())
            verification_request = {
                "request_id": request_id,
                "user_id": user_id,
                "verification_type": verification_type,
                "status": "pending",
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "result": None
            }
            
            # Store the verification request
            self.verification_requests[request_id] = verification_request
            
            # Record the verification request on the blockchain
            verifier_id = "system"  # In a real system, this would be a specific verifier
            tx_hash = self.blockchain_manager.verify_identity(user_id, verifier_id, verification_type)
            verification_request["blockchain_tx"] = tx_hash
            
            # Perform the appropriate verification
            if verification_type == "facial":
                # Verify face against stored template
                if "biometric_template" not in identity or identity["biometric_template"]["type"] != "facial":
                    logger.warning(f"Facial template not found for user {user_id}")
                    verification_request["status"] = "failed"
                    verification_request["result"] = {
                        "success": False,
                        "error": "Facial template not found"
                    }
                else:
                    # In a real system, this would retrieve the actual reference image
                    # For this demo, we'll assume we have the reference and verification images
                    reference_image = verification_data.get("reference_image")
                    verification_image = verification_data.get("verification_image")
                    
                    if not reference_image or not verification_image:
                        logger.warning(f"Missing images for facial verification of user {user_id}")
                        verification_request["status"] = "failed"
                        verification_request["result"] = {
                            "success": False,
                            "error": "Missing images for verification"
                        }
                    else:
                        # Perform facial verification using AI manager
                        result = self.ai_manager.verify_face(verification_image, reference_image)
                        verification_request["status"] = "completed"
                        verification_request["result"] = result
                        
                        # Update identity verification status if successful
                        if result["match"]:
                            identity["verification_status"] = "verified"
                            identity["updated_at"] = datetime.datetime.now().isoformat()
                            
                            # Re-encrypt and store the updated identity
                            self.identity_store[user_id]["encrypted_data"] = self.crypto_manager.encrypt_data(identity)
            
            elif verification_type == "document":
                # Verify government ID document
                if not verification_data.get("document_image"):
                    logger.warning(f"Missing document image for user {user_id}")
                    verification_request["status"] = "failed"
                    verification_request["result"] = {
                        "success": False,
                        "error": "Missing document image"
                    }
                else:
                    # Verify the document using AI manager
                    result = self.ai_manager.verify_document(verification_data["document_image"])
                    verification_request["status"] = "completed"
                    verification_request["result"] = result
                    
                    # Update identity verification status if successful
                    if result["valid"]:
                        identity["verification_status"] = "verified"
                        identity["updated_at"] = datetime.datetime.now().isoformat()
                        
                        # Extract and store document information
                        if "extracted_info" in result and result["extracted_info"]:
                            identity["document_info"] = {
                                **identity.get("document_info", {}),
                                **result["extracted_info"]
                            }
                        
                        # Re-encrypt and store the updated identity
                        self.identity_store[user_id]["encrypted_data"] = self.crypto_manager.encrypt_data(identity)
            
            else:
                logger.warning(f"Unsupported verification type: {verification_type}")
                verification_request["status"] = "failed"
                verification_request["result"] = {
                    "success": False,
                    "error": f"Unsupported verification type: {verification_type}"
                }
            
            # Check for fraud
            fraud_detection_data = {
                "user_id": user_id,
                "verification_type": verification_type,
                "ip_reputation": verification_data.get("ip_reputation", 0.5),
                "geo_velocity": verification_data.get("geo_velocity", 0.5),
                "device_reputation": verification_data.get("device_reputation", 0.5),
                "behavioral_score": verification_data.get("behavioral_score", 0.5)
            }
            
            fraud_result = self.ai_manager.detect_fraud(fraud_detection_data)
            verification_request["fraud_detection"] = fraud_result
            
            # Update verification request
            verification_request["updated_at"] = datetime.datetime.now().isoformat()
            
            # Check compliance requirements
            self._ensure_compliance(user_id, verification_type, verification_request)
            
            return {
                "success": True,
                "request_id": request_id,
                "status": verification_request["status"],
                "result": verification_request["result"]
            }
            
        except Exception as e:
            logger.error(f"Error verifying identity: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_identity(self, user_id, requester_id=None):
        """
        Retrieve a user's identity data.
        
        Args:
            user_id (str): The user's identifier
            requester_id (str, optional): The ID of the entity requesting the data
            
        Returns:
            dict: The user's identity data (if authorized)
        """
        try:
            # Check if the user exists
            if user_id not in self.identity_store:
                logger.warning(f"User {user_id} not found")
                return {
                    "success": False,
                    "error": "User not found"
                }
            
            # Get the stored identity data
            stored_identity = self.identity_store[user_id]
            
            # Check permissions
            if requester_id and requester_id != user_id:
                permissions = stored_identity["permissions"]
                if requester_id not in permissions["authorized_viewers"]:
                    logger.warning(f"Unauthorized access attempt by {requester_id} for user {user_id}")
                    return {
                        "success": False,
                        "error": "Unauthorized access"
                    }
            
            # Decrypt the identity data
            identity = self.crypto_manager.decrypt_data(stored_identity["encrypted_data"])
            
            # Create a ZKP for sensitive information if needed
            if requester_id and requester_id != user_id:
                # Create proofs instead of returning raw data
                proofs = {}
                
                # Example: create age verification proof
                if "user_data" in identity and "date_of_birth" in identity["user_data"]:
                    dob = identity["user_data"]["date_of_birth"]
                    statement = "User is over 18 years old"
                    proofs["age_verification"] = self.crypto_manager.create_zero_knowledge_proof(
                        statement, dob
                    )
                
                # Return limited data with proofs
                return {
                    "success": True,
                    "user_id": user_id,
                    "verification_status": identity["verification_status"],
                    "proofs": proofs
                }
            else:
                # Remove sensitive data before returning
                if "biometric_template" in identity:
                    del identity["biometric_template"]
                
                return {
                    "success": True,
                    "identity": identity
                }
                
        except Exception as e:
            logger.error(f"Error retrieving identity: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_permissions(self, user_id, new_permissions, requester_id):
        """
        Update access permissions for a user's identity data.
        
        Args:
            user_id (str): The user's identifier
            new_permissions (dict): New permissions configuration
            requester_id (str): ID of the entity requesting the change
            
        Returns:
            dict: Update result
        """
        try:
            # Check if the user exists
            if user_id not in self.identity_store:
                logger.warning(f"User {user_id} not found for permission update")
                return {
                    "success": False,
                    "error": "User not found"
                }
            
            # Check if requester is authorized (must be the user)
            if requester_id != user_id:
                logger.warning(f"Unauthorized permission update attempt by {requester_id} for user {user_id}")
                return {
                    "success": False,
                    "error": "Unauthorized access"
                }
            
            # Get current permissions
            current_permissions = self.identity_store[user_id]["permissions"]
            
            # Update permissions
            updated_permissions = {**current_permissions, **new_permissions}
            
            # Store updated permissions
            self.identity_store[user_id]["permissions"] = updated_permissions
            
            # Update permissions on blockchain
            tx_hash = self.blockchain_manager.update_permissions(user_id, updated_permissions)
            
            if not tx_hash:
                logger.error(f"Failed to update permissions on blockchain for user {user_id}")
                return {
                    "success": False,
                    "error": "Failed to update permissions on blockchain"
                }
            
            logger.info(f"Permissions updated for user {user_id}")
            return {
                "success": True,
                "tx_hash": tx_hash
            }
            
        except Exception as e:
            logger.error(f"Error updating permissions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _ensure_compliance(self, user_id, verification_type, verification_request):
        """
        Ensure compliance with privacy regulations.
        
        Args:
            user_id (str): The user's identifier
            verification_type (str): The type of verification
            verification_request (dict): The verification request data
        """
        try:
            # GDPR compliance
            if self.compliance_config["gdpr"]["enabled"]:
                # Log for GDPR audit trail
                logger.info(f"GDPR compliance: Verification of type {verification_type} for user {user_id}")
                
                # Set data retention period
                retention_days = self.compliance_config["gdpr"]["data_retention_days"]
                expiration_date = datetime.datetime.now() + datetime.timedelta(days=retention_days)
                verification_request["data_retention"] = {
                    "expiration_date": expiration_date.isoformat(),
                    "policy": "GDPR"
                }
            
            # CCPA compliance
            if self.compliance_config["ccpa"]["enabled"]:
                # Log for CCPA audit trail
                logger.info(f"CCPA compliance: Verification of type {verification_type} for user {user_id}")
                
                # Add CCPA-specific compliance metadata
                verification_request["compliance"] = {
                    "ccpa_notice_provided": True,
                    "ccpa_timestamp": datetime.datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error ensuring compliance: {e}")
            # Log but don't fail the verification process 