import sys
import os
import json
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.privacy import ZeroKnowledgeProof, create_identity_proof, verify_identity_proof

def test_zkp_implementation():
    """Test the ZeroKnowledgeProof implementation."""
    print("Testing Zero-Knowledge Proof Implementation")
    print("-" * 50)
    
    # Create a sample identity
    identity_data = {
        "user_id": "test-user-123",
        "document_fields": {
            "name": "Test User",
            "document_number": "ABC123456",
            "date_of_birth": "1990-01-01"
        },
        "face_hash": "a1b2c3d4e5f6g7h8i9j0",
        "timestamp": time.time()
    }
    
    user_id = "test-user-123"
    
    # Create a ZKP directly
    zkp = ZeroKnowledgeProof()
    proof = zkp.create_proof(identity_data, user_id)
    
    print(f"Created proof of method: {proof.get('metadata', {}).get('method')}")
    print(f"Proof summary: {json.dumps(proof)[:200]}...")
    
    # Verify the proof
    is_valid = zkp.verify_proof(proof, user_id)
    print(f"Proof verification result: {is_valid}")
    
    # Test utility functions
    utility_proof = create_identity_proof(identity_data, user_id)
    utility_valid = verify_identity_proof(utility_proof, user_id)
    
    print(f"Utility function verification result: {utility_valid}")
    
    # Test with wrong user ID
    wrong_valid = verify_identity_proof(proof, "wrong-user-id")
    print(f"Verification with wrong user ID: {wrong_valid} (expected: False)")
    
    return is_valid and utility_valid and not wrong_valid

if __name__ == "__main__":
    # Create test directory if it doesn't exist
    os.makedirs("tests", exist_ok=True)
    
    success = test_zkp_implementation()
    if success:
        print("\nAll ZKP tests passed successfully!")
    else:
        print("\nSome ZKP tests failed!")
        sys.exit(1) 