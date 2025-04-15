import os
import json
import base64
import hashlib
import pickle
import random
import hmac
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
import pyope
from pyope.ope import OPE
from pyope.ope import ValueRange
from dotenv import load_dotenv
from utils.encryption import encrypt_data, decrypt_data, secure_hash

# Import zksk for zero-knowledge proofs
try:
    from zksk import Secret, DLRep
    from zksk import utils
    ZKSK_AVAILABLE = True
except ImportError:
    print("Warning: zksk library not available. Using simplified ZKP implementation.")
    ZKSK_AVAILABLE = False

load_dotenv()

ZKP_PARAMS = json.loads(os.getenv("ZKP_PARAMS", '{"g": 13, "h": 19, "p": 2147483647, "q": 1073741823}'))


# Zero-knowledge proof implementation
class ZeroKnowledgeProof:
    """
    Implementation of zero-knowledge proofs for identity verification.
    Uses the zksk library if available, otherwise falls back to simplified implementation.
    """
    
    def __init__(self):
        """Initialize the ZKP system with parameters."""
        self.params = ZKP_PARAMS
        self.g = self.params.get('g', 13)
        self.h = self.params.get('h', 19)
        self.p = self.params.get('p', 2147483647)  # Large prime
        self.q = self.params.get('q', 1073741823)  # p = 2q + 1
        
    def _hash_to_int(self, data):
        """Convert data to integer representation using secure hash."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        elif not isinstance(data, str):
            data = str(data)
            
        hash_bytes = hashlib.sha256(data.encode()).digest()
        # Convert to integer and ensure it's within the range of our modulus
        return int.from_bytes(hash_bytes, 'big') % self.p
    
    def create_proof(self, identity_data, user_id):
        """
        Create a zero-knowledge proof for identity verification.
        
        Args:
            identity_data: The identity data to prove knowledge of
            user_id: User identifier
            
        Returns:
            dict: Proof data that can be verified without revealing the identity data
        """
        if ZKSK_AVAILABLE:
            return self._create_zksk_proof(identity_data, user_id)
        else:
            return self._create_simplified_proof(identity_data, user_id)
    
    def _create_zksk_proof(self, identity_data, user_id):
        """Create ZKP using zksk library."""
        # Convert identity data to a numeric value
        x = self._hash_to_int(identity_data)
        
        # Set up the secret
        secret = Secret(x)
        
        # Create the first component: g^x (mod p)
        g_x = pow(self.g, x, self.p)
        
        # Create the statement to prove: knowledge of x in g^x = y
        statement = DLRep(g_x, self.g, secret)
        
        # Generate the proof
        zk_proof = statement.prove()
        
        # Create a verifiable proof package
        proof_data = {
            "user_id": user_id,
            "g_x": g_x,  # Public commitment
            "proof": utils.serialize_dict(zk_proof),
            "timestamp": int(time.time()),
            "metadata": {
                "method": "zksk-dlrep",
                "version": "1.0",
                "g": self.g,
                "p": self.p
            }
        }
        
        return proof_data
    
    def _create_simplified_proof(self, identity_data, user_id):
        """
        Create a zero-knowledge proof using a simplified Schnorr protocol.
        This is used when the zksk library is not available.
        
        The Schnorr protocol works as follows:
        1. Prover has secret x
        2. Prover computes y = g^x mod p (the public key)
        3. Prover generates random r and computes t = g^r mod p (the commitment)
        4. Verifier provides challenge e (in non-interactive version, e = hash(t || message))
        5. Prover computes response s = r + e*x mod q
        6. Verifier checks that g^s = t * y^e mod p
        
        Args:
            identity_data: The identity data (our secret x will be derived from this)
            user_id: User identifier (used as the message in the protocol)
            
        Returns:
            dict: A proof package containing the commitment, response, and public key
        """
        # Convert identity data to a numeric value (our secret x)
        x = self._hash_to_int(identity_data)
        
        # Create random value for commitment
        r = random.randint(1, self.q - 1)
        
        # Compute commitment t = g^r mod p
        commitment = pow(self.g, r, self.p)
        
        # Compute challenge e = hash(t || user_id)
        challenge_input = f"{commitment}{user_id}"
        challenge = self._hash_to_int(challenge_input)
        
        # Compute response s = r + x*e mod q
        response = (r + (x * challenge) % self.q) % self.q
        
        # Compute public key y = g^x mod p
        public_key = pow(self.g, x, self.p)
        
        # Create proof package
        proof_data = {
            "user_id": user_id,
            "commitment": commitment,
            "response": response,
            "y": public_key,
            "timestamp": int(time.time()),
            "metadata": {
                "method": "simplified-schnorr",
                "version": "1.0",
                "g": self.g,
                "p": self.p
            }
        }
        
        return proof_data
    
    def verify_proof(self, proof, user_id):
        """
        Verify a zero-knowledge proof.
        
        Args:
            proof: The proof object
            user_id: User identifier
            
        Returns:
            bool: Whether the proof is valid
        """
        # First check if the proof belongs to the claimed user
        if proof.get("user_id") != user_id:
            return False
        
        if ZKSK_AVAILABLE and proof.get("metadata", {}).get("method") == "zksk-dlrep":
            return self._verify_zksk_proof(proof)
        elif proof.get("metadata", {}).get("method") == "simplified-schnorr":
            return self._verify_simplified_proof(proof)
        else:
            # Legacy verification for older proofs
            return "hash" in proof and "timestamp" in proof and "metadata" in proof
    
    def _verify_zksk_proof(self, proof):
        """Verify a proof created with zksk."""
        try:
            # Extract proof components
            g_x = proof["g_x"]
            zk_proof = utils.deserialize_dict(proof["proof"])
            
            # Recreate the statement with the claimed public value
            secret_var = Secret()
            statement = DLRep(g_x, self.g, secret_var)
            
            # Verify the proof
            return statement.verify(zk_proof)
        except Exception as e:
            print(f"Error verifying zksk proof: {str(e)}")
            return False
    
    def _verify_simplified_proof(self, proof):
        """
        Verify a simplified Schnorr proof.
        
        The verification check is: g^s == t * y^e mod p
        Where:
        - g is the generator
        - s is the response
        - t is the commitment
        - y is the public key
        - e is the challenge
        
        Args:
            proof: The proof object containing commitment, response, and public key
            
        Returns:
            bool: Whether the proof is valid
        """
        try:
            # Extract components
            commitment = proof["commitment"]  # t
            response = proof["response"]      # s
            user_id = proof["user_id"]
            public_key = proof.get("y")       # y
            
            if public_key is None:
                return False
            
            # Recompute challenge e = hash(t || user_id)
            challenge_input = f"{commitment}{user_id}"
            challenge = self._hash_to_int(challenge_input)
            
            # Verify g^s == t * y^e mod p
            left_side = pow(self.g, response, self.p)            # g^s
            right_side = (commitment * pow(public_key, challenge, self.p)) % self.p  # t * y^e
            
            return left_side == right_side
            
        except Exception as e:
            print(f"Error verifying simplified proof: {str(e)}")
            return False
    
    def generate_verification_request(self, proof, verifier_id):
        """
        Generate a request for verification without revealing the underlying proof.
        This allows third parties to verify identity aspects without seeing the data.
        
        Args:
            proof: The original proof
            verifier_id: ID of the entity requesting verification
            
        Returns:
            dict: Verification request that can be sent to a verifier
        """
        # Create a derived proof with limited information
        request = {
            "verifier_id": verifier_id,
            "proof_id": secure_hash(json.dumps(proof))[:16],
            "user_id": proof["user_id"],
            "timestamp": int(time.time()),
            "metadata": {
                "type": "verification_request",
                "method": proof.get("metadata", {}).get("method"),
                "attributes_requested": ["identity_valid"]  # What we want to verify
            }
        }
        
        # Add a HMAC to prevent tampering
        request["hmac"] = hmac.new(
            secure_hash(json.dumps(proof)).encode(),
            json.dumps(request, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
        
        return request

# Legacy functions maintained for backward compatibility
def create_identity_proof(identity_data, user_id):
    """
    Create a zero-knowledge proof for identity verification.
    
    Args:
        identity_data: The identity data to prove knowledge of
        user_id: User identifier
    
    Returns:
        dict: Proof that can be verified without revealing the original data
    """
    zkp = ZeroKnowledgeProof()
    return zkp.create_proof(identity_data, user_id)

def verify_identity_proof(proof, user_id):
    """
    Verify a zero-knowledge proof for identity without seeing the original data.
    
    Args:
        proof: The proof object
        user_id: User identifier
    
    Returns:
        bool: Whether the proof is valid
    """
    zkp = ZeroKnowledgeProof()
    return zkp.verify_proof(proof, user_id)

# Initialize homomorphic encryption with PyOPE
def initialize_homomorphic():
    """Initialize Order-Preserving Encryption for homomorphic operations."""
    if os.getenv('HOMOMORPHIC_KEY'):
        try:
            key = base64.b64decode(os.getenv('HOMOMORPHIC_KEY'))
        except:
            key = os.getenv('HOMOMORPHIC_KEY').encode()
    else:
        key = os.urandom(16)  # Generate a new key if not provided
    
    # OPE allows computations on encrypted data while preserving order
    in_range = ValueRange(0, 2**31 - 1)  # Input range
    out_range = ValueRange(0, 2**63 - 1)  # Output range
    
    return OPE(key, in_range, out_range)


# Homomorphic encryption functionality
def homomorphic_encrypt(value, cipher=None):
    """Encrypt values while preserving order relationships."""
    if cipher is None:
        cipher = initialize_homomorphic()
    
    if isinstance(value, (int, float)):
        # For numeric values, encrypt directly
        value_int = int(value)
        return cipher.encrypt(value_int)
    elif isinstance(value, str):
        # For strings, hash first to get numeric representation
        hash_int = int(secure_hash(value)[:16], 16)  # Take first 16 chars of hash as hex int
        return cipher.encrypt(hash_int % (2**31 - 1))  # Ensure in valid range
    else:
        raise ValueError("Unsupported type for homomorphic encryption")


def homomorphic_decrypt(encrypted_value, cipher=None):
    """Decrypt homomorphically encrypted value."""
    if cipher is None:
        cipher = initialize_homomorphic()
    
    return cipher.decrypt(encrypted_value)


# Zero-knowledge proof functionality
def make_generators(n, seed=None):
    """
    Generate random generators for zero-knowledge proofs.
    In a real system, these would be carefully selected mathematical parameters.
    
    Args:
        n: Number of generators to create
        seed: Random seed for reproducibility
        
    Returns:
        List of n generators (simplified as random values)
    """
    if seed is not None:
        random.seed(seed)
    
    # In a simplified approach, we just generate random numbers
    # In a real ZKP system, these would be specific mathematical constructs
    generators = []
    for _ in range(n):
        # Generate a random 64-bit number as a "generator"
        g = random.getrandbits(64)
        generators.append(g)
    
    return generators

def generate_zkp_params():
    """Generate parameters for zero-knowledge proofs."""
    g1, g2, g3 = make_generators(3, seed=42)
    return {
        'g1': g1,
        'g2': g2,
        'g3': g3
    }


def save_zkp_params(params, file_path):
    """Save ZKP parameters to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(params, f)


def load_zkp_params(file_path=None):
    """Load ZKP parameters from file or create new ones."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    # Generate new params if not available
    return generate_zkp_params()


# Homomorphic encryption implementation using PyOPE
class HomomorphicEncryption:
    def __init__(self, key=None):
        """
        Initialize homomorphic encryption with a key.
        
        Args:
            key: Encryption key (optional)
        """
        if key is None:
            key = os.getenv("HOMOMORPHIC_KEY", os.urandom(16).hex())
        
        # Convert key to bytes if it's a string
        if isinstance(key, str):
            key = key.encode()
            
        # Create a cipher for order-preserving encryption
        self.cipher = OPE(key, in_range=[0, 2**32], out_range=[0, 2**64])
    
    def encrypt(self, value):
        """
        Encrypt a numeric value while preserving order.
        
        Args:
            value: Numeric value to encrypt
            
        Returns:
            int: Encrypted value
        """
        if not isinstance(value, int):
            value = int(value)
        return self.cipher.encrypt(value)
    
    def decrypt(self, encrypted_value):
        """
        Decrypt a homomorphically encrypted value.
        
        Args:
            encrypted_value: Encrypted value
            
        Returns:
            int: Decrypted value
        """
        return self.cipher.decrypt(encrypted_value)


# Convenience functions
def homomorphic_encrypt(value, key=None):
    """Encrypt a value using homomorphic encryption."""
    encryptor = HomomorphicEncryption(key)
    return encryptor.encrypt(value)


def homomorphic_decrypt(encrypted_value, key=None):
    """Decrypt a homomorphically encrypted value."""
    encryptor = HomomorphicEncryption(key)
    return encryptor.decrypt(encrypted_value) 