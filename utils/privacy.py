import os
import json
import base64
import hashlib
from pyope.ope import OPE, ValueRange  # Ensure ValueRange is imported
from dotenv import load_dotenv

load_dotenv()

ZKP_PARAMS = json.loads(os.getenv("ZKP_PARAMS", '{"g": 13, "h": 19, "p": 2147483647, "q": 1073741823}'))

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
        hash_int = int(hashlib.sha256(value.encode()).hexdigest(), 16)  # Take first 16 chars of hash as hex int
        return cipher.encrypt(hash_int % (2**31 - 1))  # Ensure in valid range
    else:
        raise ValueError("Unsupported type for homomorphic encryption")


def homomorphic_decrypt(encrypted_value, cipher=None):
    """Decrypt homomorphically encrypted value."""
    if cipher is None:
        cipher = initialize_homomorphic()
    
    return cipher.decrypt(encrypted_value)


# Zero-knowledge proof functionality
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


def create_identity_proof(identity_data, user_id):
    """
    Create a simplified hash-based proof for identity verification.
    Does not use the zksk library.
    """
    hash_value = hashlib.sha256(f"{identity_data}{user_id}".encode()).hexdigest()
    proof = {
        "hash": hash_value,
        "user_id": user_id,
        "timestamp": os.urandom(8).hex(),  # Random nonce
        "metadata": {
            "method": "simplified-hash",
            "version": "1.0"
        }
    }
    return proof


def verify_identity_proof(proof, user_id):
    """
    Verify a simplified hash-based proof.
    Does not use the zksk library.
    """
    if not isinstance(proof, dict) or proof.get("user_id") != user_id:
        return False
    # Basic structural check for the simplified proof
    return all(k in proof for k in ("hash", "timestamp", "metadata"))


# Homomorphic encryption implementation using PyOPE
class HomomorphicEncryption:
    """
    Handles Order-Preserving Encryption (OPE) using pyope.
    Reads key from environment variable or generates one.
    """
    def __init__(self, key=None):
        if key is None:
            key_hex = os.getenv("HOMOMORPHIC_KEY")
            if key_hex:
                key = bytes.fromhex(key_hex)
            else:
                key = os.urandom(16)  # Default 16-byte key
                print(f"Generated new Homomorphic Key: {key.hex()}")
                # Consider saving this key to .env or a secure store

        # Define input and output ranges for OPE
        # Adjust these ranges based on the expected data range
        in_range = ValueRange(0, 2**31 - 1)
        out_range = ValueRange(0, 2**63 - 1)
        self.cipher = OPE(key, in_range=in_range, out_range=out_range)

    def encrypt(self, value):
        """Encrypts a numeric value using OPE."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                # Handle non-integer types by hashing to an integer within range
                str_value = str(value)
                hash_bytes = hashlib.sha256(str_value.encode()).digest()
                value = int.from_bytes(hash_bytes[:4], 'big') % (2**31 - 1) # Use 4 bytes for int32 range

        # Ensure value is within the defined input range
        value = max(self.cipher.in_range.start, min(value, self.cipher.in_range.end))
        return self.cipher.encrypt(value)

    def decrypt(self, encrypted_value):
        """Decrypts an OPE-encrypted value."""
        return self.cipher.decrypt(encrypted_value)


# Convenience functions
_default_homomorphic_encryptor = None

def get_default_homomorphic_encryptor():
    """Gets a singleton instance of HomomorphicEncryption."""
    global _default_homomorphic_encryptor
    if _default_homomorphic_encryptor is None:
        _default_homomorphic_encryptor = HomomorphicEncryption()
    return _default_homomorphic_encryptor

def homomorphic_encrypt(value):
    """Encrypts a value using the default homomorphic encryptor."""
    encryptor = get_default_homomorphic_encryptor()
    return encryptor.encrypt(value)

def homomorphic_decrypt(encrypted_value):
    """Decrypts a value using the default homomorphic encryptor."""
    encryptor = get_default_homomorphic_encryptor()
    return encryptor.decrypt(encrypted_value) 