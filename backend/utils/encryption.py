import os
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
if not ENCRYPTION_KEY:
    # Generate a key if not provided in environment
    ENCRYPTION_KEY = Fernet.generate_key().decode()


def derive_key(password, salt=None):
    """Derive a key from password using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = kdf.derive(password.encode())
    return key, salt


def encrypt_data(data, key=None):
    """Encrypt data using AES-256-GCM."""
    if key is None:
        fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
        return fernet.encrypt(data.encode() if isinstance(data, str) else data)
    
    # If a specific key is provided, use AES-GCM
    iv = os.urandom(12)
    encryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
    ).encryptor()
    
    ciphertext = encryptor.update(data.encode() if isinstance(data, str) else data) + encryptor.finalize()
    
    # Return IV, ciphertext, and tag
    return {
        'iv': base64.b64encode(iv).decode('utf-8'),
        'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
        'tag': base64.b64encode(encryptor.tag).decode('utf-8')
    }


def decrypt_data(encrypted_data, key=None):
    """Decrypt data encrypted with AES-256-GCM."""
    if key is None:
        fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
        return fernet.decrypt(encrypted_data).decode()
    
    # If a specific key is provided, use AES-GCM
    iv = base64.b64decode(encrypted_data['iv'])
    ciphertext = base64.b64decode(encrypted_data['ciphertext'])
    tag = base64.b64decode(encrypted_data['tag'])
    
    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
    ).decryptor()
    
    return decryptor.update(ciphertext) + decryptor.finalize()


def secure_hash(data):
    """Create a secure hash of the data."""
    digest = hashes.Hash(hashes.SHA256())
    digest.update(data.encode() if isinstance(data, str) else data)
    return digest.finalize().hex() 