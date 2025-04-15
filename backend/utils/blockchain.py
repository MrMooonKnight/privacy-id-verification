import os
import json
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

load_dotenv()

# Load blockchain configuration from environment variables
WEB3_PROVIDER = os.getenv('WEB3_PROVIDER', 'http://localhost:8545')
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')

# Initialize Web3 connection
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))

# Check if connected to blockchain
def is_connected():
    """Check if connected to Ethereum blockchain."""
    return w3.is_connected()

# Load contract ABI
def load_contract_abi(contract_name):
    """Load contract ABI from compiled JSON file."""
    try:
        with open(f'blockchain/build/contracts/{contract_name}.json', 'r') as file:
            contract_json = json.load(file)
            return contract_json['abi']
    except FileNotFoundError:
        # Fallback to contracts directory
        with open(f'contracts/{contract_name}.json', 'r') as file:
            contract_json = json.load(file)
            return contract_json['abi']

# Get contract instance
def get_contract(contract_address=None, contract_name='IdentityVerification'):
    """Get contract instance at the specified address."""
    if contract_address is None:
        contract_address = CONTRACT_ADDRESS

    if not contract_address:
        raise ValueError("Contract address is not provided")

    abi = load_contract_abi(contract_name)
    return w3.eth.contract(address=contract_address, abi=abi)

# Get account from private key
def get_account():
    """Get Ethereum account from private key."""
    if not PRIVATE_KEY:
        raise ValueError("Private key is not provided in environment variables")
    
    return Account.from_key(PRIVATE_KEY)

# Store identity hash on blockchain
def store_identity_hash(user_id, identity_hash):
    """
    Store a hash of identity data on the blockchain.
    
    Args:
        user_id: Unique identifier for the user
        identity_hash: Hash of the identity data
        
    Returns:
        transaction_hash: Hash of the transaction
    """
    try:
        contract = get_contract()
        account = get_account()
        
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Prepare transaction
        txn = contract.functions.storeIdentityHash(
            Web3.to_bytes(hexstr=user_id),
            Web3.to_bytes(hexstr=identity_hash)
        ).build_transaction({
            'chainId': w3.eth.chain_id,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': nonce,
        })
        
        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'transaction_hash': tx_receipt.transactionHash.hex(),
            'status': tx_receipt.status
        }
    
    except Exception as e:
        print(f"Error storing identity hash: {str(e)}")
        return {'error': str(e)}

# Verify identity hash on blockchain
def verify_identity_hash(user_id, identity_hash):
    """
    Verify if the identity hash matches what's stored on the blockchain.
    
    Args:
        user_id: Unique identifier for the user
        identity_hash: Hash of the identity data to verify
        
    Returns:
        is_valid: Boolean indicating if the hash is valid
    """
    try:
        contract = get_contract()
        
        # Call the verification function
        stored_hash = contract.functions.getIdentityHash(
            Web3.to_bytes(hexstr=user_id)
        ).call()
        
        # Convert stored hash to hex string for comparison
        stored_hash_hex = stored_hash.hex()
        
        # Remove '0x' prefix if present for comparison
        if identity_hash.startswith('0x'):
            identity_hash = identity_hash[2:]
        
        return stored_hash_hex == identity_hash
    
    except Exception as e:
        print(f"Error verifying identity hash: {str(e)}")
        return False

# Grant access to identity data
def grant_access(user_id, recipient_address, expiration_time):
    """
    Grant access to identity data to a specific address.
    
    Args:
        user_id: Unique identifier for the user
        recipient_address: Ethereum address to grant access to
        expiration_time: Unix timestamp when access expires
        
    Returns:
        transaction_hash: Hash of the transaction
    """
    try:
        contract = get_contract()
        account = get_account()
        
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Prepare transaction
        txn = contract.functions.grantAccess(
            Web3.to_bytes(hexstr=user_id),
            recipient_address,
            expiration_time
        ).build_transaction({
            'chainId': w3.eth.chain_id,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': nonce,
        })
        
        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'transaction_hash': tx_receipt.transactionHash.hex(),
            'status': tx_receipt.status
        }
    
    except Exception as e:
        print(f"Error granting access: {str(e)}")
        return {'error': str(e)}

# Revoke access to identity data
def revoke_access(user_id, recipient_address):
    """
    Revoke access to identity data from a specific address.
    
    Args:
        user_id: Unique identifier for the user
        recipient_address: Ethereum address to revoke access from
        
    Returns:
        transaction_hash: Hash of the transaction
    """
    try:
        contract = get_contract()
        account = get_account()
        
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Prepare transaction
        txn = contract.functions.revokeAccess(
            Web3.to_bytes(hexstr=user_id),
            recipient_address
        ).build_transaction({
            'chainId': w3.eth.chain_id,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': nonce,
        })
        
        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'transaction_hash': tx_receipt.transactionHash.hex(),
            'status': tx_receipt.status
        }
    
    except Exception as e:
        print(f"Error revoking access: {str(e)}")
        return {'error': str(e)}

# Check if an address has access to identity data
def check_access(user_id, recipient_address):
    """
    Check if an address has access to identity data.
    
    Args:
        user_id: Unique identifier for the user
        recipient_address: Ethereum address to check access for
        
    Returns:
        has_access: Boolean indicating if the address has access
        expiration_time: Expiration time for the access
    """
    try:
        contract = get_contract()
        
        # Call the access check function
        access_info = contract.functions.checkAccess(
            Web3.to_bytes(hexstr=user_id),
            recipient_address
        ).call()
        
        has_access = access_info[0]
        expiration_time = access_info[1]
        
        return {
            'has_access': has_access,
            'expiration_time': expiration_time
        }
    
    except Exception as e:
        print(f"Error checking access: {str(e)}")
        return {'error': str(e)} 