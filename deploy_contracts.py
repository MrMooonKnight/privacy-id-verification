#!/usr/bin/env python3
"""
Script to compile and deploy smart contracts for the identity verification system.
"""

import os
import json
import logging
import argparse
import yaml
from web3 import Web3
from solcx import compile_standard, install_solc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yml."""
    try:
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yml not found. Run python init_data.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def compile_contract(contract_name):
    """Compile a solidity contract."""
    # Install solc if not already installed
    install_solc("0.8.0")
    
    contract_path = f"src/blockchain/contracts/{contract_name}.sol"
    
    # Read the contract source code
    with open(contract_path, 'r') as f:
        contract_source = f.read()
    
    # Compile the contract
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {
                f"{contract_name}.sol": {
                    "content": contract_source
                }
            },
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                }
            }
        },
        solc_version="0.8.0"
    )
    
    # Save compiled contract
    os.makedirs('artifacts', exist_ok=True)
    with open(f"artifacts/{contract_name}.json", 'w') as f:
        json.dump(compiled_sol, f)
    
    # Extract abi and bytecode
    abi = compiled_sol["contracts"][f"{contract_name}.sol"][contract_name]["abi"]
    bytecode = compiled_sol["contracts"][f"{contract_name}.sol"][contract_name]["evm"]["bytecode"]["object"]
    
    # Save ABI to the contracts directory
    abi_file = f"src/blockchain/contracts/{contract_name.lower()}_abi.json"
    with open(abi_file, 'w') as f:
        json.dump(abi, f)
    
    logger.info(f"Contract {contract_name} compiled successfully. ABI saved to {abi_file}")
    
    return abi, bytecode

def deploy_contract(w3, abi, bytecode, account, constructor_args=None):
    """Deploy a contract to the blockchain."""
    # Create contract object
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Get nonce
    nonce = w3.eth.get_transaction_count(account.address)
    
    # Build transaction
    if constructor_args:
        # Deploy with constructor arguments
        transaction = Contract.constructor(*constructor_args).build_transaction(
            {
                "chainId": w3.eth.chain_id,
                "from": account.address,
                "nonce": nonce,
                "gasPrice": w3.eth.gas_price
            }
        )
    else:
        # Deploy without constructor arguments
        transaction = Contract.constructor().build_transaction(
            {
                "chainId": w3.eth.chain_id,
                "from": account.address,
                "nonce": nonce,
                "gasPrice": w3.eth.gas_price
            }
        )
    
    # Sign transaction
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=account.key)
    
    # Send transaction
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    logger.info(f"Transaction sent: {tx_hash.hex()}")
    
    # Wait for transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    logger.info(f"Contract deployed at address: {tx_receipt.contractAddress}")
    
    return tx_receipt.contractAddress

def main():
    """Main function to deploy the contracts."""
    parser = argparse.ArgumentParser(description='Deploy smart contracts')
    parser.add_argument('--private-key', help='Private key for contract deployment')
    parser.add_argument('--network', default='development', help='Network to deploy to (default: development)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Connect to blockchain
    w3 = Web3(Web3.HTTPProvider(config['blockchain']['node_url']))
    if not w3.is_connected():
        logger.error("Failed to connect to blockchain node")
        return False
    
    logger.info(f"Connected to blockchain node: {config['blockchain']['node_url']}")
    logger.info(f"Current block number: {w3.eth.block_number}")
    
    # Set up account
    if args.private_key:
        private_key = args.private_key
    else:
        # For development only - NEVER use this in production!
        private_key = "0x" + "1" * 64  # A dummy private key, will only work on local development networks
    
    account = w3.eth.account.from_key(private_key)
    logger.info(f"Deploying contracts from address: {account.address}")
    
    # Compile and deploy IdentityRegistry
    logger.info("Compiling IdentityRegistry contract...")
    identity_abi, identity_bytecode = compile_contract("IdentityRegistry")
    
    logger.info("Deploying IdentityRegistry contract...")
    identity_address = deploy_contract(w3, identity_abi, identity_bytecode, account)
    
    # Compile and deploy VerificationContract
    logger.info("Compiling VerificationContract contract...")
    verification_abi, verification_bytecode = compile_contract("VerificationContract")
    
    logger.info("Deploying VerificationContract contract...")
    verification_address = deploy_contract(w3, verification_abi, verification_bytecode, account, [identity_address])
    
    # Update config with contract addresses
    config['blockchain']['contracts']['identity_registry'] = identity_address
    config['blockchain']['contracts']['verification'] = verification_address
    
    with open('config.yml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info("Contracts deployed successfully:")
    logger.info(f"IdentityRegistry: {identity_address}")
    logger.info(f"VerificationContract: {verification_address}")
    logger.info("Contract addresses updated in config.yml")
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 