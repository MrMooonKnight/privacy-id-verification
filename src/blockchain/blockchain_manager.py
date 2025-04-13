"""
Blockchain Manager module for interacting with the Ethereum blockchain.
"""

import logging
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import os

logger = logging.getLogger(__name__)

class BlockchainManager:
    """Manages all blockchain interactions including smart contracts and transactions."""
    
    def __init__(self, config):
        """
        Initialize the BlockchainManager with the provided configuration.
        
        Args:
            config (dict): Blockchain configuration including network info and contract addresses
        """
        self.config = config
        self.w3 = self._connect_to_network()
        self.contracts = self._load_contracts()
        logger.info(f"BlockchainManager initialized for {config['network']} network")
    
    def _connect_to_network(self):
        """Connect to the specified blockchain network."""
        try:
            w3 = Web3(Web3.HTTPProvider(self.config['node_url']))
            
            # Add middleware for PoA chains (like Goerli, Rinkeby)
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not w3.is_connected():
                logger.error(f"Failed to connect to {self.config['network']} network")
                raise ConnectionError(f"Could not connect to {self.config['network']}")
            
            logger.info(f"Connected to {self.config['network']} network")
            logger.debug(f"Current block number: {w3.eth.block_number}")
            return w3
        
        except Exception as e:
            logger.error(f"Error connecting to blockchain: {e}")
            raise
    
    def _load_contracts(self):
        """Load smart contract ABIs and addresses."""
        contracts = {}
        try:
            contract_dir = os.path.join(os.path.dirname(__file__), 'contracts')
            
            # Identity Registry Contract
            with open(os.path.join(contract_dir, 'identity_registry_abi.json'), 'r') as f:
                identity_registry_abi = json.load(f)
            
            identity_registry_address = self.config['contracts']['identity_registry']
            contracts['identity_registry'] = self.w3.eth.contract(
                address=identity_registry_address,
                abi=identity_registry_abi
            )
            
            # Verification Contract
            with open(os.path.join(contract_dir, 'verification_abi.json'), 'r') as f:
                verification_abi = json.load(f)
            
            verification_address = self.config['contracts']['verification']
            contracts['verification'] = self.w3.eth.contract(
                address=verification_address,
                abi=verification_abi
            )
            
            logger.info("Smart contracts loaded successfully")
            return contracts
            
        except FileNotFoundError as e:
            logger.warning(f"Contract ABI file not found: {e}. Using empty contracts.")
            return {}
        except Exception as e:
            logger.error(f"Error loading contracts: {e}")
            raise
    
    def get_identity(self, user_id):
        """
        Retrieve identity data from the blockchain.
        
        Args:
            user_id (str): The user's identifier
            
        Returns:
            dict: The user's identity data
        """
        try:
            if 'identity_registry' not in self.contracts:
                logger.warning("Identity registry contract not available")
                return None
                
            # Placeholder for actual contract call
            # In a real implementation, this would query the contract
            identity_data = self.contracts['identity_registry'].functions.getIdentity(user_id).call()
            return identity_data
        except Exception as e:
            logger.error(f"Error retrieving identity: {e}")
            return None
    
    def store_identity(self, user_id, identity_hash, permissions):
        """
        Store identity data hash on the blockchain.
        
        Args:
            user_id (str): The user's identifier
            identity_hash (str): The hash of the user's identity data
            permissions (dict): Access permissions for the identity data
            
        Returns:
            str: Transaction hash
        """
        try:
            if 'identity_registry' not in self.contracts:
                logger.warning("Identity registry contract not available")
                return None
                
            # Placeholder for actual contract call
            # In a real implementation, this would submit a transaction
            tx_hash = self.contracts['identity_registry'].functions.storeIdentity(
                user_id, 
                identity_hash, 
                json.dumps(permissions)
            ).transact({'from': self.w3.eth.accounts[0]})
            
            return self.w3.to_hex(tx_hash)
        except Exception as e:
            logger.error(f"Error storing identity: {e}")
            return None
    
    def verify_identity(self, user_id, verifier_id, verification_type):
        """
        Trigger an identity verification process on the blockchain.
        
        Args:
            user_id (str): The user's identifier
            verifier_id (str): The verifier's identifier
            verification_type (str): The type of verification to perform
            
        Returns:
            str: Transaction hash
        """
        try:
            if 'verification' not in self.contracts:
                logger.warning("Verification contract not available")
                return None
                
            # Placeholder for actual contract call
            tx_hash = self.contracts['verification'].functions.requestVerification(
                user_id,
                verifier_id,
                verification_type
            ).transact({'from': self.w3.eth.accounts[0]})
            
            return self.w3.to_hex(tx_hash)
        except Exception as e:
            logger.error(f"Error requesting verification: {e}")
            return None
    
    def update_permissions(self, user_id, new_permissions):
        """
        Update the access permissions for a user's identity data.
        
        Args:
            user_id (str): The user's identifier
            new_permissions (dict): New access permissions
            
        Returns:
            str: Transaction hash
        """
        try:
            if 'identity_registry' not in self.contracts:
                logger.warning("Identity registry contract not available")
                return None
                
            # Placeholder for actual contract call
            tx_hash = self.contracts['identity_registry'].functions.updatePermissions(
                user_id,
                json.dumps(new_permissions)
            ).transact({'from': self.w3.eth.accounts[0]})
            
            return self.w3.to_hex(tx_hash)
        except Exception as e:
            logger.error(f"Error updating permissions: {e}")
            return None 