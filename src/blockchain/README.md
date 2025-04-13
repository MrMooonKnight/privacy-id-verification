# Blockchain Module

This module manages all interactions with the blockchain, handling identity data storage and verification through smart contracts.

## Components

- **BlockchainManager**: Main class that manages connections to the blockchain and interactions with smart contracts.
- **Smart Contracts**:
  - **IdentityRegistry.sol**: Smart contract for storing and managing identity data hashes and permissions.
  - **VerificationContract.sol**: Smart contract for handling identity verification requests and results.

## Features

- Secure storage of identity data hashes on the blockchain
- Immutable verification records
- User-controlled access permissions
- Decentralized identity management

## Implementation Details

The blockchain module connects to the Ethereum network (or other compatible blockchains) using Web3.py. It interacts with deployed smart contracts to store identity hashes and verification results.

Identity data itself is not stored on the blockchain, only cryptographic hashes of the data. This ensures privacy while also leveraging the immutability of blockchain for verification.

## Smart Contract Architecture

1. **IdentityRegistry**: Maintains a mapping of user IDs to identity data hashes and permissions.
2. **VerificationContract**: Handles verification requests and manages the verification process. 