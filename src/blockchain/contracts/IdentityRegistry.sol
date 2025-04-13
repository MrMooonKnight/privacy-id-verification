// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IdentityRegistry
 * @dev Smart contract for storing and managing identity data on the blockchain.
 */
contract IdentityRegistry {
    address public owner;
    
    struct Identity {
        string dataHash;       // Hash of the identity data (stored off-chain)
        string permissions;    // JSON string of permissions
        bool exists;           // Flag to check if identity exists
        uint256 createdAt;     // Timestamp of identity creation
        uint256 updatedAt;     // Timestamp of last update
    }
    
    // Mapping from user IDs to their identity data
    mapping(string => Identity) private identities;
    
    // Events
    event IdentityCreated(string userId, uint256 timestamp);
    event IdentityUpdated(string userId, uint256 timestamp);
    event PermissionsUpdated(string userId, uint256 timestamp);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the contract owner can call this function");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Store a new identity or update an existing one
     * @param userId The unique identifier for the user
     * @param dataHash The hash of the user's identity data
     * @param permissions JSON string containing access permissions
     */
    function storeIdentity(string memory userId, string memory dataHash, string memory permissions) public onlyOwner returns (bool) {
        uint256 timestamp = block.timestamp;
        
        if (identities[userId].exists) {
            // Update existing identity
            identities[userId].dataHash = dataHash;
            identities[userId].permissions = permissions;
            identities[userId].updatedAt = timestamp;
            
            emit IdentityUpdated(userId, timestamp);
        } else {
            // Create new identity
            identities[userId] = Identity({
                dataHash: dataHash,
                permissions: permissions,
                exists: true,
                createdAt: timestamp,
                updatedAt: timestamp
            });
            
            emit IdentityCreated(userId, timestamp);
        }
        
        return true;
    }
    
    /**
     * @dev Retrieve identity data for a given user
     * @param userId The unique identifier for the user
     * @return dataHash The hash of the user's identity data
     * @return permissions JSON string containing access permissions
     * @return createdAt Timestamp when the identity was created
     * @return updatedAt Timestamp when the identity was last updated
     */
    function getIdentity(string memory userId) public view returns (
        string memory dataHash,
        string memory permissions,
        uint256 createdAt,
        uint256 updatedAt
    ) {
        require(identities[userId].exists, "Identity does not exist");
        
        Identity memory identity = identities[userId];
        return (
            identity.dataHash,
            identity.permissions,
            identity.createdAt,
            identity.updatedAt
        );
    }
    
    /**
     * @dev Update permissions for a given user's identity
     * @param userId The unique identifier for the user
     * @param newPermissions JSON string containing new access permissions
     */
    function updatePermissions(string memory userId, string memory newPermissions) public onlyOwner returns (bool) {
        require(identities[userId].exists, "Identity does not exist");
        
        identities[userId].permissions = newPermissions;
        identities[userId].updatedAt = block.timestamp;
        
        emit PermissionsUpdated(userId, block.timestamp);
        
        return true;
    }
    
    /**
     * @dev Check if an identity exists for a given user
     * @param userId The unique identifier for the user
     * @return exists Whether the identity exists
     */
    function identityExists(string memory userId) public view returns (bool exists) {
        return identities[userId].exists;
    }
    
    /**
     * @dev Transfer ownership of the contract
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be the zero address");
        owner = newOwner;
    }
} 