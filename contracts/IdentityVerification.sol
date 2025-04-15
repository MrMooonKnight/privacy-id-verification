// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IdentityVerification
 * @dev Smart contract for storing and verifying identity hashes and managing access permissions
 */
contract IdentityVerification {
    // Owner of the contract
    address public owner;
    
    // Mapping from user ID to identity hash
    mapping(bytes32 => bytes32) private identityHashes;
    
    // Mapping from user ID to access control mapping (address => expiration time)
    mapping(bytes32 => mapping(address => uint256)) private accessControl;
    
    // Events
    event IdentityHashStored(bytes32 indexed userId, address storedBy);
    event AccessGranted(bytes32 indexed userId, address indexed grantedTo, uint256 expirationTime);
    event AccessRevoked(bytes32 indexed userId, address indexed revokedFrom);
    
    // Constructor
    constructor() {
        owner = msg.sender;
    }
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyDataOwnerOrAuthorized(bytes32 userId) {
        require(
            msg.sender == owner || 
            (accessControl[userId][msg.sender] > block.timestamp),
            "Not authorized to access this identity data"
        );
        _;
    }
    
    /**
     * @dev Store hash of identity data on the blockchain
     * @param userId Unique identifier for the user
     * @param identityHash Hash of the identity data
     */
    function storeIdentityHash(bytes32 userId, bytes32 identityHash) public {
        identityHashes[userId] = identityHash;
        emit IdentityHashStored(userId, msg.sender);
    }
    
    /**
     * @dev Get hash of identity data from the blockchain
     * @param userId Unique identifier for the user
     * @return Identity hash
     */
    function getIdentityHash(bytes32 userId) 
        public 
        view 
        onlyDataOwnerOrAuthorized(userId) 
        returns (bytes32) 
    {
        return identityHashes[userId];
    }
    
    /**
     * @dev Grant access to identity data to a specific address
     * @param userId Unique identifier for the user
     * @param recipient Address to grant access to
     * @param expirationTime Unix timestamp when access expires
     */
    function grantAccess(bytes32 userId, address recipient, uint256 expirationTime) public {
        // Only owner or the user themselves can grant access
        require(msg.sender == owner, "Only owner can grant access");
        require(expirationTime > block.timestamp, "Expiration time must be in the future");
        
        accessControl[userId][recipient] = expirationTime;
        emit AccessGranted(userId, recipient, expirationTime);
    }
    
    /**
     * @dev Revoke access to identity data from a specific address
     * @param userId Unique identifier for the user
     * @param recipient Address to revoke access from
     */
    function revokeAccess(bytes32 userId, address recipient) public {
        // Only owner or the user themselves can revoke access
        require(msg.sender == owner, "Only owner can revoke access");
        
        accessControl[userId][recipient] = 0;
        emit AccessRevoked(userId, recipient);
    }
    
    /**
     * @dev Check if an address has access to identity data
     * @param userId Unique identifier for the user
     * @param recipient Address to check access for
     * @return (bool, uint256) Has access and expiration time
     */
    function checkAccess(bytes32 userId, address recipient) 
        public 
        view 
        returns (bool, uint256) 
    {
        uint256 expirationTime = accessControl[userId][recipient];
        bool hasAccess = expirationTime > block.timestamp;
        
        return (hasAccess, expirationTime);
    }
    
    /**
     * @dev Verify if an identity hash matches what's stored
     * @param userId Unique identifier for the user
     * @param identityHash Hash to verify
     * @return bool True if the hash is valid
     */
    function verifyIdentity(bytes32 userId, bytes32 identityHash) 
        public 
        view 
        onlyDataOwnerOrAuthorized(userId) 
        returns (bool) 
    {
        return identityHashes[userId] == identityHash;
    }
    
    /**
     * @dev Transfer ownership of the contract
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        owner = newOwner;
    }
} 