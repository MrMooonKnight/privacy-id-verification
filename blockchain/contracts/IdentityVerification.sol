// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract IdentityVerification {
    // Struct to store identity hash and access control
    struct Identity {
        string identityHash;
        mapping(address => bool) accessControl;
        bool exists;
    }
    
    // Mapping from user address to their identity information
    mapping(address => Identity) private identities;
    
    // Events
    event IdentityStored(address indexed user);
    event AccessGranted(address indexed user, address indexed thirdParty);
    event AccessRevoked(address indexed user, address indexed thirdParty);
    event IdentityVerified(address indexed user, address indexed verifier, bool success);
    
    /**
     * @dev Store the hash of a user's identity
     * @param _identityHash The hash of the user's identity data
     */
    function storeIdentityHash(string memory _identityHash) public {
        require(bytes(_identityHash).length > 0, "Identity hash cannot be empty");
        
        // If this is a new identity, initialize the mapping
        if (!identities[msg.sender].exists) {
            identities[msg.sender].exists = true;
        }
        
        identities[msg.sender].identityHash = _identityHash;
        emit IdentityStored(msg.sender);
    }
    
    /**
     * @dev Get a user's identity hash (only callable by the user or authorized parties)
     * @return The identity hash
     */
    function getIdentityHash() public view returns (string memory) {
        require(identities[msg.sender].exists, "Identity does not exist");
        return identities[msg.sender].identityHash;
    }
    
    /**
     * @dev Get a user's identity hash (by authorized third parties)
     * @param _user The address of the user whose identity is being checked
     * @return The identity hash
     */
    function verifyIdentity(address _user) public view returns (string memory) {
        require(identities[_user].exists, "Identity does not exist");
        require(identities[_user].accessControl[msg.sender] || msg.sender == _user, "Not authorized");
        return identities[_user].identityHash;
    }
    
    /**
     * @dev Grant access to a third party to view the user's identity hash
     * @param _thirdParty The address of the third party
     */
    function grantAccess(address _thirdParty) public {
        require(identities[msg.sender].exists, "Identity does not exist");
        identities[msg.sender].accessControl[_thirdParty] = true;
        emit AccessGranted(msg.sender, _thirdParty);
    }
    
    /**
     * @dev Revoke access from a third party
     * @param _thirdParty The address of the third party
     */
    function revokeAccess(address _thirdParty) public {
        require(identities[msg.sender].exists, "Identity does not exist");
        identities[msg.sender].accessControl[_thirdParty] = false;
        emit AccessRevoked(msg.sender, _thirdParty);
    }
    
    /**
     * @dev Check if a third party has access to a user's identity
     * @param _thirdParty The address of the third party
     * @return Whether the third party has access
     */
    function checkAccess(address _thirdParty) public view returns (bool) {
        require(identities[msg.sender].exists, "Identity does not exist");
        return identities[msg.sender].accessControl[_thirdParty];
    }
} 