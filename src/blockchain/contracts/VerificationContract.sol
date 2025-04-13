// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./IdentityRegistry.sol";

/**
 * @title VerificationContract
 * @dev Smart contract for managing identity verification processes.
 */
contract VerificationContract {
    address public owner;
    IdentityRegistry public identityRegistry;
    
    enum VerificationStatus { Pending, Approved, Rejected }
    
    struct VerificationRequest {
        string userId;
        string verifierId;
        string verificationType;
        VerificationStatus status;
        uint256 requestedAt;
        uint256 updatedAt;
        string resultHash;     // Hash of verification result data
        bool exists;
    }
    
    // Mapping from request ID to verification request
    mapping(bytes32 => VerificationRequest) private verificationRequests;
    
    // Events
    event VerificationRequested(bytes32 requestId, string userId, string verifierId, string verificationType, uint256 timestamp);
    event VerificationCompleted(bytes32 requestId, VerificationStatus status, uint256 timestamp);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the contract owner can call this function");
        _;
    }
    
    modifier onlyVerifier(string memory verifierId) {
        // In a real implementation, this would check if the sender is the verifier
        // For simplicity, we're using onlyOwner for now
        require(msg.sender == owner, "Only the authorized verifier can call this function");
        _;
    }
    
    constructor(address identityRegistryAddress) {
        owner = msg.sender;
        identityRegistry = IdentityRegistry(identityRegistryAddress);
    }
    
    /**
     * @dev Generate a unique request ID
     * @param userId The user's identifier
     * @param verifierId The verifier's identifier
     * @param verificationType The type of verification
     * @return requestId The unique request ID
     */
    function generateRequestId(string memory userId, string memory verifierId, string memory verificationType) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(userId, verifierId, verificationType, block.timestamp));
    }
    
    /**
     * @dev Request verification for a user
     * @param userId The user's identifier
     * @param verifierId The verifier's identifier
     * @param verificationType The type of verification to perform
     * @return requestId The unique request ID
     */
    function requestVerification(string memory userId, string memory verifierId, string memory verificationType) public returns (bytes32) {
        // Check if the identity exists
        require(identityRegistry.identityExists(userId), "Identity does not exist");
        
        // Generate a unique request ID
        bytes32 requestId = generateRequestId(userId, verifierId, verificationType);
        
        // Create the verification request
        verificationRequests[requestId] = VerificationRequest({
            userId: userId,
            verifierId: verifierId,
            verificationType: verificationType,
            status: VerificationStatus.Pending,
            requestedAt: block.timestamp,
            updatedAt: block.timestamp,
            resultHash: "",
            exists: true
        });
        
        // Emit the event
        emit VerificationRequested(requestId, userId, verifierId, verificationType, block.timestamp);
        
        return requestId;
    }
    
    /**
     * @dev Complete a verification request
     * @param requestId The ID of the verification request
     * @param status The status of the verification (Approved or Rejected)
     * @param resultHash The hash of the verification result data
     * @return success Whether the operation was successful
     */
    function completeVerification(bytes32 requestId, VerificationStatus status, string memory resultHash) public onlyVerifier(verificationRequests[requestId].verifierId) returns (bool) {
        // Check if the request exists and is pending
        require(verificationRequests[requestId].exists, "Verification request does not exist");
        require(verificationRequests[requestId].status == VerificationStatus.Pending, "Verification request is not pending");
        
        // Update the request
        verificationRequests[requestId].status = status;
        verificationRequests[requestId].resultHash = resultHash;
        verificationRequests[requestId].updatedAt = block.timestamp;
        
        // Emit the event
        emit VerificationCompleted(requestId, status, block.timestamp);
        
        return true;
    }
    
    /**
     * @dev Get the details of a verification request
     * @param requestId The ID of the verification request
     * @return userId The user's identifier
     * @return verifierId The verifier's identifier
     * @return verificationType The type of verification
     * @return status The status of the verification
     * @return requestedAt Timestamp when the verification was requested
     * @return updatedAt Timestamp when the verification was last updated
     * @return resultHash The hash of the verification result data
     */
    function getVerificationRequest(bytes32 requestId) public view returns (
        string memory userId,
        string memory verifierId,
        string memory verificationType,
        VerificationStatus status,
        uint256 requestedAt,
        uint256 updatedAt,
        string memory resultHash
    ) {
        require(verificationRequests[requestId].exists, "Verification request does not exist");
        
        VerificationRequest memory request = verificationRequests[requestId];
        return (
            request.userId,
            request.verifierId,
            request.verificationType,
            request.status,
            request.requestedAt,
            request.updatedAt,
            request.resultHash
        );
    }
    
    /**
     * @dev Check the status of a verification request
     * @param requestId The ID of the verification request
     * @return status The status of the verification
     */
    function checkVerificationStatus(bytes32 requestId) public view returns (VerificationStatus) {
        require(verificationRequests[requestId].exists, "Verification request does not exist");
        return verificationRequests[requestId].status;
    }
    
    /**
     * @dev Transfer ownership of the contract
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be the zero address");
        owner = newOwner;
    }
    
    /**
     * @dev Set the address of the identity registry contract
     * @param identityRegistryAddress Address of the identity registry contract
     */
    function setIdentityRegistry(address identityRegistryAddress) public onlyOwner {
        identityRegistry = IdentityRegistry(identityRegistryAddress);
    }
} 