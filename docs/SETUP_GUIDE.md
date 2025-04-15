# Setup Guide: Blockchain-Based AI for Privacy-Preserving Identity Verification

This guide will help you set up and run the complete identity verification system.

## Prerequisites

- Python 3.8+
- Node.js 14+
- MongoDB installed and running
- Ethereum development environment (Ganache for local development)
- Webcam for face verification (optional, but recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd blockchain-ai-identity
```

### 2. Set Up Backend

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Set Up Smart Contracts

```bash
# Start a local Ethereum blockchain (Ganache)
ganache-cli

# Deploy smart contracts
cd blockchain
truffle migrate --reset
# Note the contract address and update it in your .env file
```

### 4. Set Up Frontend

```bash
# Install frontend dependencies
cd frontend
npm install

# Configure environment variables
cp .env.example .env
# Set REACT_APP_API_URL to point to your backend API
```

## Running the System

### 1. Start Backend Server

```bash
# From the project root directory
cd backend
python app.py
```

The backend API will be available at http://localhost:5000.

### 2. Start Frontend Development Server

```bash
# From the project root directory
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000.

## Testing the System

### 1. Register a New User

1. Navigate to http://localhost:3000/register
2. Upload a clear facial image
3. Upload a government-issued ID document
4. Fill in the required information
5. Submit the registration form

### 2. Verify Identity

1. Navigate to http://localhost:3000/verify
2. Enter your user ID
3. Upload a facial image for verification
4. The system will verify your identity

### 3. Manage Access Control

1. Log in to the system
2. Navigate to the Access Control page
3. Grant access to a third party by entering their Ethereum address
4. Revoke access when no longer needed

## System Architecture

The system consists of several components:

1. **Backend API**: Flask-based REST API that handles identity verification requests.
2. **Frontend**: React-based user interface.
3. **Blockchain**: Ethereum smart contracts for storing identity hashes and managing access control.
4. **AI Models**:
   - Face recognition for biometric verification
   - Document verification for ID validation
   - Fraud detection for security

## Security Considerations

- All sensitive data is encrypted using AES-256-GCM encryption
- Only identity hashes are stored on the blockchain, not the actual data
- Zero-knowledge proofs allow verification without revealing sensitive information
- User controls access to their data through blockchain-based permissions

## Troubleshooting

### Common Issues

1. **Backend fails to start**:
   - Check MongoDB connection
   - Verify environment variables are set correctly
   - Ensure required Python packages are installed

2. **Frontend fails to connect to backend**:
   - Check CORS settings
   - Verify API URL is correct in frontend .env file

3. **Smart contract deployment issues**:
   - Ensure Ganache is running
   - Check Truffle configuration
   - Verify you have enough ETH in your deployment account

4. **Face recognition issues**:
   - Ensure proper lighting for face verification
   - Use a clear, front-facing image
   - Check webcam permissions if using a camera

## Performance Tuning

For production deployment, consider the following optimizations:

1. Use a production-grade web server (Gunicorn, uWSGI)
2. Configure proper caching for API responses
3. Optimize database queries and create appropriate indexes
4. Use a production Ethereum node or service (Infura, Alchemy)
5. Implement rate limiting to prevent abuse

## Compliance

This system is designed to comply with:
- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- Industry standards for identity verification (W3C DID) 