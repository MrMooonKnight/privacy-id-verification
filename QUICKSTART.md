# Quick Start Guide

This guide will help you set up and run the Blockchain-Based AI for Privacy-Preserving Identity Verification system.

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- Access to an Ethereum network (can be local, testnet, or mainnet)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blockchain-identity-verification.git
   cd blockchain-identity-verification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```bash
   cp config.example.yml config.yml
   ```
   
5. Edit `config.yml` to match your environment:
   - Set your blockchain provider URL
   - Configure AI model paths
   - Set security parameters

## Running the System

Start the API server:
```bash
python src/main.py
```

The API will be available at `http://localhost:8000`.

## Using the API

### Authentication

Get an access token:
```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user&password=password"
```

### Register an Identity

```bash
curl -X POST "http://localhost:8000/identity/register" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_data": {
      "first_name": "John",
      "last_name": "Doe",
      "email": "john.doe@example.com",
      "date_of_birth": "1990-01-01",
      "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "country": "US"
      }
    }
  }'
```

### Upload Biometric Data

```bash
curl -X POST "http://localhost:8000/identity/upload-biometric" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "user_id=USER_ID" \
  -F "biometric_type=facial" \
  -F "biometric_file=@/path/to/face.jpg"
```

### Verify Identity

```bash
curl -X POST "http://localhost:8000/verification/verify-face" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "user_id=USER_ID" \
  -F "reference_image=@/path/to/reference.jpg" \
  -F "verification_image=@/path/to/verify.jpg"
```

## API Documentation

The API documentation is available at `http://localhost:8000/docs` when the server is running.

## Troubleshooting

- **Blockchain Connection Issues**: Ensure your blockchain node URL is correct and accessible.
- **File Upload Problems**: Check that file paths are correct and files are readable.
- **Authentication Errors**: Verify your token is valid and not expired.

For more detailed information, check the logs in the `logs` directory. 