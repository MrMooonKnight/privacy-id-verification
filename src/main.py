#!/usr/bin/env python3
"""
Main entry point for the Blockchain-Based AI for Privacy-Preserving Identity Verification system.
"""

import os
import logging
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import modules using relative imports
from src.blockchain.blockchain_manager import BlockchainManager
from src.identity.identity_manager import IdentityManager
from src.ai.ai_manager import AIManager
from src.crypto.crypto_manager import CryptoManager
from src.interface.api_manager import APIManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found.")
        logger.info("Please copy config.example.yml to config.yml and update it.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        exit(1)

def create_app(config):
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Blockchain Identity Verification",
        description="Privacy-Preserving Identity Verification System using Blockchain and AI",
        version="0.1.0",
        docs_url="/docs" if config['api']['enable_docs'] else None
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config['api']['cors_origins'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize managers
    blockchain_manager = BlockchainManager(config['blockchain'])
    crypto_manager = CryptoManager(config['crypto'])
    ai_manager = AIManager(config['ai'])
    identity_manager = IdentityManager(blockchain_manager, crypto_manager, ai_manager, config['compliance'])
    api_manager = APIManager(app, identity_manager, config['security'])

    # Include routers
    api_manager.include_routers()

    return app

def main():
    """Main function to start the application."""
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)

    # Load configuration
    config = load_config()

    # Create the FastAPI app
    app = create_app(config)

    # Run the server
    logger.info(f"Starting server on {config['api']['host']}:{config['api']['port']}")
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        log_level="info"
    )

if __name__ == "__main__":
    main() 