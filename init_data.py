#!/usr/bin/env python3
"""
Initialize data directories and sample data for the Blockchain-Based AI Identity Verification System.
"""

import os
import logging
import shutil
import yaml
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create the necessary directories for the system."""
    directories = [
        'logs',
        'data',
        'data/identity',
        'keys',
        'keys/zkp_proving_key',
        'keys/zkp_verification_key',
        'keys/homomorphic_key',
        'models',
        'models/facial_recognition',
        'models/document_verification',
        'models/fraud_detection'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def initialize_config():
    """Initialize the configuration file."""
    if not os.path.exists('config.yml'):
        if os.path.exists('config.example.yml'):
            shutil.copy('config.example.yml', 'config.yml')
            logger.info("Created config.yml from example. Please edit it with your settings.")
        else:
            logger.error("config.example.yml not found. Cannot create config.yml.")
            return False
    else:
        logger.info("config.yml already exists. Skipping.")
    
    return True

def create_placeholder_files():
    """Create placeholder files for the system."""
    placeholder_files = [
        'logs/app.log',
        'models/facial_recognition/placeholder.txt',
        'models/document_verification/placeholder.txt',
        'models/fraud_detection/placeholder.txt',
        'keys/zkp_proving_key/placeholder.txt',
        'keys/zkp_verification_key/placeholder.txt',
        'keys/homomorphic_key/placeholder.txt'
    ]
    
    for file_path in placeholder_files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("This is a placeholder file created by init_data.py\n")
            logger.info(f"Created placeholder file: {file_path}")

def main():
    """Main function to initialize the system."""
    parser = argparse.ArgumentParser(description='Initialize data directories for the system')
    parser.add_argument('--force', action='store_true', help='Force reinitialization of config.yml')
    args = parser.parse_args()
    
    logger.info("Initializing system data directories...")
    
    # Create directories
    create_directories()
    
    # Initialize config
    if args.force and os.path.exists('config.yml'):
        os.rename('config.yml', 'config.yml.backup')
        logger.info("Backed up existing config.yml to config.yml.backup")
    
    if not initialize_config():
        logger.error("Failed to initialize configuration. Exiting.")
        return False
    
    # Create placeholder files
    create_placeholder_files()
    
    logger.info("Initialization completed successfully.")
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 