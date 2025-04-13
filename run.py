#!/usr/bin/env python3
"""
Run script for the Blockchain-Based AI Identity Verification System.
"""

import os
import sys
import subprocess
import argparse
import logging

# Add the current directory to Python path
sys.path.append(os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is set up correctly."""
    # Check if config.yml exists
    if not os.path.exists('config.yml'):
        logger.warning("config.yml not found. Copying from example...")
        if os.path.exists('config.example.yml'):
            import shutil
            shutil.copy('config.example.yml', 'config.yml')
            logger.info("Created config.yml from example. Please edit it with your settings.")
        else:
            logger.error("config.example.yml not found. Cannot create config.yml.")
            return False
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/identity', exist_ok=True)
    os.makedirs('keys', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required.")
        return False
    
    return True

def run_server(debug=False):
    """Run the identity verification server."""
    logger.info("Starting the identity verification server...")
    
    # Run the server
    try:
        from src.main import main
        main()
    except ImportError as e:
        logger.error(f"Failed to import: {e}")
        logger.info("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False
    
    return True

def run_tests():
    """Run the test suite."""
    logger.info("Running tests...")
    
    # Run tests with unittest
    try:
        import unittest
        tests = unittest.defaultTestLoader.discover('tests')
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(tests)
        
        if result.wasSuccessful():
            logger.info("All tests passed!")
            return True
        else:
            logger.error("Some tests failed.")
            return False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def main():
    """Main function to run the system."""
    parser = argparse.ArgumentParser(description='Run the Blockchain-Based AI Identity Verification System')
    parser.add_argument('--test', action='store_true', help='Run the test suite')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run tests if requested
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    
    # Run server
    success = run_server(debug=args.debug)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 