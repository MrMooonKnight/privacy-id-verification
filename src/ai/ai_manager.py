"""
AI Manager module for handling identity verification and fraud detection using machine learning models.
"""

import logging
import os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import cv2
import face_recognition

logger = logging.getLogger(__name__)

class AIManager:
    """Manages all AI and machine learning operations for identity verification and fraud detection."""
    
    def __init__(self, config):
        """
        Initialize the AIManager with the provided configuration.
        
        Args:
            config (dict): AI configuration including model paths and thresholds
        """
        self.config = config
        self.models = self._load_models()
        logger.info("AIManager initialized")
    
    def _load_models(self):
        """Load machine learning models for different verification tasks."""
        models = {}
        try:
            # Create models directory structure if it doesn't exist
            os.makedirs(os.path.dirname(self.config['facial_recognition']['model_path']), exist_ok=True)
            os.makedirs(os.path.dirname(self.config['document_verification']['model_path']), exist_ok=True)
            os.makedirs(os.path.dirname(self.config['fraud_detection']['model_path']), exist_ok=True)
            
            # Initialize fraud detection model (isolation forest for anomaly detection)
            # In a real implementation, this would load a pre-trained model
            models['fraud_detection'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            logger.info("AI models initialized")
            return models
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            return {}
    
    def verify_face(self, face_image, reference_image):
        """
        Verify a face against a reference image.
        
        Args:
            face_image (bytes): The face image to verify
            reference_image (bytes): The reference image to compare against
            
        Returns:
            dict: Verification result including match status and confidence
        """
        try:
            # Convert images to numpy arrays
            face_array = self._convert_image_to_array(face_image)
            reference_array = self._convert_image_to_array(reference_image)
            
            # Get face encodings
            face_encoding = face_recognition.face_encodings(face_array)
            reference_encoding = face_recognition.face_encodings(reference_array)
            
            if not face_encoding or not reference_encoding:
                logger.warning("No face found in one or both images")
                return {
                    "match": False,
                    "confidence": 0.0,
                    "error": "No face found in one or both images"
                }
            
            # Compare faces
            face_distances = face_recognition.face_distance(reference_encoding, face_encoding[0])
            confidence = 1 - face_distances[0]
            
            # Check if confidence exceeds threshold
            threshold = self.config['facial_recognition']['confidence_threshold']
            match = confidence >= threshold
            
            return {
                "match": match,
                "confidence": float(confidence),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error verifying face: {e}")
            return {
                "match": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def verify_document(self, document_image):
        """
        Verify a government-issued ID document.
        
        Args:
            document_image (bytes): The document image to verify
            
        Returns:
            dict: Verification result including validity and extracted information
        """
        try:
            # Convert image to numpy array
            document_array = self._convert_image_to_array(document_image)
            
            # In a real implementation, this would use OCR and document analysis
            # Here we just do some basic checks on the image dimensions
            if document_array.shape[0] < 300 or document_array.shape[1] < 300:
                logger.warning("Document image too small")
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "extracted_info": {},
                    "error": "Document image too small"
                }
            
            # Simulate document verification with random confidence
            # In a real system, this would be based on actual analysis
            confidence = np.random.uniform(0.7, 0.99)
            valid = confidence >= self.config['document_verification']['confidence_threshold']
            
            # Simulate extracted information
            extracted_info = {
                "document_type": "passport",
                "id_number": "XXXXX1234",
                "name": "REDACTED",
                "date_of_birth": "XXXX-XX-XX",
                "expiration_date": "XXXX-XX-XX"
            }
            
            return {
                "valid": valid,
                "confidence": float(confidence),
                "extracted_info": extracted_info,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error verifying document: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "extracted_info": {},
                "error": str(e)
            }
    
    def detect_fraud(self, verification_data):
        """
        Detect fraudulent verification attempts using anomaly detection.
        
        Args:
            verification_data (dict): Data about the verification attempt
            
        Returns:
            dict: Fraud detection result including risk score
        """
        try:
            if 'fraud_detection' not in self.models:
                logger.warning("Fraud detection model not available")
                return {
                    "risk_score": 0.5,
                    "is_fraudulent": None,
                    "error": "Fraud detection model not available"
                }
            
            # Extract features from verification data
            # In a real system, this would be based on actual verification data
            features = np.array([
                verification_data.get("ip_reputation", 0.5),
                verification_data.get("geo_velocity", 0.5),
                verification_data.get("device_reputation", 0.5),
                verification_data.get("behavioral_score", 0.5)
            ]).reshape(1, -1)
            
            # Predict anomaly score
            anomaly_score = self.models['fraud_detection'].decision_function(features)[0]
            # Convert to a risk score between 0 and 1
            risk_score = 1 - (anomaly_score - self.models['fraud_detection'].offset_) / 0.5
            risk_score = max(0, min(1, risk_score))
            
            # Check if risk score exceeds threshold
            threshold = self.config['fraud_detection']['anomaly_threshold']
            is_fraudulent = risk_score >= threshold
            
            return {
                "risk_score": float(risk_score),
                "is_fraudulent": is_fraudulent,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            return {
                "risk_score": 0.5,
                "is_fraudulent": None,
                "error": str(e)
            }
    
    def _convert_image_to_array(self, image_data):
        """
        Convert image data to a numpy array.
        
        Args:
            image_data (bytes): The image data
            
        Returns:
            numpy.ndarray: The image as a numpy array
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert from BGR to RGB (face_recognition uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def train_fraud_detection_model(self, training_data):
        """
        Train the fraud detection model with new data.
        
        Args:
            training_data (list): List of feature vectors for training
            
        Returns:
            bool: Whether the training was successful
        """
        try:
            if 'fraud_detection' not in self.models:
                logger.warning("Fraud detection model not available")
                return False
            
            # Convert training data to numpy array
            X = np.array(training_data)
            
            # Train the model
            self.models['fraud_detection'].fit(X)
            
            # Save the model
            # In a real implementation, this would save the model to disk
            logger.info("Fraud detection model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training fraud detection model: {e}")
            return False 