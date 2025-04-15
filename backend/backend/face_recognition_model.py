import os
import cv2
import numpy as np
import face_recognition
from PIL import Image
import pickle
from dotenv import load_dotenv

load_dotenv()

class FaceRecognitionModel:
    """Face recognition model for identity verification."""
    
    def __init__(self, model_path=None):
        """
        Initialize face recognition model.
        
        Args:
            model_path: Path to saved model data (if any)
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_database = {}
        
        # Load saved model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: Image as numpy array or file path
            
        Returns:
            List of face locations in the image
        """
        if isinstance(image, str):
            # Load image from file path
            image = face_recognition.load_image_file(image)
        
        # Find all faces in the image
        face_locations = face_recognition.face_locations(image, model="hog")
        
        return face_locations
    
    def extract_face_encodings(self, image, face_locations=None):
        """
        Extract face encodings from detected faces.
        
        Args:
            image: Image as numpy array or file path
            face_locations: Optional pre-detected face locations
            
        Returns:
            List of face encodings
        """
        if isinstance(image, str):
            # Load image from file path
            image = face_recognition.load_image_file(image)
        
        if face_locations is None:
            face_locations = self.detect_faces(image)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        return face_encodings
    
    def register_face(self, user_id, image):
        """
        Register a face for a user.
        
        Args:
            user_id: Unique identifier for the user
            image: Image as numpy array or file path
            
        Returns:
            success: Boolean indicating if registration was successful
            message: Message describing the result
        """
        try:
            if isinstance(image, str):
                # Load image from file path
                image = face_recognition.load_image_file(image)
            
            # Detect faces in the image
            face_locations = self.detect_faces(image)
            
            if not face_locations:
                return False, "No face detected in the image"
            
            if len(face_locations) > 1:
                return False, "Multiple faces detected. Please provide an image with a single face."
            
            # Extract face encoding
            face_encodings = self.extract_face_encodings(image, face_locations)
            
            if not face_encodings:
                return False, "Failed to extract facial features"
            
            # Store the face encoding
            self.face_database[user_id] = {
                'encoding': face_encodings[0],
                'face_location': face_locations[0]
            }
            
            # Update lists for faster lookup
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(user_id)
            
            return True, "Face registered successfully"
        
        except Exception as e:
            return False, f"Error registering face: {str(e)}"
    
    def verify_face(self, user_id, image, tolerance=0.6):
        """
        Verify a face against a registered user.
        
        Args:
            user_id: Unique identifier for the user
            image: Image as numpy array or file path
            tolerance: How much distance between faces to consider it a match (lower is stricter)
            
        Returns:
            is_match: Boolean indicating if the face matches
            confidence: Confidence score of the match
            message: Message describing the result
        """
        try:
            if isinstance(image, str):
                # Load image from file path
                image = face_recognition.load_image_file(image)
            
            # Check if user is registered
            if user_id not in self.face_database:
                return False, 0.0, "User not registered"
            
            # Detect faces in the image
            face_locations = self.detect_faces(image)
            
            if not face_locations:
                return False, 0.0, "No face detected in the verification image"
            
            if len(face_locations) > 1:
                return False, 0.0, "Multiple faces detected. Please provide an image with a single face."
            
            # Extract face encoding
            face_encodings = self.extract_face_encodings(image, face_locations)
            
            if not face_encodings:
                return False, 0.0, "Failed to extract facial features"
            
            # Get the registered face encoding
            registered_encoding = self.face_database[user_id]['encoding']
            
            # Calculate face distance (lower distance = better match)
            face_distance = face_recognition.face_distance([registered_encoding], face_encodings[0])[0]
            
            # Convert distance to confidence score (0-1)
            confidence = 1.0 - face_distance
            
            # Check if the face matches
            is_match = face_distance <= tolerance
            
            if is_match:
                return True, confidence, "Face verification successful"
            else:
                return False, confidence, "Face verification failed"
        
        except Exception as e:
            return False, 0.0, f"Error verifying face: {str(e)}"
    
    def identify_face(self, image, tolerance=0.6):
        """
        Identify a face in an image from registered users.
        
        Args:
            image: Image as numpy array or file path
            tolerance: How much distance between faces to consider it a match
            
        Returns:
            matched_users: List of (user_id, confidence) tuples for matched users
            message: Message describing the result
        """
        try:
            if isinstance(image, str):
                # Load image from file path
                image = face_recognition.load_image_file(image)
            
            # Detect faces in the image
            face_locations = self.detect_faces(image)
            
            if not face_locations:
                return [], "No face detected in the image"
            
            matched_users = []
            
            # Process each detected face
            for i, face_location in enumerate(face_locations):
                # Extract face encoding
                face_encodings = self.extract_face_encodings(image, [face_location])
                
                if not face_encodings:
                    continue
                
                face_encoding = face_encodings[0]
                
                # Compare with known faces
                if not self.known_face_encodings:
                    return [], "No users registered in the system"
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                # Get matches
                for j, distance in enumerate(face_distances):
                    if distance <= tolerance:
                        user_id = self.known_face_names[j]
                        confidence = 1.0 - distance
                        matched_users.append((user_id, confidence))
            
            if matched_users:
                # Sort by confidence (highest first)
                matched_users.sort(key=lambda x: x[1], reverse=True)
                return matched_users, "Face identification successful"
            else:
                return [], "No matching users found"
        
        except Exception as e:
            return [], f"Error identifying face: {str(e)}"
    
    def save_model(self, model_path):
        """
        Save the face recognition model to a file.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            success: Boolean indicating if saving was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model data
            model_data = {
                'face_database': self.face_database,
                'known_face_encodings': self.known_face_encodings,
                'known_face_names': self.known_face_names
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """
        Load a saved face recognition model from a file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            success: Boolean indicating if loading was successful
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.face_database = model_data['face_database']
            self.known_face_encodings = model_data['known_face_encodings']
            self.known_face_names = model_data['known_face_names']
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def extract_face_image(self, image, face_location=None):
        """
        Extract a face from an image based on its location.
        
        Args:
            image: Image as numpy array or file path
            face_location: Location of the face to extract
            
        Returns:
            face_image: Extracted face image
        """
        if isinstance(image, str):
            # Load image from file path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if face_location is None:
            # Detect faces if location not provided
            face_locations = self.detect_faces(image)
            if not face_locations:
                return None
            face_location = face_locations[0]
        
        # Extract face from image
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        
        return face_image
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for face recognition.
        
        Args:
            image_path: Path to the image
            
        Returns:
            processed_image: Preprocessed image
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Convert to RGB (face_recognition uses RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image 