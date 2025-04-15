import os
import json
import numpy as np
import face_recognition
import time
import cv2
from pathlib import Path

class FaceRecognitionModel:
    def __init__(self):
        """
        Initialize the face recognition model using the pre-trained model from face_recognition library.
        This uses dlib's pre-trained facial recognition models which have 99.38% accuracy on the LFW dataset.
        """
        self.model_path = os.path.join(os.getenv('FACE_RECOGNITION_MODEL', 'ai_models/face_recognition'))
        self.model_ready = True
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        print(f"Initialized Face Recognition Model using pre-trained models")
        print(f"Model storage path: {self.model_path}")
        
        # Map of user IDs to their face encodings
        self.user_encodings = {}
        
        # Load any existing encodings
        self._load_stored_encodings()
        
    def _load_stored_encodings(self):
        """Load stored face encodings from the model directory."""
        encodings_file = os.path.join(self.model_path, 'face_encodings.json')
        
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'r') as f:
                    encodings_data = json.load(f)
                
                for user_id, encodings in encodings_data.items():
                    # Convert back from list to numpy array
                    self.user_encodings[user_id] = [np.array(encoding) for encoding in encodings]
                
                print(f"Loaded face encodings for {len(self.user_encodings)} users")
            except Exception as e:
                print(f"Error loading stored encodings: {str(e)}")
        else:
            print("No stored encodings found. Starting with empty encodings.")
    
    def _save_encodings(self):
        """Save face encodings to disk."""
        encodings_file = os.path.join(self.model_path, 'face_encodings.json')
        
        # Convert numpy arrays to lists for JSON serialization
        encodings_data = {
            user_id: [encoding.tolist() for encoding in encodings]
            for user_id, encodings in self.user_encodings.items()
        }
        
        try:
            with open(encodings_file, 'w') as f:
                json.dump(encodings_data, f)
            print(f"Saved face encodings for {len(self.user_encodings)} users")
        except Exception as e:
            print(f"Error saving encodings: {str(e)}")
    
    def register_face(self, image_data, user_id):
        """
        Register a user's face by extracting and storing facial encodings.
        
        Args:
            image_data: Base64 encoded image or path to image file
            user_id: Unique identifier for the user
            
        Returns:
            dict: Registration result with status and details
        """
        try:
            # Load image from file path or convert from base64
            if isinstance(image_data, str) and (os.path.exists(image_data) or Path(image_data).exists()):
                image = face_recognition.load_image_file(image_data)
            elif isinstance(image_data, str) and image_data.startswith('data:image'):
                # Handle base64 image data
                import base64
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Assume it's already a numpy array
                image = image_data
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return {
                    "success": False,
                    "message": "No face detected in the image",
                    "details": {
                        "user_id": user_id,
                        "face_count": 0
                    }
                }
            
            # Generate encodings for all detected faces
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Store the encodings for this user
            self.user_encodings[user_id] = face_encodings
            
            # Save updated encodings to disk
            self._save_encodings()
            
            return {
                "success": True,
                "message": f"Successfully registered {len(face_encodings)} face(s) for user {user_id}",
                "details": {
                    "user_id": user_id,
                    "face_count": len(face_encodings),
                    "face_locations": [
                        {"top": top, "right": right, "bottom": bottom, "left": left}
                        for top, right, bottom, left in face_locations
                    ]
                }
            }
            
        except Exception as e:
            print(f"Error registering face: {str(e)}")
            return {
                "success": False,
                "message": f"Error registering face: {str(e)}",
                "details": {
                    "user_id": user_id
                }
            }
    
    def verify_face(self, image_data, user_id, tolerance=0.6):
        """
        Verify a face against a registered user.
        
        Args:
            image_data: Base64 encoded image or path to image file
            user_id: User ID to verify against
            tolerance: Matching tolerance (lower is stricter)
            
        Returns:
            dict: Verification result with match status and confidence
        """
        try:
            start_time = time.time()
            
            # Check if user has registered faces
            if user_id not in self.user_encodings or not self.user_encodings[user_id]:
                return {
                    "success": False,
                    "matched": False,
                    "message": f"No registered faces found for user {user_id}",
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Load image from file path or convert from base64
            if isinstance(image_data, str) and (os.path.exists(image_data) or Path(image_data).exists()):
                image = face_recognition.load_image_file(image_data)
            elif isinstance(image_data, str) and image_data.startswith('data:image'):
                # Handle base64 image data
                import base64
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Assume it's already a numpy array
                image = image_data
            
            # Detect faces in the verification image
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return {
                    "success": False,
                    "matched": False,
                    "message": "No face detected in the verification image",
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Get encodings for the detected faces
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Compare against the registered faces for this user
            registered_encodings = self.user_encodings[user_id]
            
            best_match = 0.0
            for verify_encoding in face_encodings:
                for reg_encoding in registered_encodings:
                    # Calculate face distance (lower is better)
                    face_distance = face_recognition.face_distance([reg_encoding], verify_encoding)[0]
                    # Convert to similarity score (higher is better)
                    similarity = 1.0 - min(face_distance, 1.0)
                    best_match = max(best_match, similarity)
            
            # Determine if it's a match based on tolerance
            is_match = best_match >= (1.0 - tolerance)
            
            return {
                "success": True,
                "matched": is_match,
                "message": "Face verification completed",
                "confidence": float(best_match),
                "processing_time": time.time() - start_time,
                "details": {
                    "user_id": user_id,
                    "face_count": len(face_locations),
                    "threshold": 1.0 - tolerance,
                    "face_locations": [
                        {"top": top, "right": right, "bottom": bottom, "left": left}
                        for top, right, bottom, left in face_locations
                    ]
                }
            }
            
        except Exception as e:
            print(f"Error verifying face: {str(e)}")
            return {
                "success": False,
                "matched": False,
                "message": f"Error verifying face: {str(e)}",
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
    
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
                if not self.user_encodings:
                    return [], "No users registered in the system"
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(list(self.user_encodings.values()), face_encoding)
                
                # Get matches
                for j, distance in enumerate(face_distances):
                    if distance <= tolerance:
                        user_id = list(self.user_encodings.keys())[j]
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
                'user_encodings': self.user_encodings
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """
        Load a saved face recognition model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            # Face recognition primarily uses the saved encodings
            encodings_file = os.path.join(model_path, 'face_encodings.json')
            
            if os.path.exists(encodings_file):
                with open(encodings_file, 'r') as f:
                    encodings_data = json.load(f)
                
                for user_id, encodings in encodings_data.items():
                    # Convert back from list to numpy array
                    self.user_encodings[user_id] = [np.array(encoding) for encoding in encodings]
                
                self.model_path = model_path
                print(f"Loaded face recognition model with {len(self.user_encodings)} users")
                return True
            else:
                print(f"No valid encodings file found at {encodings_file}")
                return False
            
        except Exception as e:
            print(f"Error loading face recognition model: {str(e)}")
            return False
    
    def get_face_hash(self, user_id):
        """
        Generate a secure hash of a user's face encodings.
        This provides a way to reference the biometric data without exposing it.
        
        Args:
            user_id: ID of the user
        
        Returns:
            str: Secure hash of the face encodings or None if user not found
        """
        if user_id not in self.user_encodings or not self.user_encodings[user_id]:
            return None
            
        # Create a string representation of the face encodings
        face_data = []
        for encoding in self.user_encodings[user_id]:
            # Convert numpy array to list and round to reduce sensitivity
            face_data.append([round(float(x), 5) for x in encoding.tolist()])
            
        # Convert to JSON string for consistent hashing
        face_json = json.dumps(face_data, sort_keys=True)
        
        # Generate a secure hash
        import hashlib
        face_hash = hashlib.sha256(face_json.encode()).hexdigest()
        
        return face_hash
    
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