import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import json
import time
import cv2
from pathlib import Path

load_dotenv()

class FraudDetectionModel:
    """Fraud detection model for identity verification."""
    
    def __init__(self, model_path=None):
        """
        Initialize fraud detection model.
        
        Args:
            model_path: Path to saved model (if any)
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            # User behavior features
            'login_frequency', 'avg_session_duration', 'device_change_frequency',
            'location_change_frequency', 'unusual_time_access', 'failed_login_attempts',
            
            # Identity document features
            'doc_tampering_score', 'doc_consistency_score', 'doc_pattern_match_score',
            'doc_font_consistency', 'doc_micro_feature_score',
            
            # Biometric features 
            'biometric_confidence', 'biometric_liveness_score', 'biometric_consistency',
            
            # Network and device features
            'ip_reputation_score', 'vpn_proxy_score', 'device_integrity_score',
            'browser_integrity_score', 'connection_anomaly_score'
        ]
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Create a new model
            self._create_model()
            self.scaler = StandardScaler()
        
        # Initialize the fraud detection model using pre-trained MobileNetV2
        self.model_path = os.path.join(os.getenv('FRAUD_DETECTION_MODEL', 'ai_models/fraud_detection'))
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load pre-trained MobileNetV2 model
        self.model = self._load_pretrained_model()
        
        # Keep track of detected fraud patterns
        self.fraud_patterns = self._load_fraud_patterns()
        
        print(f"Initialized Fraud Detection Model using pre-trained MobileNetV2")
        print(f"Model storage path: {self.model_path}")
    
    def _create_model(self):
        """Create a new fraud detection model."""
        # Number of features
        n_features = len(self.feature_names)
        
        # Define model architecture
        inputs = Input(shape=(n_features,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Compile model
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_data(self, features):
        """
        Preprocess input data for model prediction.
        
        Args:
            features: Dictionary or list of dictionaries with feature values
            
        Returns:
            X: Preprocessed feature array
        """
        # Convert single instance to list
        if isinstance(features, dict):
            features = [features]
        
        # Extract features in the correct order
        X = np.zeros((len(features), len(self.feature_names)))
        for i, instance in enumerate(features):
            for j, feature_name in enumerate(self.feature_names):
                X[i, j] = instance.get(feature_name, 0.0)
        
        # Scale features if scaler is fitted
        if hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        
        return X
    
    def detect_fraud(self, features):
        """
        Detect potential fraud in identity verification.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            is_fraudulent: Boolean indicating if fraud is detected
            fraud_score: Probability of fraud (0-1)
            risk_factors: List of high-risk factors
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model is not trained. Please train or load a model first.")
        
        # Preprocess features
        X = self.preprocess_data(features)
        
        # Get fraud probability
        fraud_score = float(self.model.predict(X)[0][0])
        
        # Determine if fraudulent based on threshold
        is_fraudulent = fraud_score >= 0.7  # Adjust threshold as needed
        
        # Identify risk factors (features with high contribution to fraud score)
        risk_factors = []
        if is_fraudulent:
            # Get feature importance by making predictions with one feature zeroed out
            feature_importances = []
            base_score = fraud_score
            
            for i in range(len(self.feature_names)):
                # Create a copy with one feature zeroed
                X_modified = X.copy()
                X_modified[0, i] = 0
                
                # Predict with modified features
                modified_score = float(self.model.predict(X_modified)[0][0])
                
                # Calculate importance as the difference in prediction
                importance = base_score - modified_score
                feature_importances.append((self.feature_names[i], importance))
            
            # Sort by importance (highest first)
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            
            # Get top risk factors (positive importance means feature contributes to fraud)
            risk_factors = [f[0] for f in feature_importances if f[1] > 0][:5]
        
        return is_fraudulent, fraud_score, risk_factors
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the fraud detection model.
        
        Args:
            X: Feature matrix or list of feature dictionaries
            y: Target labels (1 for fraud, 0 for legitimate)
            validation_split: Portion of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            history: Training history
        """
        # Preprocess input data if it's a list of dictionaries
        if isinstance(X, list) and isinstance(X[0], dict):
            X = self.preprocess_data(X)
        
        # Fit the scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train the model
        history = self.model.fit(
            X_scaled, np.array(y),
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate the fraud detection model.
        
        Args:
            X: Feature matrix or list of feature dictionaries
            y: True labels
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Preprocess input data if it's a list of dictionaries
        if isinstance(X, list) and isinstance(X[0], dict):
            X = self.preprocess_data(X)
        else:
            X = self.scaler.transform(X)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X, np.array(y), verbose=0)
        
        # Make predictions
        y_pred_prob = self.model.predict(X).flatten()
        y_pred = (y_pred_prob >= 0.7).astype(int)  # Use same threshold as detection
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_pred_prob)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    def save_model(self, model_path):
        """
        Save the fraud detection model to a file.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            success: Boolean indicating if saving was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save Keras model
            keras_path = os.path.join(os.path.dirname(model_path), 'fraud_model.h5')
            self.model.save(keras_path)
            
            # Save scaler
            scaler_path = os.path.join(os.path.dirname(model_path), 'fraud_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature names
            metadata = {
                'feature_names': self.feature_names,
                'keras_path': keras_path,
                'scaler_path': scaler_path
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
        
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """
        Load a saved fraud detection model from a file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            success: Boolean indicating if loading was successful
        """
        try:
            # Load metadata
            with open(model_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Set feature names
            self.feature_names = metadata['feature_names']
            
            # Load Keras model
            self.model = load_model(metadata['keras_path'])
            
            # Load scaler
            with open(metadata['scaler_path'], 'rb') as f:
                self.scaler = pickle.load(f)
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def update_model(self, training_data):
        """
        Update the fraud detection model with new training data.
        
        Args:
            training_data: Dictionary containing training samples
                - document_images: List of document images
                - labels: List of fraud/not-fraud labels
                - metadata: Additional training metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Extract training data
            images = training_data.get('document_images', [])
            labels = training_data.get('labels', [])
            
            if not images or not labels or len(images) != len(labels):
                print("Invalid training data format")
                return False
            
            # Preprocess images
            processed_images = []
            for img in images:
                proc_img = self._preprocess_image(img)
                if proc_img is not None:
                    processed_images.append(proc_img[0])  # Remove batch dimension
            
            if not processed_images:
                print("Failed to process any training images")
                return False
            
            # Convert to numpy arrays
            X_train = np.array(processed_images)
            y_train = np.array(labels)
            
            # Update model weights with new data
            self.model.fit(
                X_train, y_train,
                epochs=5,
                batch_size=16,
                verbose=0
            )
            
            # Save updated model
            model_file = os.path.join(self.model_path, 'mobilenet_model.h5')
            self.model.save(model_file)
            
            print(f"Model updated with {len(processed_images)} new samples")
            return True
            
        except Exception as e:
            print(f"Error updating fraud detection model: {str(e)}")
            return False
    
    def generate_dummy_data(self, n_samples=1000, fraud_ratio=0.2):
        """
        Generate dummy data for testing or demonstration.
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent samples
            
        Returns:
            X: Feature dictionaries
            y: Labels
        """
        np.random.seed(42)
        
        # Number of fraudulent samples
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud
        
        # Generate legitimate samples (normal distribution around benign values)
        X_legit = []
        for _ in range(n_legit):
            sample = {}
            # User behavior (normal ranges)
            sample['login_frequency'] = np.random.normal(5, 1)  # ~5 logins per week
            sample['avg_session_duration'] = np.random.normal(15, 5)  # ~15 minutes
            sample['device_change_frequency'] = np.random.normal(0.1, 0.05)  # Low device changes
            sample['location_change_frequency'] = np.random.normal(0.2, 0.1)  # Some location changes
            sample['unusual_time_access'] = np.random.normal(0.1, 0.1)  # Mostly regular hours
            sample['failed_login_attempts'] = np.random.normal(0.5, 0.5)  # Few failed attempts
            
            # Document features (high scores are good)
            sample['doc_tampering_score'] = np.random.normal(0.9, 0.05)
            sample['doc_consistency_score'] = np.random.normal(0.95, 0.03)
            sample['doc_pattern_match_score'] = np.random.normal(0.92, 0.04)
            sample['doc_font_consistency'] = np.random.normal(0.98, 0.02)
            sample['doc_micro_feature_score'] = np.random.normal(0.94, 0.03)
            
            # Biometric features (high scores are good)
            sample['biometric_confidence'] = np.random.normal(0.95, 0.03)
            sample['biometric_liveness_score'] = np.random.normal(0.97, 0.02)
            sample['biometric_consistency'] = np.random.normal(0.93, 0.04)
            
            # Network/device features (high scores are good)
            sample['ip_reputation_score'] = np.random.normal(0.9, 0.05)
            sample['vpn_proxy_score'] = np.random.normal(0.05, 0.05)  # Low VPN usage
            sample['device_integrity_score'] = np.random.normal(0.96, 0.03)
            sample['browser_integrity_score'] = np.random.normal(0.97, 0.02)
            sample['connection_anomaly_score'] = np.random.normal(0.1, 0.08)
            
            X_legit.append(sample)
        
        # Generate fraudulent samples (skewed distributions towards suspicious values)
        X_fraud = []
        for _ in range(n_fraud):
            sample = {}
            # Randomly select which aspects look fraudulent (not all at once for realism)
            fraud_aspects = np.random.choice([0, 1, 2, 3], size=2, replace=False)
            
            # User behavior (anomalous ranges)
            if 0 in fraud_aspects:
                sample['login_frequency'] = np.random.normal(20, 5)  # Too many logins
                sample['avg_session_duration'] = np.random.normal(1, 0.5)  # Very short sessions
                sample['device_change_frequency'] = np.random.normal(0.8, 0.1)  # Frequent device changes
                sample['location_change_frequency'] = np.random.normal(0.9, 0.1)  # Frequent location changes
                sample['unusual_time_access'] = np.random.normal(0.8, 0.1)  # Unusual hours
                sample['failed_login_attempts'] = np.random.normal(5, 2)  # Many failed attempts
            else:
                # Normal behavior to avoid too obvious patterns
                sample['login_frequency'] = np.random.normal(5, 1)
                sample['avg_session_duration'] = np.random.normal(15, 5)
                sample['device_change_frequency'] = np.random.normal(0.1, 0.05)
                sample['location_change_frequency'] = np.random.normal(0.2, 0.1)
                sample['unusual_time_access'] = np.random.normal(0.1, 0.1)
                sample['failed_login_attempts'] = np.random.normal(0.5, 0.5)
            
            # Document features (low scores suggest tampering)
            if 1 in fraud_aspects:
                sample['doc_tampering_score'] = np.random.normal(0.3, 0.1)
                sample['doc_consistency_score'] = np.random.normal(0.4, 0.1)
                sample['doc_pattern_match_score'] = np.random.normal(0.3, 0.1)
                sample['doc_font_consistency'] = np.random.normal(0.5, 0.1)
                sample['doc_micro_feature_score'] = np.random.normal(0.35, 0.1)
            else:
                sample['doc_tampering_score'] = np.random.normal(0.9, 0.05)
                sample['doc_consistency_score'] = np.random.normal(0.95, 0.03)
                sample['doc_pattern_match_score'] = np.random.normal(0.92, 0.04)
                sample['doc_font_consistency'] = np.random.normal(0.98, 0.02)
                sample['doc_micro_feature_score'] = np.random.normal(0.94, 0.03)
            
            # Biometric features (low scores suggest spoofing)
            if 2 in fraud_aspects:
                sample['biometric_confidence'] = np.random.normal(0.4, 0.1)
                sample['biometric_liveness_score'] = np.random.normal(0.3, 0.1)
                sample['biometric_consistency'] = np.random.normal(0.35, 0.1)
            else:
                sample['biometric_confidence'] = np.random.normal(0.95, 0.03)
                sample['biometric_liveness_score'] = np.random.normal(0.97, 0.02)
                sample['biometric_consistency'] = np.random.normal(0.93, 0.04)
            
            # Network/device features (suspicious patterns)
            if 3 in fraud_aspects:
                sample['ip_reputation_score'] = np.random.normal(0.2, 0.1)
                sample['vpn_proxy_score'] = np.random.normal(0.9, 0.05)  # High VPN usage
                sample['device_integrity_score'] = np.random.normal(0.3, 0.1)
                sample['browser_integrity_score'] = np.random.normal(0.4, 0.1)
                sample['connection_anomaly_score'] = np.random.normal(0.8, 0.1)
            else:
                sample['ip_reputation_score'] = np.random.normal(0.9, 0.05)
                sample['vpn_proxy_score'] = np.random.normal(0.05, 0.05)
                sample['device_integrity_score'] = np.random.normal(0.96, 0.03)
                sample['browser_integrity_score'] = np.random.normal(0.97, 0.02)
                sample['connection_anomaly_score'] = np.random.normal(0.1, 0.08)
            
            X_fraud.append(sample)
        
        # Combine datasets and create labels
        X = X_legit + X_fraud
        y = [0] * n_legit + [1] * n_fraud
        
        # Shuffle data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = [X[i] for i in indices]
        y_shuffled = [y[i] for i in indices]
        
        return X_shuffled, y_shuffled
    
    def _load_pretrained_model(self):
        """Load pre-trained MobileNetV2 model for document analysis."""
        model_file = os.path.join(self.model_path, 'mobilenet_model.h5')
        
        if os.path.exists(model_file):
            try:
                # Load existing model
                model = tf.keras.models.load_model(model_file)
                print("Loaded existing fraud detection model")
                return model
            except Exception as e:
                print(f"Could not load existing model: {str(e)}")
        
        # Create new model based on MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom layers for fraud detection
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Fraud probability
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        try:
            model.save(model_file)
            print("Created and saved new fraud detection model based on MobileNetV2")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
        return model
    
    def _load_fraud_patterns(self):
        """Load fraud patterns from storage."""
        patterns_file = os.path.join(self.model_path, 'fraud_patterns.json')
        
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                print(f"Loaded {len(patterns)} fraud patterns")
                return patterns
            except Exception as e:
                print(f"Error loading fraud patterns: {str(e)}")
        
        # Default patterns if file doesn't exist
        default_patterns = {
            "document_manipulation": {
                "threshold": 0.75,
                "weight": 0.4
            },
            "identity_mismatch": {
                "threshold": 0.8,
                "weight": 0.3
            },
            "suspicious_behavior": {
                "threshold": 0.85,
                "weight": 0.2
            },
            "unusual_access_pattern": {
                "threshold": 0.9,
                "weight": 0.1
            }
        }
        
        # Save default patterns
        try:
            with open(patterns_file, 'w') as f:
                json.dump(default_patterns, f, indent=2)
            print("Created default fraud patterns")
        except Exception as e:
            print(f"Error saving default fraud patterns: {str(e)}")
        
        return default_patterns
    
    def _save_fraud_patterns(self):
        """Save updated fraud patterns to storage."""
        patterns_file = os.path.join(self.model_path, 'fraud_patterns.json')
        
        try:
            with open(patterns_file, 'w') as f:
                json.dump(self.fraud_patterns, f, indent=2)
            print("Saved updated fraud patterns")
        except Exception as e:
            print(f"Error saving fraud patterns: {str(e)}")
    
    def _preprocess_image(self, image_data):
        """Preprocess an image for the model."""
        try:
            # Handle different image input formats
            if isinstance(image_data, str) and (os.path.exists(image_data) or Path(image_data).exists()):
                # Load from file
                img = cv2.imread(image_data)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(image_data, str) and image_data.startswith('data:image'):
                # Handle base64 image data
                import base64
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Assume it's already a numpy array
                img = image_data
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Resize to expected input size
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Expand dimensions for batch
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def detect_document_fraud(self, document_image):
        """
        Detect potential fraud in a document image using the pre-trained model.
        
        Args:
            document_image: Path to document image or image data
            
        Returns:
            dict: Fraud detection results with probability and analysis
        """
        start_time = time.time()
        
        try:
            # Preprocess the image
            processed_image = self._preprocess_image(document_image)
            
            if processed_image is None:
                return {
                    "success": False,
                    "is_fraudulent": False,
                    "fraud_probability": 0.0,
                    "message": "Failed to process document image",
                    "processing_time": time.time() - start_time
                }
            
            # Run inference with the model
            prediction = self.model.predict(processed_image, verbose=0)[0][0]
            
            # Analyze structural similarity for tampering detection
            tampering_score = self._analyze_document_tampering(processed_image)
            
            # Combine scores with weighted average
            combined_score = 0.7 * prediction + 0.3 * tampering_score
            
            # Determine if fraudulent based on threshold
            is_fraudulent = combined_score > 0.5
            
            return {
                "success": True,
                "is_fraudulent": bool(is_fraudulent),
                "fraud_probability": float(combined_score),
                "message": "Document fraud detection completed",
                "processing_time": time.time() - start_time,
                "analysis": {
                    "model_score": float(prediction),
                    "tampering_score": float(tampering_score),
                    "threshold": 0.5
                }
            }
            
        except Exception as e:
            print(f"Error in document fraud detection: {str(e)}")
            return {
                "success": False,
                "is_fraudulent": False,
                "fraud_probability": 0.0,
                "message": f"Error in document fraud detection: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _analyze_document_tampering(self, image):
        """
        Analyze document for signs of tampering using image processing techniques.
        
        Args:
            image: Preprocessed image
            
        Returns:
            float: Tampering probability (0-1)
        """
        try:
            # Convert batch to single image
            img = image[0]
            
            # Convert to grayscale
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Edge detection (Canny)
            edges = cv2.Canny(gray, 100, 200)
            
            # Noise analysis - standard deviation in small blocks
            blocks = []
            for i in range(0, gray.shape[0] - 8, 8):
                for j in range(0, gray.shape[1] - 8, 8):
                    block = gray[i:i+8, j:j+8]
                    blocks.append(np.std(block))
            
            block_std_variation = np.std(blocks) / np.mean(blocks) if np.mean(blocks) > 0 else 0
            
            # Calculate tampering score
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
            
            # Combine metrics (higher value = more tampering)
            tampering_score = 0.6 * block_std_variation + 0.4 * edge_density
            
            # Normalize to 0-1 range
            normalized_score = min(1.0, tampering_score * 5)
            
            return normalized_score
            
        except Exception as e:
            print(f"Error in tampering analysis: {str(e)}")
            return 0.0
    
    def detect_fraud(self, verification_data):
        """
        Comprehensive fraud detection combining document analysis and behavioral patterns.
        
        Args:
            verification_data: Dictionary containing verification data
                - document_image: Document image data or path
                - user_id: User ID
                - verification_context: Additional context data
            
        Returns:
            dict: Fraud detection results
        """
        start_time = time.time()
        
        try:
            # Extract data
            document_image = verification_data.get('document_image')
            user_id = verification_data.get('user_id', 'unknown')
            context = verification_data.get('verification_context', {})
            
            # If no document image is provided, use simplified analysis
            if document_image is None:
                return self._analyze_contextual_fraud(user_id, context)
            
            # Document fraud detection
            doc_result = self.detect_document_fraud(document_image)
            
            # Behavioral and contextual analysis
            contextual_score = self._analyze_context(user_id, context)
            
            # Combine scores
            combined_probability = 0.7 * doc_result['fraud_probability'] + 0.3 * contextual_score
            
            # Determine if fraudulent
            is_fraudulent = combined_probability > 0.5
            
            result = {
                "success": True,
                "is_fraudulent": is_fraudulent,
                "fraud_probability": float(combined_probability),
                "confidence": float(1.0 - min(combined_probability, 1.0 - combined_probability)),
                "processing_time": time.time() - start_time,
                "analysis": {
                    "document_consistency": float(1.0 - doc_result['fraud_probability']),
                    "behavioral_score": float(1.0 - contextual_score),
                    "identity_risk": float(combined_probability)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in comprehensive fraud detection: {str(e)}")
            return {
                "success": False,
                "is_fraudulent": False,
                "fraud_probability": 0.0,
                "confidence": 0.0,
                "message": f"Error in fraud detection: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _analyze_contextual_fraud(self, user_id, context):
        """Analyze fraud based on contextual data without document analysis."""
        contextual_score = self._analyze_context(user_id, context)
        
        return {
            "success": True,
            "is_fraudulent": contextual_score > 0.7,  # Higher threshold without document verification
            "fraud_probability": float(contextual_score),
            "confidence": float(1.0 - min(contextual_score, 1.0 - contextual_score)),
            "analysis": {
                "behavioral_score": float(1.0 - contextual_score),
                "identity_risk": float(contextual_score)
            }
        }
    
    def _analyze_context(self, user_id, context):
        """
        Analyze contextual factors for fraud detection.
        
        Args:
            user_id: User identifier
            context: Dictionary with contextual data like IP, device, time, etc.
            
        Returns:
            float: Contextual fraud probability (0-1)
        """
        # Default score - moderate suspicion
        base_score = 0.3
        
        # If no context provided, return base score
        if not context:
            return base_score
        
        # Analyze location/IP if provided
        ip_score = 0.0
        if 'ip_address' in context:
            ip_score = self._analyze_ip_risk(context['ip_address'])
            
        # Analyze device if provided
        device_score = 0.0
        if 'device_info' in context:
            device_score = self._analyze_device_risk(context['device_info'], user_id)
            
        # Analyze time if provided
        time_score = 0.0
        if 'timestamp' in context:
            time_score = self._analyze_time_risk(context['timestamp'], user_id)
            
        # Analyze behavior if provided
        behavior_score = 0.0
        if 'behavior' in context:
            behavior_score = self._analyze_behavior_risk(context['behavior'])
        
        # Weighted combination of factors
        weights = {
            'ip': 0.25,
            'device': 0.25,
            'time': 0.2,
            'behavior': 0.3
        }
        
        scores = {
            'ip': ip_score,
            'device': device_score,
            'time': time_score,
            'behavior': behavior_score
        }
        
        # Calculate weighted score
        available_factors = sum(weights[k] for k, v in scores.items() if v > 0)
        
        if available_factors == 0:
            return base_score
            
        # Normalize weights
        norm_weights = {k: (v / available_factors) for k, v in weights.items() if scores[k] > 0}
        
        # Calculate weighted average
        contextual_score = sum(scores[k] * norm_weights[k] for k in norm_weights)
        
        return contextual_score
    
    def _analyze_ip_risk(self, ip_address):
        """Analyze risk based on IP address."""
        # For development, return low risk
        return 0.2
    
    def _analyze_device_risk(self, device_info, user_id):
        """Analyze risk based on device information."""
        # For development, return low risk
        return 0.15
    
    def _analyze_time_risk(self, timestamp, user_id):
        """Analyze risk based on time of verification."""
        # For development, return low risk
        return 0.1
    
    def _analyze_behavior_risk(self, behavior_data):
        """Analyze risk based on user behavior."""
        # For development, return low risk
        return 0.25 