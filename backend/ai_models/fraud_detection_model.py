import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.info(f"Loading model from {model_path}")
            self.load_model(model_path)
        else:
            # Create a new model
            logger.info("Creating new model and training with synthetic data")
            self._create_model()
            self.scaler = StandardScaler()
            
            # Generate and train on synthetic data for initial model
            X, y = self.generate_dummy_data(n_samples=5000, fraud_ratio=0.2)
            self.train(X, y, epochs=10, batch_size=32)
            
            # Save the trained model if path is provided
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.save_model(model_path)
                logger.info(f"Saved new model to {model_path}")
    
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
        logger.info("Created fraud detection neural network model")
    
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
            logger.error("Model is not trained. Please train or load a model first.")
            raise ValueError("Model is not trained. Please train or load a model first.")
        
        # Preprocess features
        X = self.preprocess_data(features)
        
        # Get fraud probability
        fraud_score = float(self.model.predict(X, verbose=0)[0][0])
        logger.info(f"Fraud detection score: {fraud_score:.4f}")
        
        # Dynamic threshold based on feature values
        # Higher threshold for first-time users with good document scores
        if features.get('login_frequency', 0) <= 1.0 and features.get('doc_consistency_score', 0) > 0.9:
            threshold = 0.85  # Higher threshold for new users with good docs
            logger.info(f"Using higher threshold (0.85) for new user with good documents")
        else:
            threshold = 0.75  # Default threshold
            logger.info(f"Using standard threshold (0.75)")
        
        # Determine if fraudulent based on threshold
        is_fraudulent = fraud_score >= threshold
        
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
                modified_score = float(self.model.predict(X_modified, verbose=0)[0][0])
                
                # Calculate importance as the difference in prediction
                importance = base_score - modified_score
                feature_importances.append((self.feature_names[i], importance))
            
            # Sort by importance (highest first)
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            
            # Get top risk factors (positive importance means feature contributes to fraud)
            risk_factors = [f[0] for f in feature_importances if f[1] > 0][:5]
            logger.info(f"Identified risk factors: {risk_factors}")
        
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
        logger.info(f"Training model on {len(X)} samples for {epochs} epochs")
        history = self.model.fit(
            X_scaled, np.array(y),
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Log training results
        val_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None
        accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else None
        logger.info(f"Training complete. Final accuracy: {accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
        
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
    
    def update_model(self, X, y):
        """
        Update the model with new data (incremental learning).
        
        Args:
            X: New feature data
            y: New labels
            
        Returns:
            history: Training history
        """
        # Preprocess input data if it's a list of dictionaries
        if isinstance(X, list) and isinstance(X[0], dict):
            X = self.preprocess_data(X)
        
        # Update scaler with new data
        # This is a simplified approach; in practice, you might want a more sophisticated way to update the scaler
        X_combined = np.vstack([
            self.scaler.inverse_transform(np.zeros((1, X.shape[1]))),  # A dummy sample to get the right scale
            X
        ])
        self.scaler.fit(X_combined)
        X_scaled = self.scaler.transform(X)
        
        # Fine-tune the model with new data
        history = self.model.fit(
            X_scaled, np.array(y),
            epochs=10,  # Shorter training for updates
            batch_size=32,
            verbose=1
        )
        
        return history
    
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