import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pickle
import re
from dotenv import load_dotenv
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

load_dotenv()

# Set Tesseract OCR path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/macOS

class DocumentVerificationModel:
    """Document verification model for government-issued IDs."""
    
    def __init__(self, model_path=None):
        """
        Initialize document verification model.
        
        Args:
            model_path: Path to saved model data (if any)
        """
        self.document_types = ['passport', 'drivers_license', 'national_id']
        self.model = None
        self.template_patterns = {
            'passport': {
                'format': r'^[A-Z0-9]{8,9}$',  # Passport number format
                'fields': ['surname', 'given_name', 'passport_no', 'nationality', 'date_of_birth', 'expiry_date']
            },
            'drivers_license': {
                'format': r'^[A-Z0-9]{5,12}$',  # Driver's license number format
                'fields': ['surname', 'given_name', 'license_no', 'date_of_birth', 'issue_date', 'expiry_date', 'address']
            },
            'national_id': {
                'format': r'^[0-9]{9,12}$',  # National ID number format
                'fields': ['surname', 'given_name', 'id_no', 'date_of_birth', 'issue_date', 'address']
            }
        }
        
        # Load saved model if provided, otherwise create new one
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_model()
    
    def _create_model(self):
        """Create a document classification model."""
        # Use MobileNetV2 as a base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(self.document_types), activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for document classification or OCR.
        
        Args:
            image_path: Path to the image or image as numpy array
            
        Returns:
            processed_image: Preprocessed image
        """
        if isinstance(image_path, str):
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
        else:
            # Use the provided image array
            image = image_path
        
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to enhance text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def preprocess_for_classification(self, image_path):
        """
        Preprocess an image for document classification.
        
        Args:
            image_path: Path to the image or image as numpy array
            
        Returns:
            processed_image: Preprocessed image for model input
        """
        if isinstance(image_path, str):
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
        else:
            # Use the provided image array
            image = image_path.copy()
        
        # Resize image to model input size
        image = cv2.resize(image, (224, 224))
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def detect_document_type(self, image_path):
        """
        Detect the type of document from an image.
        
        Args:
            image_path: Path to the image or image as numpy array
            
        Returns:
            document_type: Detected document type
            confidence: Confidence score for the detection
        """
        if self.model is None:
            raise ValueError("Document classification model not initialized")
        
        # Preprocess image for classification
        processed_image = self.preprocess_for_classification(image_path)
        
        # Predict document type
        predictions = self.model.predict(processed_image)[0]
        
        # Get the document type with highest confidence
        doc_type_index = np.argmax(predictions)
        confidence = float(predictions[doc_type_index])
        
        return self.document_types[doc_type_index], confidence
    
    def extract_text_from_document(self, image_path):
        """
        Extract text from a document image using OCR.
        
        Args:
            image_path: Path to the image or image as numpy array
            
        Returns:
            text: Extracted text
        """
        # Preprocess image for OCR
        processed_image = self.preprocess_image(image_path)
        
        # Use Tesseract OCR to extract text
        text = pytesseract.image_to_string(processed_image)
        
        return text
    
    def extract_fields(self, image_path, document_type=None):
        """
        Extract specific fields from a document.
        
        Args:
            image_path: Path to the image or image as numpy array
            document_type: Optional document type (if known)
            
        Returns:
            fields: Dictionary of extracted fields
            document_type: Detected document type
        """
        # Detect document type if not provided
        if document_type is None:
            document_type, _ = self.detect_document_type(image_path)
        
        # Extract all text from document
        text = self.extract_text_from_document(image_path)
        
        # Extract fields based on document type
        fields = {}
        
        if document_type == 'passport':
            # Extract passport number (format varies by country)
            passport_match = re.search(r'passport no[.:]\s*([A-Z0-9]{6,9})', text, re.IGNORECASE)
            if passport_match:
                fields['passport_no'] = passport_match.group(1)
            
            # Extract name
            name_match = re.search(r'surname[.:]\s*([A-Za-z\s]+)', text, re.IGNORECASE)
            if name_match:
                fields['surname'] = name_match.group(1).strip()
            
            given_name_match = re.search(r'given names[.:]\s*([A-Za-z\s]+)', text, re.IGNORECASE)
            if given_name_match:
                fields['given_name'] = given_name_match.group(1).strip()
            
            # Extract nationality
            nationality_match = re.search(r'nationality[.:]\s*([A-Za-z\s]+)', text, re.IGNORECASE)
            if nationality_match:
                fields['nationality'] = nationality_match.group(1).strip()
            
            # Extract dates
            dob_match = re.search(r'birth[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if dob_match:
                fields['date_of_birth'] = dob_match.group(1)
            
            expiry_match = re.search(r'expiry[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if expiry_match:
                fields['expiry_date'] = expiry_match.group(1)
        
        elif document_type == 'drivers_license':
            # Extract license number
            license_match = re.search(r'license(?:\s+no)?[.:]\s*([A-Z0-9]{5,12})', text, re.IGNORECASE)
            if license_match:
                fields['license_no'] = license_match.group(1)
            
            # Extract name
            name_match = re.search(r'name[.:]\s*([A-Za-z\s]+)', text, re.IGNORECASE)
            if name_match:
                full_name = name_match.group(1).strip()
                name_parts = full_name.split(' ', 1)
                if len(name_parts) > 1:
                    fields['given_name'] = name_parts[0]
                    fields['surname'] = name_parts[1]
                else:
                    fields['surname'] = full_name
            
            # Extract address
            address_match = re.search(r'address[.:]\s*([A-Za-z0-9\s,.-]+)', text, re.IGNORECASE)
            if address_match:
                fields['address'] = address_match.group(1).strip()
            
            # Extract dates
            dob_match = re.search(r'birth(?:day)?(?:\s+date)?[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if dob_match:
                fields['date_of_birth'] = dob_match.group(1)
            
            issue_match = re.search(r'issue[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if issue_match:
                fields['issue_date'] = issue_match.group(1)
            
            expiry_match = re.search(r'expiry[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if expiry_match:
                fields['expiry_date'] = expiry_match.group(1)
        
        elif document_type == 'national_id':
            # Extract ID number
            id_match = re.search(r'id(?:\s+no)?[.:]\s*([0-9]{9,12})', text, re.IGNORECASE)
            if id_match:
                fields['id_no'] = id_match.group(1)
            
            # Extract name
            name_match = re.search(r'name[.:]\s*([A-Za-z\s]+)', text, re.IGNORECASE)
            if name_match:
                full_name = name_match.group(1).strip()
                name_parts = full_name.split(' ', 1)
                if len(name_parts) > 1:
                    fields['given_name'] = name_parts[0]
                    fields['surname'] = name_parts[1]
                else:
                    fields['surname'] = full_name
            
            # Extract address
            address_match = re.search(r'address[.:]\s*([A-Za-z0-9\s,.-]+)', text, re.IGNORECASE)
            if address_match:
                fields['address'] = address_match.group(1).strip()
            
            # Extract dates
            dob_match = re.search(r'birth(?:day)?(?:\s+date)?[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if dob_match:
                fields['date_of_birth'] = dob_match.group(1)
            
            issue_match = re.search(r'issue[.:]\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if issue_match:
                fields['issue_date'] = issue_match.group(1)
        
        return fields, document_type
    
    def verify_document(self, image_path, expected_fields=None):
        """
        Verify a document for authenticity.
        
        Args:
            image_path: Path to the image or image as numpy array
            expected_fields: Optional dictionary of expected field values
            
        Returns:
            is_authentic: Boolean indicating if the document appears authentic
            confidence: Confidence score for the authenticity check
            details: Dictionary with verification details
        """
        # Extract document type and fields
        fields, document_type = self.extract_fields(image_path)
        
        # Initialize verification results
        verification_results = {
            'document_type': document_type,
            'extracted_fields': fields,
            'format_check': False,
            'field_presence_check': False,
            'field_consistency_check': True,
            'expected_field_match': True if expected_fields is None else False
        }
        
        # Check document ID format
        id_field = 'passport_no' if document_type == 'passport' else ('license_no' if document_type == 'drivers_license' else 'id_no')
        
        if id_field in fields:
            id_value = fields[id_field]
            id_format = self.template_patterns[document_type]['format']
            verification_results['format_check'] = bool(re.match(id_format, id_value))
        
        # Check if all required fields are present
        required_fields = self.template_patterns[document_type]['fields']
        fields_present = all(field in fields for field in required_fields)
        verification_results['field_presence_check'] = fields_present
        
        # Check field consistency (if dates make sense, etc.)
        if 'date_of_birth' in fields and 'expiry_date' in fields:
            try:
                # Simplified check - would need better date parsing in production
                dob_year = int(re.search(r'\d{4}', fields['date_of_birth']).group())
                expiry_year = int(re.search(r'\d{4}', fields['expiry_date']).group())
                
                if dob_year >= expiry_year:
                    verification_results['field_consistency_check'] = False
            except:
                # If date parsing fails, can't check consistency
                pass
        
        # Check against expected fields
        if expected_fields:
            matches = []
            for field, expected_value in expected_fields.items():
                if field in fields:
                    # Case-insensitive comparison with some flexibility
                    actual_value = fields[field].lower().strip()
                    expected_value = expected_value.lower().strip()
                    
                    # Exact match or soft match (actual contains expected or vice versa)
                    match = (actual_value == expected_value or 
                             actual_value in expected_value or 
                             expected_value in actual_value)
                    matches.append(match)
            
            # Document matches if majority of fields match
            if matches and sum(matches) / len(matches) >= 0.7:
                verification_results['expected_field_match'] = True
        
        # Calculate overall authenticity score
        checks = [
            verification_results['format_check'],
            verification_results['field_presence_check'],
            verification_results['field_consistency_check'],
            verification_results['expected_field_match']
        ]
        
        check_weights = [0.3, 0.3, 0.2, 0.2]
        confidence = sum(check * weight for check, weight in zip(checks, check_weights))
        
        # Document is considered authentic if confidence is above threshold
        is_authentic = confidence >= 0.7
        
        return is_authentic, confidence, verification_results
    
    def detect_tampering(self, image_path):
        """
        Detect potential tampering in a document.
        
        Args:
            image_path: Path to the image or image as numpy array
            
        Returns:
            is_tampered: Boolean indicating if tampering is detected
            tampering_score: Score indicating likelihood of tampering (0-1)
            tampering_details: Dictionary with tampering detection details
        """
        if isinstance(image_path, str):
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
        else:
            # Use the provided image array
            image = image_path.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        tampering_details = {}
        
        # Check for noise inconsistencies
        noise_score = self._check_noise_consistency(gray)
        tampering_details['noise_score'] = noise_score
        
        # Check for edge inconsistencies
        edge_score = self._check_edge_consistency(gray)
        tampering_details['edge_score'] = edge_score
        
        # Check for compression inconsistencies
        compression_score = self._check_compression_artifacts(gray)
        tampering_details['compression_score'] = compression_score
        
        # Check for color inconsistencies
        if len(image.shape) == 3:  # Color image
            color_score = self._check_color_consistency(image)
            tampering_details['color_score'] = color_score
        else:
            color_score = 1.0  # Grayscale image, no color check
        
        # Calculate overall tampering score (inverse of authenticity)
        # Lower score means higher likelihood of tampering
        tampering_score = 1.0 - ((noise_score + edge_score + compression_score + color_score) / 4.0)
        
        # Document is considered tampered if tampering score is above threshold
        is_tampered = tampering_score >= 0.3
        
        return is_tampered, tampering_score, tampering_details
    
    def _check_noise_consistency(self, gray_image):
        """Check for noise inconsistencies in the image."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Calculate difference between original and blurred
        noise = cv2.absdiff(gray_image, blurred)
        
        # Divide image into regions and check noise consistency
        h, w = gray_image.shape
        region_size = min(h, w) // 4
        
        noise_levels = []
        for y in range(0, h - region_size, region_size):
            for x in range(0, w - region_size, region_size):
                region = noise[y:y+region_size, x:x+region_size]
                noise_level = np.mean(region)
                noise_levels.append(noise_level)
        
        if not noise_levels:
            return 1.0
        
        # Calculate variance of noise levels
        noise_variance = np.var(noise_levels)
        
        # Normalize and invert (lower variance means more consistent noise)
        normalized_score = 1.0 - min(noise_variance / 50.0, 1.0)
        
        return normalized_score
    
    def _check_edge_consistency(self, gray_image):
        """Check for edge inconsistencies in the image."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Check for abrupt edges or unnaturally straight lines
        # This is a simplified approach; in practice, more sophisticated methods would be used
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 1.0
        
        # Count number of suspiciously straight lines
        suspicious_lines = 0
        total_lines = len(lines)
        
        if total_lines == 0:
            return 1.0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is perfectly horizontal or vertical
            if x1 == x2 or y1 == y2:
                suspicious_lines += 1
        
        # Calculate ratio of suspicious lines to total lines
        suspicious_ratio = suspicious_lines / total_lines
        
        # Normalize score (lower ratio means more natural edges)
        normalized_score = 1.0 - suspicious_ratio
        
        return normalized_score
    
    def _check_compression_artifacts(self, gray_image):
        """Check for inconsistent compression artifacts."""
        # Apply DCT transform to detect JPEG compression blocks
        h, w = gray_image.shape
        h_blocks = h // 8
        w_blocks = w // 8
        
        if h_blocks == 0 or w_blocks == 0:
            return 1.0
        
        # Check for 8x8 block artifacts (common in JPEG)
        block_scores = []
        for y in range(0, h_blocks * 8, 8):
            for x in range(0, w_blocks * 8, 8):
                if y + 8 <= h and x + 8 <= w:
                    block = gray_image[y:y+8, x:x+8].astype(float)
                    # Calculate variance within block
                    block_var = np.var(block)
                    block_scores.append(block_var)
        
        if not block_scores:
            return 1.0
        
        # Calculate variance of block variances
        block_variance = np.var(block_scores)
        
        # Normalize and invert (lower variance means more consistent compression)
        normalized_score = 1.0 - min(block_variance / 1000.0, 1.0)
        
        return normalized_score
    
    def _check_color_consistency(self, color_image):
        """Check for color inconsistencies that might indicate tampering."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Split into channels
        h, s, v = cv2.split(hsv)
        
        # Check saturation consistency (tampered regions often have different saturation)
        s_blurred = cv2.GaussianBlur(s, (15, 15), 0)
        s_diff = cv2.absdiff(s, s_blurred)
        s_thresh = cv2.threshold(s_diff, 30, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate ratio of inconsistent pixels
        inconsistent_ratio = np.sum(s_thresh) / (255.0 * s.size)
        
        # Normalize score (lower ratio means more consistent colors)
        normalized_score = 1.0 - min(inconsistent_ratio * 10.0, 1.0)
        
        return normalized_score
    
    def save_model(self, model_path):
        """
        Save the document verification model to a file.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            success: Boolean indicating if saving was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save Keras model
            keras_path = os.path.join(os.path.dirname(model_path), 'document_model.h5')
            self.model.save(keras_path)
            
            # Save metadata
            metadata = {
                'document_types': self.document_types,
                'template_patterns': self.template_patterns,
                'keras_path': keras_path
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
        
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """
        Load a saved document verification model from a file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            success: Boolean indicating if loading was successful
        """
        try:
            # Load metadata
            with open(model_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Set document types and patterns
            self.document_types = metadata['document_types']
            self.template_patterns = metadata['template_patterns']
            
            # Load Keras model
            self.model = load_model(metadata['keras_path'])
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False 