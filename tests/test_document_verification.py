import sys
import os
import json
import cv2
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.document_verification_model import DocumentVerificationModel

def create_sample_id_image():
    """Create a simple sample ID image for testing when no real image is available."""
    # Create a blank image (white background)
    img = np.ones((600, 900, 3), dtype=np.uint8) * 255
    
    # Add simulated ID card border
    cv2.rectangle(img, (50, 50), (850, 550), (200, 200, 200), 2)
    
    # Add a colored header for the ID
    cv2.rectangle(img, (50, 50), (850, 120), (173, 216, 230), -1)  # Light blue
    cv2.putText(img, "DRIVER LICENSE", (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Add a photo placeholder
    cv2.rectangle(img, (100, 150), (250, 300), (200, 200, 200), -1)
    cv2.rectangle(img, (100, 150), (250, 300), (100, 100, 100), 2)
    
    # Add text fields
    fields = [
        ("DL: 123456789", 280, 180),
        ("NAME: JOHN DOE", 280, 220),
        ("DOB: 01/01/1990", 280, 260),
        ("EXP: 12/31/2025", 280, 300),
        ("ADDRESS: 123 MAIN ST", 100, 350),
        ("CLASS: C", 100, 400),
        ("SEX: M", 280, 400),
        ("EYES: BRN", 450, 400)
    ]
    
    for text, x, y in fields:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Add a small security pattern
    for i in range(100, 800, 20):
        cv2.line(img, (i, 500), (i + 10, 520), (220, 220, 220), 1)
    
    return img

def find_test_image():
    """Find a suitable test image in common locations."""
    # Check in test_images directory
    test_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_images")
    
    # Check for common image filenames
    common_names = ["driver_license.jpg", "id.jpg", "passport.jpg", "document.jpg", "license.jpg", "id_card.jpg"]
    
    for name in common_names:
        path = os.path.join(test_images_dir, name)
        if os.path.exists(path):
            print(f"Found test image: {path}")
            return path
    
    # Look in the root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for any jpg or png files that might be suitable
    for ext in [".jpg", ".jpeg", ".png"]:
        for file in os.listdir(root_dir):
            if file.lower().endswith(ext):
                path = os.path.join(root_dir, file)
                print(f"Found potential test image: {path}")
                return path
    
    # No existing image found, create a sample one
    print("No test image found, creating a sample image...")
    sample_img = create_sample_id_image()
    os.makedirs(test_images_dir, exist_ok=True)
    sample_path = os.path.join(test_images_dir, "sample_id.jpg")
    cv2.imwrite(sample_path, sample_img)
    print(f"Created sample test image at: {sample_path}")
    return sample_path

def test_document_verification():
    """Test the document verification functionality with enhanced OCR and flexible document recognition."""
    print("Testing Document Verification with Enhanced OCR")
    print("-" * 50)
    
    # Initialize the document verification model
    doc_model = DocumentVerificationModel()
    
    # Find a test image
    test_doc_path = find_test_image()
    
    print(f"\nTesting document verification with image: {test_doc_path}")
    
    # Test document type detection
    doc_type, confidence = doc_model.detect_document_type(test_doc_path)
    print(f"\nDocument Type Detection:")
    print(f"Detected type: {doc_type}")
    print(f"Confidence: {confidence:.2f}")
    
    # Test field extraction with enhanced OCR
    print(f"\nTesting Field Extraction with Enhanced OCR:")
    fields, doc_type = doc_model.extract_fields(test_doc_path)
    
    print(f"Extracted Fields ({len(fields)} fields found):")
    for field, value in fields.items():
        print(f"  - {field}: {value}")
    
    # Test document verification
    print(f"\nTesting Document Verification:")
    is_authentic, auth_confidence, verification_results = doc_model.verify_document(test_doc_path)
    
    print(f"Document authentic: {is_authentic}")
    print(f"Authentication confidence: {auth_confidence:.2f}")
    print(f"Verification details:")
    print(f"  - Format check: {verification_results.get('format_check', False)}")
    print(f"  - Field presence: {verification_results.get('field_presence_check', False)}")
    
    # Test tampering detection
    print(f"\nTesting Tampering Detection:")
    is_tampered, tampering_score, tampering_details = doc_model.detect_tampering(test_doc_path)
    
    print(f"Tampering detected: {is_tampered}")
    print(f"Tampering score: {tampering_score:.2f}")
    
    # Final result - for our testing purposes, we'll consider it a success
    # if we can at least extract some fields and get a non-tampered result
    success = len(fields) > 0 and not is_tampered
    
    return success

if __name__ == "__main__":
    # Create test directory if it doesn't exist
    os.makedirs("tests", exist_ok=True)
    
    success = test_document_verification()
    
    if success:
        print("\nDocument verification test passed successfully!")
        sys.exit(0)
    else:
        print("\nDocument verification test failed!")
        sys.exit(1) 