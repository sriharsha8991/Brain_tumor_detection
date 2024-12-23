import streamlit as st
from PIL import Image
import google.generativeai as genai
import re
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini model at the top level
def initialize_model():
    try:
        API_KEY = "AIzaSyBHTV8_2Ul2nrKdLEht5BKWbQEkgIZvqIA" # Get from Streamlit secrets
    except:
        API_KEY = os.getenv("GENAI_API_KEY")  # Fallback to environment variable
        
    if not API_KEY:
        st.error("Please set the GENAI_API_KEY in Streamlit secrets or environment variables.")
        st.stop()
        
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(model_name="gemini-1.5-pro")

# Initialize model as a session state variable
if 'model' not in st.session_state:
    st.session_state.model = initialize_model()

class TumorDetection:
    def __init__(self, coordinates: List[int], confidence: float):
        self.coordinates = coordinates  # [ymin, xmin, ymax, xmax]
        self.confidence = confidence

def parse_tumor_detection(response_text: str, image_shape: Tuple[int, int]) -> List[TumorDetection]:
    """
    Parse the LLM response and scale coordinates to actual image dimensions
    """
    try:
        # Extract bounding box coordinates with more flexible pattern matching
        coord_pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\'"]?brain_tumor[\'"]?\]'
        coordinates = re.findall(coord_pattern, response_text)
        
        # Extract confidence scores (optional)
        confidence_pattern = r'confidence.*?(\d+)'
        confidences = re.findall(confidence_pattern, response_text.lower())
        
        detections = []
        height, width = image_shape[:2]
        
        for i, coords in enumerate(coordinates):
            # Convert coordinates to integers
            coords = [int(c) for c in coords]
            
            # Scale coordinates to actual image dimensions
            ymin = int((coords[0] / 1000) * height)
            xmin = int((coords[1] / 1000) * width)
            ymax = int((coords[2] / 1000) * height)
            xmax = int((coords[3] / 1000) * width)
            
            # Ensure coordinates are within image bounds
            ymin = max(0, min(ymin, height))
            xmin = max(0, min(xmin, width))
            ymax = max(0, min(ymax, height))
            xmax = max(0, min(xmax, width))
            
            confidence = float(confidences[i]) if i < len(confidences) else 90.0
            
            detection = TumorDetection(
                coordinates=[ymin, xmin, ymax, xmax],
                confidence=confidence
            )
            detections.append(detection)
        
        return detections
    except Exception as e:
        logger.error(f"Error parsing tumor detection: {str(e)}")
        return []

def enhance_mri_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance MRI image for better visualization
    """
    try:
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Convert back to RGB for drawing
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image

def draw_tumor_boxes(image: np.ndarray, detections: List[TumorDetection]) -> np.ndarray:
    """
    Draw accurate tumor bounding boxes
    """
    try:
        image_with_boxes = image.copy()
        
        for detection in detections:
            ymin, xmin, ymax, xmax = detection.coordinates
            
            # Draw the main bounding box
            cv2.rectangle(image_with_boxes, 
                        (xmin, ymin), 
                        (xmax, ymax),
                        (0, 255, 0),  # Green color
                        2)  # Line thickness
            
            # Add confidence label
            label = f"Tumor {detection.confidence:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(image_with_boxes,
                        (xmin, ymin - label_size[1] - 10),
                        (xmin + label_size[0], ymin),
                        (0, 255, 0),
                        -1)  # Filled rectangle
            
            # Draw label text
            cv2.putText(image_with_boxes,
                       label,
                       (xmin, ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,  # Font scale
                       (0, 0, 0),  # Black text
                       2)  # Line thickness
            
        return image_with_boxes
    except Exception as e:
        logger.error(f"Error drawing boxes: {str(e)}")
        return image

def main():
    st.set_page_config(page_title="Brain Tumor MRI Analysis", layout="wide")
    
    st.title("Brain Tumor Detection in MRI Scans")
    st.write("Upload an MRI scan to detect and analyze potential tumor regions.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            # Load and preprocess image
            input_image = Image.open(uploaded_file)
            np_image = np.array(input_image)
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original MRI Scan")
                st.image(input_image, use_container_width=True)  # Updated parameter
            
            if st.button("Analyze Scan"):
                with st.spinner("Analyzing MRI scan..."):
                    # Generate analysis with Gemini
                    prompt = """Analyze this brain MRI scan and detect any tumors. For each tumor:
                    1. Provide exact bounding box coordinates as [ymin, xmin, ymax, xmax, 'brain_tumor']
                    2. Ensure coordinates are proportional to image dimensions (0-1000 range)
                    3. Indicate confidence level (0-100%)
                    Focus on precise localization of the tumor mass."""
                    
                    # Get model response using session state
                    response = st.session_state.model.generate_content([input_image, prompt])
                    
                    # Parse and process detections
                    detections = parse_tumor_detection(response.text, np_image.shape)
                    
                    if not detections:
                        st.warning("No tumors detected in the scan.")
                        return
                    
                    # Enhance image and draw detections
                    enhanced_image = enhance_mri_image(np_image)
                    result_image = draw_tumor_boxes(enhanced_image, detections)
                    
                    # Display results
                    with col2:
                        st.subheader("Detection Results")
                        st.image(result_image, use_container_width=True)  # Updated parameter
                    
                    # Show detection details
                    st.subheader("Detection Details")
                    for i, detection in enumerate(detections, 1):
                        st.write(f"Tumor Region {i}:")
                        st.write(f"- Coordinates (y1, x1, y2, x2): {detection.coordinates}")
                        st.write(f"- Confidence: {detection.confidence}%")
                    
                    # Show raw response in expandable section
                    with st.expander("Show Raw Analysis"):
                        st.text(response.text)
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()