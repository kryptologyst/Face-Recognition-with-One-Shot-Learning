"""Streamlit demo for face recognition with one-shot learning."""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device import get_device
from src.utils.face_utils import FaceDetector
from src.models.siamese import SiameseNetwork
from src.models.clip_models import CLIPSiameseNetwork


class FaceRecognitionDemo:
    """Face recognition demo using Streamlit."""
    
    def __init__(self):
        """Initialize the demo."""
        self.device = get_device()
        self.face_detector = FaceDetector()
        self.model = None
        self.gallery_embeddings = {}
        self.gallery_labels = {}
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            # Try to load Siamese network
            checkpoint_path = "checkpoints/best_model.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Create model
                self.model = SiameseNetwork(
                    backbone="resnet18",
                    embedding_dim=512,
                    dropout=0.3,
                    pretrained=True
                )
                
                # Load weights
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                
                st.success("Model loaded successfully!")
            else:
                st.warning("No trained model found. Please train a model first.")
                self.model = None
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Preprocess image for model input.
        
        Args:
            image: Input PIL image.
            
        Returns:
            Preprocessed tensor or None if no face detected.
        """
        try:
            # Convert to numpy array
            image_np = np.array(image)
            
            # Detect faces
            faces = self.face_detector.detect_faces(image_np)
            if not faces:
                return None
            
            # Extract largest face
            largest_face = max(faces, key=lambda box: box[2] * box[3])
            face = self.face_detector.extract_face(image_np, largest_face, (112, 112))
            
            # Normalize
            face = face.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Convert to tensor
            face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
            return face_tensor.to(self.device)
            
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def get_embedding(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Get embedding for image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor.
            
        Returns:
            Embedding tensor or None.
        """
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                embedding = self.model.forward_one(image_tensor)
                return embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return None
    
    def add_to_gallery(self, image: Image.Image, label: str):
        """Add image to gallery.
        
        Args:
            image: Image to add.
            label: Label for the image.
        """
        image_tensor = self.preprocess_image(image)
        if image_tensor is None:
            st.error("No face detected in the image.")
            return
        
        embedding = self.get_embedding(image_tensor)
        if embedding is None:
            st.error("Failed to get embedding.")
            return
        
        # Store in gallery
        if label not in self.gallery_embeddings:
            self.gallery_embeddings[label] = []
            self.gallery_labels[label] = []
        
        self.gallery_embeddings[label].append(embedding)
        self.gallery_labels[label].append(image)
        
        st.success(f"Added {label} to gallery!")
    
    def recognize_face(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Recognize face in image.
        
        Args:
            image: Image to recognize.
            
        Returns:
            Tuple of (predicted_label, confidence).
        """
        if not self.gallery_embeddings:
            return None, 0.0
        
        image_tensor = self.preprocess_image(image)
        if image_tensor is None:
            return None, 0.0
        
        query_embedding = self.get_embedding(image_tensor)
        if query_embedding is None:
            return None, 0.0
        
        # Compute distances to all gallery embeddings
        min_distance = float('inf')
        best_match = None
        
        for label, embeddings in self.gallery_embeddings.items():
            for embedding in embeddings:
                distance = torch.norm(query_embedding - embedding).item()
                if distance < min_distance:
                    min_distance = distance
                    best_match = label
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0.0, 1.0 - min_distance / 2.0)
        
        return best_match, confidence


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Face Recognition with One-Shot Learning",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("Face Recognition with One-Shot Learning")
    st.markdown("Upload images to build a gallery and test face recognition!")
    
    # Initialize demo
    if "demo" not in st.session_state:
        st.session_state.demo = FaceRecognitionDemo()
    
    demo = st.session_state.demo
    
    # Sidebar for gallery management
    with st.sidebar:
        st.header("Gallery Management")
        
        # Add to gallery
        st.subheader("Add to Gallery")
        uploaded_image = st.file_uploader(
            "Choose an image", 
            type=['jpg', 'jpeg', 'png'],
            key="gallery_upload"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            label = st.text_input("Enter label for this person:")
            if st.button("Add to Gallery") and label:
                demo.add_to_gallery(image, label)
        
        # Show gallery
        st.subheader("Current Gallery")
        if demo.gallery_embeddings:
            for label in demo.gallery_embeddings:
                st.write(f"**{label}**: {len(demo.gallery_embeddings[label])} images")
        else:
            st.write("No images in gallery yet.")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Test Face Recognition")
        
        # Upload test image
        test_image = st.file_uploader(
            "Choose an image to recognize", 
            type=['jpg', 'jpeg', 'png'],
            key="test_upload"
        )
        
        if test_image:
            image = Image.open(test_image)
            st.image(image, caption="Test Image", use_column_width=True)
            
            if st.button("Recognize Face"):
                if not demo.gallery_embeddings:
                    st.error("Please add some images to the gallery first!")
                else:
                    predicted_label, confidence = demo.recognize_face(image)
                    
                    if predicted_label:
                        st.success(f"Recognized as: **{predicted_label}**")
                        st.info(f"Confidence: {confidence:.2%}")
                    else:
                        st.error("No face detected or recognition failed.")
    
    with col2:
        st.header("Gallery Preview")
        
        if demo.gallery_embeddings:
            for label, images in demo.gallery_labels.items():
                st.subheader(f"{label}")
                cols = st.columns(min(3, len(images)))
                for i, img in enumerate(images):
                    with cols[i % 3]:
                        st.image(img, use_column_width=True)
        else:
            st.info("No images in gallery yet. Add some images using the sidebar!")
    
    # Model information
    with st.expander("Model Information"):
        st.write("**Model**: Siamese Network with ResNet18 backbone")
        st.write("**Input Size**: 112x112 pixels")
        st.write("**Embedding Dimension**: 512")
        st.write("**Device**:", demo.device)
        
        if demo.model:
            st.write("**Status**: Model loaded successfully")
        else:
            st.write("**Status**: No model loaded")


if __name__ == "__main__":
    main()
