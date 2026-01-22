"""Face detection and preprocessing utilities."""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple, Union

try:
    from mtcnn import MTCNN
except ImportError:
    MTCNN = None

try:
    import face_recognition
except ImportError:
    face_recognition = None


class FaceDetector:
    """Face detection and preprocessing class."""
    
    def __init__(self, method: str = "mtcnn", device: Optional[str] = None):
        """Initialize face detector.
        
        Args:
            method: Detection method ('mtcnn', 'opencv', 'face_recognition').
            device: Device to run detection on.
        """
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if method == "mtcnn" and MTCNN is not None:
            self.detector = MTCNN(device=self.device)
        elif method == "opencv":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        elif method == "face_recognition" and face_recognition is not None:
            self.detector = face_recognition
        else:
            raise ValueError(f"Unsupported detection method: {method}")
    
    def detect_faces(self, image: Union[np.ndarray, Image.Image]) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image.
        
        Args:
            image: Input image as numpy array or PIL Image.
            
        Returns:
            List of face bounding boxes as (x, y, w, h).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.method == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.method == "opencv":
            return self._detect_opencv(image)
        elif self.method == "face_recognition":
            return self._detect_face_recognition(image)
        else:
            raise ValueError(f"Unsupported detection method: {self.method}")
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        faces = self.detector.detect_faces(image)
        boxes = []
        for face in faces:
            x, y, w, h = face["box"]
            boxes.append((x, y, w, h))
        return boxes
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for x, y, w, h in faces]
    
    def _detect_face_recognition(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library."""
        face_locations = self.detector.face_locations(image)
        boxes = []
        for top, right, bottom, left in face_locations:
            x, y, w, h = left, top, right - left, bottom - top
            boxes.append((x, y, w, h))
        return boxes
    
    def extract_face(
        self, 
        image: Union[np.ndarray, Image.Image], 
        box: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """Extract and resize face from bounding box.
        
        Args:
            image: Input image.
            box: Face bounding box (x, y, w, h).
            target_size: Target size for extracted face.
            
        Returns:
            Extracted and resized face as numpy array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        x, y, w, h = box
        
        # Add padding
        padding = max(w, h) // 4
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face = image[y1:y2, x1:x2]
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        
        return face
    
    def preprocess_image(
        self, 
        image_path: str, 
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """Preprocess image: load, detect face, extract and normalize.
        
        Args:
            image_path: Path to image file.
            target_size: Target size for face.
            
        Returns:
            Preprocessed face image or None if no face detected.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            
            faces = self.detect_faces(image)
            if not faces:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda box: box[2] * box[3])
            face = self.extract_face(image, largest_face, target_size)
            
            # Normalize to [0, 1]
            face = face.astype(np.float32) / 255.0
            
            return face
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
