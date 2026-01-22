"""Generate sample face dataset for testing."""

import os
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_synthetic_face(
    width: int = 112,
    height: int = 112,
    face_id: int = 0,
    variation: int = 0
) -> np.ndarray:
    """Generate a synthetic face image.
    
    Args:
        width: Image width.
        height: Image height.
        face_id: Face identity (affects color).
        variation: Variation within identity.
        
    Returns:
        Generated face image as numpy array.
    """
    # Create base image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set background color based on face_id
    bg_color = [
        (50 + face_id * 20, 100 + face_id * 15, 150 + face_id * 25),
        (150 + face_id * 15, 50 + face_id * 20, 100 + face_id * 25),
        (100 + face_id * 25, 150 + face_id * 15, 50 + face_id * 20),
    ][face_id % 3]
    
    image[:] = bg_color
    
    # Add face shape (ellipse)
    face_color = (200 + variation * 10, 180 + variation * 8, 160 + variation * 12)
    cv2.ellipse(image, (width//2, height//2), (width//3, height//3), 0, 0, 360, face_color, -1)
    
    # Add eyes
    eye_color = (50, 50, 50)
    left_eye_center = (width//2 - width//6, height//2 - height//8)
    right_eye_center = (width//2 + width//6, height//2 - height//8)
    
    cv2.circle(image, left_eye_center, width//20, eye_color, -1)
    cv2.circle(image, right_eye_center, width//20, eye_color, -1)
    
    # Add nose
    nose_color = (150 + variation * 5, 120 + variation * 5, 100 + variation * 5)
    cv2.ellipse(image, (width//2, height//2), (width//30, height//15), 0, 0, 360, nose_color, -1)
    
    # Add mouth
    mouth_color = (100, 50, 50)
    cv2.ellipse(image, (width//2, height//2 + height//6), (width//8, height//20), 0, 0, 180, mouth_color, -1)
    
    # Add some noise for realism
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def create_sample_dataset(
    output_dir: str = "data/raw",
    num_identities: int = 10,
    images_per_identity: int = 5
) -> None:
    """Create a sample face dataset.
    
    Args:
        output_dir: Output directory for the dataset.
        num_identities: Number of different identities.
        images_per_identity: Number of images per identity.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample dataset with {num_identities} identities...")
    
    for identity in range(num_identities):
        # Create identity directory
        identity_dir = output_path / f"person_{identity:03d}"
        identity_dir.mkdir(exist_ok=True)
        
        # Generate images for this identity
        for img_idx in range(images_per_identity):
            # Generate synthetic face
            face_image = generate_synthetic_face(
                width=112,
                height=112,
                face_id=identity,
                variation=img_idx
            )
            
            # Save image
            image_path = identity_dir / f"image_{img_idx:03d}.jpg"
            cv2.imwrite(str(image_path), face_image)
        
        print(f"Created {images_per_identity} images for person_{identity:03d}")
    
    print(f"Dataset created successfully in {output_dir}")
    print(f"Total images: {num_identities * images_per_identity}")


def create_realistic_sample_dataset(
    output_dir: str = "data/raw",
    num_identities: int = 5,
    images_per_identity: int = 10
) -> None:
    """Create a more realistic sample dataset using PIL.
    
    Args:
        output_dir: Output directory for the dataset.
        num_identities: Number of different identities.
        images_per_identity: Number of images per identity.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating realistic sample dataset with {num_identities} identities...")
    
    # Define face colors for different identities
    face_colors = [
        (255, 220, 177),  # Light skin
        (255, 200, 150),  # Medium-light skin
        (255, 180, 120),  # Medium skin
        (255, 160, 100),  # Medium-dark skin
        (255, 140, 80),   # Dark skin
    ]
    
    for identity in range(num_identities):
        # Create identity directory
        identity_dir = output_path / f"person_{identity:03d}"
        identity_dir.mkdir(exist_ok=True)
        
        # Get face color for this identity
        face_color = face_colors[identity % len(face_colors)]
        
        # Generate images for this identity
        for img_idx in range(images_per_identity):
            # Create image
            img = Image.new('RGB', (112, 112), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Draw face (circle)
            face_center = (56, 56)
            face_radius = 40
            draw.ellipse(
                [face_center[0] - face_radius, face_center[1] - face_radius,
                 face_center[0] + face_radius, face_center[1] + face_radius],
                fill=face_color,
                outline=(200, 200, 200)
            )
            
            # Draw eyes
            eye_y = 45
            left_eye_x = 40
            right_eye_x = 72
            
            # Add variation to eye positions
            eye_offset = random.randint(-3, 3)
            draw.ellipse([left_eye_x - 5 + eye_offset, eye_y - 3,
                         left_eye_x + 5 + eye_offset, eye_y + 3], fill=(50, 50, 50))
            draw.ellipse([right_eye_x - 5 + eye_offset, eye_y - 3,
                         right_eye_x + 5 + eye_offset, eye_y + 3], fill=(50, 50, 50))
            
            # Draw nose
            nose_y = 55
            draw.ellipse([55, nose_y - 2, 57, nose_y + 2], fill=(200, 150, 100))
            
            # Draw mouth
            mouth_y = 70
            mouth_width = random.randint(8, 15)
            draw.ellipse([56 - mouth_width//2, mouth_y - 2,
                         56 + mouth_width//2, mouth_y + 2], fill=(150, 50, 50))
            
            # Add some random variation
            if random.random() < 0.3:
                # Add glasses
                draw.rectangle([35, 40, 77, 50], outline=(100, 100, 100), width=2)
            
            if random.random() < 0.2:
                # Add mustache
                draw.ellipse([50, 60, 62, 65], fill=(100, 100, 100))
            
            # Save image
            image_path = identity_dir / f"image_{img_idx:03d}.jpg"
            img.save(image_path, "JPEG", quality=95)
        
        print(f"Created {images_per_identity} images for person_{identity:03d}")
    
    print(f"Realistic dataset created successfully in {output_dir}")
    print(f"Total images: {num_identities * images_per_identity}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample face dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--num_identities", type=int, default=10, help="Number of identities")
    parser.add_argument("--images_per_identity", type=int, default=5, help="Images per identity")
    parser.add_argument("--realistic", action="store_true", help="Use realistic face generation")
    
    args = parser.parse_args()
    
    if args.realistic:
        create_realistic_sample_dataset(
            args.output_dir,
            args.num_identities,
            args.images_per_identity
        )
    else:
        create_sample_dataset(
            args.output_dir,
            args.num_identities,
            args.images_per_identity
        )
