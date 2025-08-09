"""
Face Recognition Service yang kompatibel dengan NumPy 2.0+
Mengatasi masalah ARRAY_API dan dependency conflicts
"""

import os
import sys
import warnings
import logging

# Set environment variables untuk mengatasi compatibility issues
os.environ['NPY_DISABLE_LEGACY_ARRAY_API'] = '0'
os.environ['ARRAY_API_STRICT'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

logger = logging.getLogger(__name__)

# Import NumPy dengan error handling
try:
    import numpy as np
    # Force NumPy to use legacy array API for compatibility
    if hasattr(np, '_set_array_api_strict'):
        np._set_array_api_strict(False)
    logger.info(f"NumPy {np.__version__} loaded with legacy compatibility")
except ImportError:
    raise ImportError("NumPy is required")

# Import OpenCV
try:
    import cv2
    logger.info(f"OpenCV {cv2.__version__} loaded successfully")
except ImportError:
    raise ImportError("OpenCV is required for face detection")

# Django imports
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone
from ..models import Guest
import base64
import io
from PIL import Image
import json

class FaceRecognitionService:
    """
    Simple Face Recognition Service tanpa DeepFace
    Menggunakan OpenCV + feature extraction sederhana
    Kompatibel dengan NumPy 2.0+
    """
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.threshold = 0.7  # Similarity threshold
        
        # Initialize face detector
        self._initialize_face_detector()
        self.load_known_faces()
        
        logger.info("Simple Face Recognition Service initialized successfully")
    
    def _initialize_face_detector(self):
        """Initialize OpenCV face detector"""
        try:
            # Load Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Also load eye cascade for better face validation
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            logger.info("Face detectors initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.face_cascade = None
            self.eye_cascade = None
    
    def process_frame(self, frame):
        """Detect faces in frame and generate encodings"""
        try:
            face_locations = []
            face_encodings = []
            
            if self.face_cascade is None:
                logger.error("Face detector not available")
                return face_locations, face_encodings
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),  # Minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            logger.info(f"Detected {len(faces)} faces in frame")
            
            for (x, y, w, h) in faces:
                # Validate face by checking for eyes
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]
                
                # Check if face contains eyes (basic validation)
                is_valid_face = self._validate_face(face_roi_gray)
                
                if is_valid_face:
                    # Generate encoding for this face
                    encoding = self._generate_face_encoding(face_roi_color)
                    
                    if encoding is not None:
                        # Convert to face_recognition format: (top, right, bottom, left)
                        face_locations.append((y, x + w, y + h, x))
                        face_encodings.append(encoding)
                        logger.info(f"Generated encoding for face at ({x}, {y}, {w}, {h})")
                else:
                    logger.info(f"Face at ({x}, {y}, {w}, {h}) failed validation")
            
            return face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], []
    
    def _validate_face(self, face_gray):
        """Validate detected face by checking for eyes"""
        try:
            if self.eye_cascade is None:
                return True  # Skip validation if eye detector not available
            
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            return len(eyes) >= 1  # At least one eye should be detected
            
        except Exception:
            return True  # Default to valid if validation fails
    
    def _generate_face_encoding(self, face_image):
        """Generate 128-dimensional face encoding using multiple features"""
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_image, (96, 96))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Initialize feature vector
            features = []
            
            # 1. Local Binary Pattern features (32 dimensions)
            lbp_features = self._extract_lbp_features(gray)
            features.extend(lbp_features)
            
            # 2. Histogram of Oriented Gradients features (32 dimensions)
            hog_features = self._extract_hog_features(gray)
            features.extend(hog_features)
            
            # 3. Geometric features (32 dimensions)
            geometric_features = self._extract_geometric_features(gray)
            features.extend(geometric_features)
            
            # 4. Color features from original image (32 dimensions)
            color_features = self._extract_color_features(face_resized)
            features.extend(color_features)
            
            # Ensure exactly 128 dimensions
            encoding = np.array(features, dtype=np.float32)
            if len(encoding) > 128:
                encoding = encoding[:128]
            elif len(encoding) < 128:
                # Pad with zeros
                padding = np.zeros(128 - len(encoding), dtype=np.float32)
                encoding = np.concatenate([encoding, padding])
            
            # Normalize the encoding
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
            
            logger.debug(f"Generated {len(encoding)}-dimensional encoding")
            return encoding
            
        except Exception as e:
            logger.error(f"Error generating face encoding: {e}")
            return None
    
    def _extract_lbp_features(self, gray_image):
        """Extract Local Binary Pattern features"""
        try:
            # Simple LBP implementation
            height, width = gray_image.shape
            lbp_image = np.zeros_like(gray_image)
            
            # Calculate LBP for each pixel
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = gray_image[i, j]
                    binary_string = ''
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp_image[i, j] = int(binary_string, 2)
            
            # Calculate histogram
            hist, _ = np.histogram(lbp_image, bins=32, range=(0, 256))
            hist_normalized = hist.astype(np.float32)
            hist_normalized = hist_normalized / (np.sum(hist_normalized) + 1e-7)
            
            return hist_normalized.tolist()
            
        except Exception as e:
            logger.error(f"Error in LBP extraction: {e}")
            return [0.0] * 32
    
    def _extract_hog_features(self, gray_image):
        """Extract simplified HOG features"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude and orientation
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
            orientation[orientation < 0] += 180  # Convert to 0-180 range
            
            # Create histogram of orientations
            hist, _ = np.histogram(orientation, bins=16, range=(0, 180), weights=magnitude)
            hist_normalized = hist.astype(np.float32)
            hist_normalized = hist_normalized / (np.sum(hist_normalized) + 1e-7)
            
            # Extend to 32 dimensions by adding statistics
            features = hist_normalized.tolist()
            features.extend([
                np.mean(magnitude), np.std(magnitude),
                np.mean(grad_x), np.std(grad_x),
                np.mean(grad_y), np.std(grad_y),
                np.max(magnitude), np.min(magnitude),
                np.percentile(magnitude, 25), np.percentile(magnitude, 75),
                np.mean(orientation), np.std(orientation),
                np.median(magnitude), np.median(orientation),
                np.var(magnitude), np.var(orientation)
            ])
            
            return features[:32]
            
        except Exception as e:
            logger.error(f"Error in HOG extraction: {e}")
            return [0.0] * 32
    
    def _extract_geometric_features(self, gray_image):
        """Extract geometric features from face"""
        try:
            features = []
            h, w = gray_image.shape
            
            # Basic image statistics
            features.extend([
                np.mean(gray_image) / 255.0,
                np.std(gray_image) / 255.0,
                np.median(gray_image) / 255.0,
                np.min(gray_image) / 255.0,
                np.max(gray_image) / 255.0,
                np.var(gray_image) / (255.0**2)
            ])
            
            # Image moments
            moments = cv2.moments(gray_image)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                features.extend([cx/w, cy/h])
            else:
                features.extend([0.5, 0.5])
            
            # Intensity distribution features
            for percentile in [10, 25, 50, 75, 90]:
                features.append(np.percentile(gray_image, percentile) / 255.0)
            
            # Symmetry features (compare left and right halves)
            left_half = gray_image[:, :w//2]
            right_half = cv2.flip(gray_image[:, w//2:], 1)
            if left_half.shape == right_half.shape:
                symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                features.append(symmetry_score if not np.isnan(symmetry_score) else 0.0)
            else:
                features.append(0.0)
            
            # Regional features (divide image into 4 quadrants)
            for i in range(2):
                for j in range(2):
                    quad = gray_image[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                    features.extend([
                        np.mean(quad) / 255.0,
                        np.std(quad) / 255.0
                    ])
            
            # Pad or truncate to exactly 32 features
            while len(features) < 32:
                features.append(0.0)
            
            return features[:32]
            
        except Exception as e:
            logger.error(f"Error in geometric feature extraction: {e}")
            return [0.0] * 32
    
    def _extract_color_features(self, color_image):
        """Extract color-based features"""
        try:
            features = []
            
            # Convert to different color spaces and extract statistics
            # HSV color space
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            for channel in range(3):
                ch_data = hsv[:, :, channel]
                features.extend([
                    np.mean(ch_data) / 255.0,
                    np.std(ch_data) / 255.0
                ])
            
            # LAB color space
            lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
            for channel in range(3):
                ch_data = lab[:, :, channel]
                features.extend([
                    np.mean(ch_data) / 255.0,
                    np.std(ch_data) / 255.0
                ])
            
            # Original BGR channels
            for channel in range(3):
                ch_data = color_image[:, :, channel]
                features.extend([
                    np.mean(ch_data) / 255.0,
                    np.std(ch_data) / 255.0,
                    np.median(ch_data) / 255.0
                ])
            
            # Color dominance (most frequent colors)
            reshaped = color_image.reshape(-1, 3)
            unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
            if len(counts) > 0:
                dominant_idx = np.argmax(counts)
                dominant_color = unique_colors[dominant_idx]
                features.extend(dominant_color.astype(np.float32) / 255.0)
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Pad or truncate to exactly 32 features
            while len(features) < 32:
                features.append(0.0)
            
            return features[:32]
            
        except Exception as e:
            logger.error(f"Error in color feature extraction: {e}")
            return [0.0] * 32
    
    def recognize_face(self, face_encoding):
        """Recognize face using cosine similarity"""
        if len(self.known_face_encodings) == 0:
            logger.info("No known faces to compare against")
            return None, None
        
        try:
            best_similarity = -1
            best_match_index = -1
            
            # Compare with all known faces
            for i, known_encoding in enumerate(self.known_face_encodings):
                similarity = self._cosine_similarity(face_encoding, known_encoding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = i
            
            logger.info(f"Best similarity: {best_similarity:.3f} (threshold: {self.threshold})")
            
            # Check if similarity is above threshold
            if best_similarity > self.threshold:
                guest_id = self.known_face_ids[best_match_index]
                guest_name = self.known_face_names[best_match_index]
                logger.info(f"Face recognized as: {guest_name} (ID: {guest_id})")
                return guest_id, guest_name
            else:
                logger.info("Face not recognized - below threshold")
                return None, None
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, None
    
    def _cosine_similarity(self, encoding1, encoding2):
        """Calculate cosine similarity between two encodings"""
        try:
            # Ensure numpy arrays with float32 type
            enc1 = np.array(encoding1, dtype=np.float32)
            enc2 = np.array(encoding2, dtype=np.float32)
            
            # Calculate dot product and norms
            dot_product = np.dot(enc1, enc2)
            norm1 = np.linalg.norm(enc1)
            norm2 = np.linalg.norm(enc2)
            
            # Handle zero norm case
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in valid range
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def load_known_faces(self):
        """Load known faces from database"""
        logger.info("Loading known faces from database...")
        
        try:
            guests = Guest.objects.all()
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_ids = []
            
            success_count = 0
            error_count = 0
            
            for guest in guests:
                try:
                    # Try to load existing encoding
                    if guest.face_encoding:
                        encoding = guest.get_face_encoding()
                        if len(encoding) == 128:
                            self.known_face_encodings.append(np.array(encoding, dtype=np.float32))
                            self.known_face_names.append(guest.name)
                            self.known_face_ids.append(guest.id)
                            success_count += 1
                            continue
                    
                    # Try to generate from image if no valid encoding
                    if guest.face_image and default_storage.exists(guest.face_image.name):
                        image_path = guest.face_image.path
                        if os.path.exists(image_path):
                            image = cv2.imread(image_path)
                            if image is not None:
                                face_locations, face_encodings = self.process_frame(image)
                                if face_encodings:
                                    # Save new encoding
                                    guest.set_face_encoding(face_encodings[0])
                                    guest.save()
                                    
                                    self.known_face_encodings.append(face_encodings[0])
                                    self.known_face_names.append(guest.name)
                                    self.known_face_ids.append(guest.id)
                                    success_count += 1
                                    logger.info(f"Generated new encoding for: {guest.name}")
                                else:
                                    logger.warning(f"No face detected in image for: {guest.name}")
                                    error_count += 1
                            else:
                                logger.warning(f"Could not load image for: {guest.name}")
                                error_count += 1
                        else:
                            logger.warning(f"Image file not found for: {guest.name}")
                            error_count += 1
                    else:
                        logger.warning(f"No face image available for: {guest.name}")
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing guest {guest.name}: {e}")
                    error_count += 1
            
            logger.info(f"Loaded {success_count} known faces successfully")
            if error_count > 0:
                logger.warning(f"{error_count} guests could not be processed")
                
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
    
    def save_face_image(self, base64_image, guest_name):
        """Save base64 encoded face image"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate filename
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{guest_name}_{timestamp}.jpg"
            
            # Save image
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            file_content = ContentFile(buffer.getvalue(), filename)
            
            file_path = default_storage.save(f'face_images/{filename}', file_content)
            logger.info(f"Face image saved: {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving face image: {e}")
            return None
    
    def generate_encoding_from_path(self, image_path):
        """Generate face encoding from image file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            face_locations, face_encodings = self.process_frame(image)
            if face_encodings:
                return face_encodings[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating encoding from path: {e}")
            return None