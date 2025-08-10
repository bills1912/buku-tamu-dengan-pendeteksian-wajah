"""
Face Recognition Service menggunakan DeepFace dengan model FaceNet
Fixed version - mengatasi error detection dan encoding issues
"""

import os
import sys
import warnings
import logging
import json
import base64
import io
from pathlib import Path

# Set environment variables untuk compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

logger = logging.getLogger(__name__)

# Import required libraries dengan error handling
try:
    import numpy as np
    logger.info(f"NumPy {np.__version__} loaded successfully")
except ImportError:
    raise ImportError("NumPy is required")

try:
    import cv2
    logger.info(f"OpenCV {cv2.__version__} loaded successfully")
except ImportError:
    raise ImportError("OpenCV is required")

# Safe TensorFlow import
try:
    import tensorflow as tf
    # Safe TensorFlow configuration
    if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
        tf.get_logger().setLevel('ERROR')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError:
                pass  # Memory growth must be set before GPUs have been initialized
    logger.info(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError:
    logger.error("TensorFlow is required for DeepFace")
    raise ImportError("TensorFlow is required for DeepFace")

# Safe DeepFace import
try:
    from deepface import DeepFace
    logger.info("DeepFace loaded successfully")
except ImportError:
    logger.error("DeepFace is required. Install with: pip install deepface")
    raise ImportError("DeepFace is required. Install with: pip install deepface")

try:
    from PIL import Image
    logger.info("PIL loaded successfully")
except ImportError:
    raise ImportError("Pillow is required")

# Django imports
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone
from django.conf import settings
from ..models import Guest


class FaceRecognitionService:
    """
    Advanced Face Recognition Service menggunakan DeepFace dengan model FaceNet
    Fixed version dengan better error handling
    """
    
    def __init__(self):
        # Configuration dari settings dengan defaults
        self.config = getattr(settings, 'FACE_RECOGNITION', {})
        self.deepface_config = getattr(settings, 'DEEPFACE_CONFIG', {})
        
        # Model configuration dengan safe defaults
        self.model_name = self.config.get('MODEL', 'Facenet')
        self.distance_metric = self.config.get('DISTANCE_METRIC', 'cosine')
        self.detector_backend = self.config.get('DETECTOR_BACKEND', 'opencv')
        self.threshold = self.config.get('THRESHOLD', 0.5)
        self.enforce_detection = self.config.get('ENFORCE_DETECTION', False)  # Set False untuk stability
        self.align_faces = self.config.get('ALIGN_FACES', True)
        self.normalization = self.config.get('NORMALIZATION', 'base')
        
        # Storage untuk face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Initialize service dengan error handling
        try:
            self._initialize_deepface()
            self.load_known_faces()
            logger.info(f"DeepFace Face Recognition Service initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Face Recognition Service: {e}")
            # Continue without crashing
    
    def _initialize_deepface(self):
        """Initialize DeepFace dengan safe warm-up"""
        try:
            logger.info("Initializing DeepFace models...")
            
            # Create simple test image
            dummy_image = np.ones((160, 160, 3), dtype=np.uint8) * 128  # Gray image
            
            # Test model availability
            try:
                embedding = DeepFace.represent(
                    img_path=dummy_image,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,  # Always False for initialization
                    align=False,  # No alignment for dummy image
                    normalization=self.normalization
                )
                logger.info(f"Model '{self.model_name}' initialized successfully")
                logger.info(f"Embedding dimension: {len(embedding[0]['embedding'])}")
            except Exception as e:
                logger.warning(f"Model initialization warning: {e}")
                # Try with minimal settings
                try:
                    embedding = DeepFace.represent(
                        img_path=dummy_image,
                        model_name='Facenet',  # Fallback to Facenet
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=False
                    )
                    self.model_name = 'Facenet'
                    self.detector_backend = 'opencv'
                    logger.info("Fallback to Facenet with OpenCV successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback initialization failed: {fallback_error}")
            
            logger.info("DeepFace initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepFace: {e}")
            # Don't raise, allow service to continue with limited functionality
    
    def process_frame(self, frame):
        """
        Detect faces in frame dan generate encodings dengan improved error handling
        
        Args:
            frame: OpenCV image array (BGR format)
            
        Returns:
            tuple: (face_locations, face_encodings)
        """
        try:
            face_locations = []
            face_encodings = []
            
            if frame is None or frame.size == 0:
                logger.warning("Empty or invalid frame received")
                return face_locations, face_encodings
            
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.error(f"Invalid frame shape: {frame.shape}")
                return face_locations, face_encodings
            
            # Convert BGR to RGB untuk DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure proper data type and range
            if rgb_frame.dtype != np.uint8:
                rgb_frame = rgb_frame.astype(np.uint8)
            
            # Try multiple approaches for face detection
            face_encodings_generated = False
            
            # Approach 1: Direct embedding generation (skip face extraction)
            try:
                logger.info("Attempting direct embedding generation...")
                embeddings = DeepFace.represent(
                    img_path=rgb_frame,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,  # Critical: set to False
                    align=self.align_faces,
                    normalization=self.normalization
                )
                
                if embeddings and len(embeddings) > 0:
                    for embedding_data in embeddings:
                        if 'embedding' in embedding_data:
                            embedding_vector = np.array(embedding_data['embedding'], dtype=np.float32)
                            face_encodings.append(embedding_vector)
                            
                            # Estimate face location using OpenCV
                            face_location = self._estimate_face_location(rgb_frame)
                            face_locations.append(face_location)
                            
                            logger.info(f"Generated embedding: {len(embedding_vector)} dimensions")
                            face_encodings_generated = True
                
            except Exception as direct_error:
                logger.warning(f"Direct embedding generation failed: {direct_error}")
            
            # Approach 2: Face extraction then embedding (if direct method failed)
            if not face_encodings_generated:
                try:
                    logger.info("Attempting face extraction then embedding...")
                    detected_faces = DeepFace.extract_faces(
                        img_path=rgb_frame,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        align=self.align_faces,
                        grayscale=False
                    )
                    
                    logger.info(f"Detected {len(detected_faces)} faces using {self.detector_backend}")
                    
                    for i, face_data in enumerate(detected_faces):
                        try:
                            # Handle different return types from extract_faces
                            if isinstance(face_data, dict):
                                if 'face' in face_data:
                                    face_array = face_data['face']
                                else:
                                    continue
                            else:
                                face_array = face_data
                            
                            # Ensure face_array is numpy array
                            if not isinstance(face_array, np.ndarray):
                                logger.warning(f"Face {i} is not numpy array: {type(face_array)}")
                                continue
                            
                            # Generate encoding untuk face
                            embedding = self._generate_face_encoding_safe(face_array)
                            
                            if embedding is not None:
                                face_encodings.append(embedding)
                                face_location = self._estimate_face_location(rgb_frame)
                                face_locations.append(face_location)
                                logger.info(f"Generated encoding for face {i+1}")
                                face_encodings_generated = True
                            
                        except Exception as face_error:
                            logger.warning(f"Error processing face {i+1}: {face_error}")
                            continue
                    
                except Exception as extraction_error:
                    logger.warning(f"Face extraction failed: {extraction_error}")
            
            # Approach 3: OpenCV fallback detection
            if not face_encodings_generated:
                logger.info("Trying OpenCV fallback detection...")
                face_locations_cv = self._opencv_face_detection(rgb_frame)
                if face_locations_cv:
                    # Use the whole frame for encoding if face detected
                    try:
                        embeddings = DeepFace.represent(
                            img_path=rgb_frame,
                            model_name=self.model_name,
                            detector_backend='skip',  # Skip detection, use whole image
                            enforce_detection=False,
                            align=False,
                            normalization=self.normalization
                        )
                        
                        if embeddings and len(embeddings) > 0:
                            embedding_vector = np.array(embeddings[0]['embedding'], dtype=np.float32)
                            face_encodings.append(embedding_vector)
                            face_locations.append(face_locations_cv[0])
                            logger.info("Generated encoding using OpenCV fallback")
                            
                    except Exception as fallback_error:
                        logger.error(f"OpenCV fallback also failed: {fallback_error}")
            
            logger.info(f"Final result: {len(face_encodings)} encodings, {len(face_locations)} locations")
            return face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return [], []
    
    def _generate_face_encoding_safe(self, face_array):
        """
        Safe face encoding generation dengan multiple fallbacks
        
        Args:
            face_array: Face array dari extract_faces
            
        Returns:
            numpy.ndarray: Face embedding vector atau None
        """
        try:
            # Validate input
            if face_array is None:
                logger.warning("Face array is None")
                return None
            
            if not isinstance(face_array, np.ndarray):
                logger.warning(f"Face array is not numpy array: {type(face_array)}")
                return None
            
            if face_array.size == 0:
                logger.warning("Face array is empty")
                return None
            
            # Ensure proper shape (height, width, channels)
            if len(face_array.shape) != 3 or face_array.shape[2] != 3:
                logger.warning(f"Invalid face array shape: {face_array.shape}")
                return None
            
            # Handle different value ranges
            if face_array.max() <= 1.0:
                # Convert dari [0,1] ke [0,255]
                face_image = (face_array * 255).astype(np.uint8)
                logger.debug("Converted face array from [0,1] to [0,255]")
            else:
                face_image = face_array.astype(np.uint8)
                logger.debug("Face array already in [0,255] range")
            
            # Ensure minimum size
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                logger.warning(f"Face too small: {face_image.shape}")
                # Resize to minimum acceptable size
                face_image = cv2.resize(face_image, (96, 96))
            
            # Generate embedding menggunakan DeepFace
            embeddings = DeepFace.represent(
                img_path=face_image,
                model_name=self.model_name,
                detector_backend='skip',  # Skip detection karena sudah di-crop
                enforce_detection=False,
                align=False,  # Sudah aligned dari extract_faces
                normalization=self.normalization
            )
            
            if embeddings and len(embeddings) > 0:
                embedding_vector = np.array(embeddings[0]['embedding'], dtype=np.float32)
                logger.debug(f"Generated {len(embedding_vector)}-dimensional embedding")
                return embedding_vector
            else:
                logger.warning("No embedding generated from DeepFace.represent")
                return None
                
        except Exception as e:
            logger.error(f"Error in safe face encoding generation: {e}")
            return None
    
    def _estimate_face_location(self, rgb_frame):
        """
        Estimate face location menggunakan OpenCV sebagai fallback
        
        Args:
            rgb_frame: RGB image array
            
        Returns:
            tuple: (top, right, bottom, left) coordinates
        """
        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            # Load Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take first face
                # Convert to (top, right, bottom, left) format
                return (y, x + w, y + h, x)
            else:
                # Return center area as default
                h, w = rgb_frame.shape[:2]
                margin = min(w, h) // 4
                return (margin, w - margin, h - margin, margin)
                
        except Exception as e:
            logger.warning(f"Error estimating face location: {e}")
            # Return default location covering most of image
            h, w = rgb_frame.shape[:2]
            return (0, w, h, 0)
    
    def _opencv_face_detection(self, rgb_frame):
        """
        OpenCV face detection sebagai fallback
        
        Args:
            rgb_frame: RGB image array
            
        Returns:
            list: List of face locations
        """
        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_locations = []
            for (x, y, w, h) in faces:
                # Convert to (top, right, bottom, left) format
                face_locations.append((y, x + w, y + h, x))
            
            return face_locations
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def recognize_face(self, face_encoding):
        """
        Recognize face menggunakan distance comparison dengan improved error handling
        
        Args:
            face_encoding: Face embedding vector
            
        Returns:
            tuple: (guest_id, guest_name) atau (None, None) jika tidak dikenali
        """
        if len(self.known_face_encodings) == 0:
            logger.info("No known faces to compare against")
            return None, None
        
        try:
            # Validate input encoding
            if face_encoding is None:
                logger.warning("Face encoding is None")
                return None, None
            
            if not isinstance(face_encoding, (np.ndarray, list)):
                logger.warning(f"Invalid face encoding type: {type(face_encoding)}")
                return None, None
            
            # Convert to numpy array if needed
            if isinstance(face_encoding, list):
                face_encoding = np.array(face_encoding, dtype=np.float32)
            
            if face_encoding.size == 0:
                logger.warning("Face encoding is empty")
                return None, None
            
            best_distance = float('inf')
            best_match_index = -1
            
            # Compare dengan semua known faces
            for i, known_encoding in enumerate(self.known_face_encodings):
                try:
                    distance = self._calculate_distance(face_encoding, known_encoding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match_index = i
                except Exception as distance_error:
                    logger.warning(f"Error calculating distance for face {i}: {distance_error}")
                    continue
            
            logger.info(f"Best distance: {best_distance:.4f} (threshold: {self.threshold})")
            
            # Check jika distance di bawah threshold
            if best_distance < self.threshold and best_match_index >= 0:
                guest_id = self.known_face_ids[best_match_index]
                guest_name = self.known_face_names[best_match_index]
                logger.info(f"Face recognized as: {guest_name} (ID: {guest_id}) with distance: {best_distance:.4f}")
                return guest_id, guest_name
            else:
                logger.info("Face not recognized - above threshold")
                return None, None
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, None
    
    def _calculate_distance(self, encoding1, encoding2):
        """
        Calculate distance between two face encodings dengan error handling
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            float: Distance value
        """
        try:
            # Convert to numpy arrays
            enc1 = np.array(encoding1, dtype=np.float32)
            enc2 = np.array(encoding2, dtype=np.float32)
            
            # Validate dimensions
            if enc1.shape != enc2.shape:
                logger.warning(f"Encoding dimension mismatch: {enc1.shape} vs {enc2.shape}")
                return float('inf')
            
            if enc1.size == 0 or enc2.size == 0:
                logger.warning("Empty encoding arrays")
                return float('inf')
            
            if self.distance_metric == 'cosine':
                # Cosine distance = 1 - cosine_similarity
                dot_product = np.dot(enc1, enc2)
                norm1 = np.linalg.norm(enc1)
                norm2 = np.linalg.norm(enc2)
                
                if norm1 == 0 or norm2 == 0:
                    return 1.0  # Maximum cosine distance
                
                cosine_similarity = dot_product / (norm1 * norm2)
                cosine_distance = 1.0 - cosine_similarity
                return float(np.clip(cosine_distance, 0.0, 2.0))
                
            elif self.distance_metric == 'euclidean':
                # Euclidean distance
                distance = np.linalg.norm(enc1 - enc2)
                return float(distance)
                
            elif self.distance_metric == 'euclidean_l2':
                # L2 normalized Euclidean distance
                enc1_normalized = enc1 / (np.linalg.norm(enc1) + 1e-8)
                enc2_normalized = enc2 / (np.linalg.norm(enc2) + 1e-8)
                distance = np.linalg.norm(enc1_normalized - enc2_normalized)
                return float(distance)
                
            else:
                logger.warning(f"Unknown distance metric: {self.distance_metric}, using cosine")
                return self._calculate_distance_cosine(enc1, enc2)
                
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def _calculate_distance_cosine(self, enc1, enc2):
        """Fallback cosine distance calculation"""
        try:
            dot_product = np.dot(enc1, enc2)
            norm1 = np.linalg.norm(enc1)
            norm2 = np.linalg.norm(enc2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            return float(1.0 - cosine_similarity)
        except:
            return float('inf')
    
    def load_known_faces(self):
        """Load known faces dari database dengan improved error handling"""
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
                    # Coba load existing encoding
                    if guest.face_encoding:
                        try:
                            encoding = guest.get_face_encoding()
                            if isinstance(encoding, (list, np.ndarray)) and len(encoding) >= 64:  # Minimum acceptable dimension
                                encoding_array = np.array(encoding, dtype=np.float32)
                                self.known_face_encodings.append(encoding_array)
                                self.known_face_names.append(guest.name)
                                self.known_face_ids.append(guest.id)
                                success_count += 1
                                continue
                            else:
                                logger.warning(f"Invalid encoding for {guest.name}: {len(encoding) if encoding else 0} dimensions")
                        except Exception as encoding_error:
                            logger.warning(f"Error loading encoding for {guest.name}: {encoding_error}")
                    
                    # Generate dari image jika tidak ada encoding atau encoding invalid
                    if guest.face_image and default_storage.exists(guest.face_image.name):
                        try:
                            # Handle both local file and storage
                            if hasattr(guest.face_image, 'path') and os.path.exists(guest.face_image.path):
                                image_path = guest.face_image.path
                            else:
                                # For cloud storage, download image temporarily
                                image_data = default_storage.open(guest.face_image.name).read()
                                temp_image = Image.open(io.BytesIO(image_data))
                                # Convert to cv2 format
                                image_array = cv2.cvtColor(np.array(temp_image), cv2.COLOR_RGB2BGR)
                                encoding = self._generate_encoding_from_array(image_array)
                            
                            if hasattr(guest.face_image, 'path') and os.path.exists(guest.face_image.path):
                                encoding = self._generate_encoding_from_file(guest.face_image.path)
                            
                            if encoding is not None and len(encoding) >= 64:
                                # Save encoding ke database
                                guest.set_face_encoding(encoding)
                                guest.save()
                                
                                self.known_face_encodings.append(encoding)
                                self.known_face_names.append(guest.name)
                                self.known_face_ids.append(guest.id)
                                success_count += 1
                                logger.info(f"Generated new encoding for: {guest.name}")
                            else:
                                logger.warning(f"Failed to generate valid encoding for: {guest.name}")
                                error_count += 1
                                
                        except Exception as img_error:
                            logger.error(f"Error processing image for {guest.name}: {img_error}")
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
    
    def _generate_encoding_from_file(self, image_path):
        """Generate face encoding dari file image dengan error handling"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            return self._generate_encoding_from_array(image)
            
        except Exception as e:
            logger.error(f"Error generating encoding from file {image_path}: {e}")
            return None
    
    def _generate_encoding_from_array(self, image_array):
        """Generate encoding dari numpy array"""
        try:
            # Convert ke RGB
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Generate encoding menggunakan DeepFace
            embeddings = DeepFace.represent(
                img_path=rgb_image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Critical: set to False
                align=self.align_faces,
                normalization=self.normalization
            )
            
            if embeddings and len(embeddings) > 0:
                encoding = np.array(embeddings[0]['embedding'], dtype=np.float32)
                logger.debug(f"Generated encoding from array: {len(encoding)} dimensions")
                return encoding
            else:
                logger.warning("No face found in image array")
                return None
                
        except Exception as e:
            logger.error(f"Error generating encoding from array: {e}")
            return None
    
    def save_face_image(self, base64_image, guest_name):
        """Save base64 encoded face image dengan error handling"""
        try:
            # Decode base64 image
            if ',' in base64_image:
                image_data = base64.b64decode(base64_image.split(',')[1])
            else:
                image_data = base64.b64decode(base64_image)
                
            image = Image.open(io.BytesIO(image_data))
            
            # Convert ke RGB jika perlu
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image untuk konsistensi
            face_settings = getattr(settings, 'FACE_IMAGES_SETTINGS', {})
            original_size = face_settings.get('ORIGINAL_SIZE', (256, 256))
            image = image.resize(original_size, Image.Resampling.LANCZOS)
            
            # Generate filename yang aman
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            safe_name = "".join(c for c in guest_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            if not safe_name:
                safe_name = "guest"
            filename = f"{safe_name}_{timestamp}.jpg"
            
            # Save image
            buffer = io.BytesIO()
            quality = face_settings.get('QUALITY', 85)
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
            file_content = ContentFile(buffer.getvalue(), filename)
            
            upload_path = face_settings.get('UPLOAD_PATH', 'face_images/')
            file_path = default_storage.save(f'{upload_path}{filename}', file_content)
            
            logger.info(f"Face image saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving face image for {guest_name}: {e}")
            return None
    
    def get_model_info(self):
        """Get informasi tentang model dan konfigurasi"""
        try:
            return {
                'model_name': self.model_name,
                'detector_backend': self.detector_backend,
                'distance_metric': self.distance_metric,
                'threshold': self.threshold,
                'enforce_detection': self.enforce_detection,
                'align_faces': self.align_faces,
                'normalization': self.normalization,
                'known_faces_count': len(self.known_face_encodings),
                'service_status': 'active',
                'tensorflow_available': True,
                'deepface_available': True,
                'opencv_version': cv2.__version__ if cv2 else 'unknown'
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'service_status': 'error',
                'error_message': str(e)
            }