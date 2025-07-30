import cv2
import face_recognition
import numpy as np
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from ..models import Guest
import base64
import io
from PIL import Image

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known faces from database"""
        guests = Guest.objects.all()
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        for guest in guests:
            try:
                encoding = np.array(guest.get_face_encoding())
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(guest.name)
                self.known_face_ids.append(guest.id)
            except:
                continue
    
    def process_frame(self, frame):
        """Process a frame and return face locations and encodings"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return face_locations, face_encodings
    
    def recognize_face(self, face_encoding):
        """Recognize a face encoding against known faces"""
        if len(self.known_face_encodings) == 0:
            return None, None
        
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if matches[best_match_index] and distances[best_match_index] < 0.6:
                return self.known_face_ids[best_match_index], self.known_face_names[best_match_index]
        
        return None, None
    
    def save_face_image(self, base64_image, guest_name):
        """Save base64 image to file"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
            # Save to file
            filename = f"{guest_name}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            file_content = ContentFile(buffer.getvalue(), filename)
            
            return default_storage.save(f'face_images/{filename}', file_content)
        except Exception as e:
            print(f"Error saving face image: {e}")
            return None