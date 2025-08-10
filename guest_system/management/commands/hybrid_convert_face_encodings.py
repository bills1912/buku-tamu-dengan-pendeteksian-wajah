"""
Script untuk convert face encoding lama ke format baru
Dan update face recognition service untuk support backward compatibility
"""

import numpy as np
import logging
from django.core.management.base import BaseCommand
from guest_system.models import Guest

logger = logging.getLogger(__name__)

def convert_old_encoding_to_new_format(old_encoding):
    """
    Convert old face_recognition encoding to new format
    """
    try:
        # Old encoding dari face_recognition (dlib) biasanya sudah normalized
        # tapi dengan range yang berbeda
        old_enc = np.array(old_encoding, dtype=np.float32)
        
        # Normalize to [0,1] range first
        old_min = np.min(old_enc)
        old_max = np.max(old_enc)
        
        if old_max != old_min:
            normalized = (old_enc - old_min) / (old_max - old_min)
        else:
            normalized = old_enc
        
        # Pad or truncate to ensure 128 dimensions
        if len(normalized) > 128:
            new_encoding = normalized[:128]
        elif len(normalized) < 128:
            padding = np.zeros(128 - len(normalized), dtype=np.float32)
            new_encoding = np.concatenate([normalized, padding])
        else:
            new_encoding = normalized
        
        # Final normalization using L2 norm
        norm = np.linalg.norm(new_encoding)
        if norm > 0:
            new_encoding = new_encoding / norm
        
        return new_encoding
        
    except Exception as e:
        logger.error(f"Error converting encoding: {e}")
        return None

class Command(BaseCommand):
    help = 'Convert old face encodings to new format'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--guest-id',
            type=int,
            help='Convert specific guest by ID',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Convert all guests with old encodings',
        )
        parser.add_argument(
            '--test-compatibility',
            action='store_true',
            help='Test compatibility after conversion',
        )
    
    def handle(self, *args, **options):
        guest_id = options.get('guest_id')
        convert_all = options.get('all')
        test_compat = options.get('test_compatibility')
        
        if guest_id:
            self.convert_single_guest(guest_id)
        elif convert_all:
            self.convert_all_guests()
        elif test_compat:
            self.test_compatibility()
        else:
            self.print_help()
    
    def convert_single_guest(self, guest_id):
        """Convert encoding for single guest"""
        try:
            guest = Guest.objects.get(id=guest_id)
            self.stdout.write(f"Converting encoding for: {guest.name}")
            
            if not guest.face_encoding:
                self.stdout.write("No encoding found")
                return
            
            old_encoding = guest.get_face_encoding()
            self.stdout.write(f"Old encoding - Length: {len(old_encoding)}, Range: {np.min(old_encoding):.3f} to {np.max(old_encoding):.3f}")
            
            new_encoding = convert_old_encoding_to_new_format(old_encoding)
            
            if new_encoding is not None:
                guest.set_face_encoding(new_encoding)
                guest.save()
                
                self.stdout.write(
                    self.style.SUCCESS(f"✓ Converted: Length: {len(new_encoding)}, Range: {np.min(new_encoding):.3f} to {np.max(new_encoding):.3f}")
                )
            else:
                self.stdout.write(self.style.ERROR("✗ Conversion failed"))
                
        except Guest.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Guest {guest_id} not found"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {e}"))
    
    def convert_all_guests(self):
        """Convert all guests with old encodings"""
        guests = Guest.objects.exclude(face_encoding='')
        total = guests.count()
        
        self.stdout.write(f"Converting {total} guests...")
        
        success = 0
        errors = 0
        
        for guest in guests:
            try:
                old_encoding = guest.get_face_encoding()
                
                # Check if it's old format (face_recognition usually has wider range)
                enc_range = np.max(old_encoding) - np.min(old_encoding)
                
                if enc_range > 1.0:  # Likely old format
                    new_encoding = convert_old_encoding_to_new_format(old_encoding)
                    
                    if new_encoding is not None:
                        guest.set_face_encoding(new_encoding)
                        guest.save()
                        success += 1
                        self.stdout.write(f"✓ {guest.name}")
                    else:
                        errors += 1
                        self.stdout.write(f"✗ {guest.name} - conversion failed")
                else:
                    self.stdout.write(f"- {guest.name} - already new format")
                    
            except Exception as e:
                errors += 1
                self.stdout.write(f"✗ {guest.name} - error: {e}")
        
        self.stdout.write(f"\nSummary: {success} converted, {errors} errors")
    
    def test_compatibility(self):
        """Test if converted encodings work with face service"""
        try:
            from guest_system.services.face_recognition_service import FaceRecognitionService
            
            face_service = FaceRecognitionService()
            self.stdout.write(f"Face service loaded with {len(face_service.known_face_encodings)} faces")
            
            # Test similarity calculation between first two faces
            if len(face_service.known_face_encodings) >= 2:
                enc1 = face_service.known_face_encodings[0]
                enc2 = face_service.known_face_encodings[1]
                
                similarity = face_service._cosine_similarity(enc1, enc2)
                self.stdout.write(f"Sample similarity: {similarity:.4f}")
                
                if 0 <= similarity <= 1:
                    self.stdout.write(self.style.SUCCESS("✓ Similarity calculation working"))
                else:
                    self.stdout.write(self.style.WARNING(f"? Unusual similarity value: {similarity}"))
            else:
                self.stdout.write("Not enough faces to test similarity")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error testing compatibility: {e}"))
    
    def print_help(self):
        self.stdout.write("Usage:")
        self.stdout.write("  python manage.py convert_encodings --guest-id 1")
        self.stdout.write("  python manage.py convert_encodings --all")
        self.stdout.write("  python manage.py convert_encodings --test-compatibility")

# Function untuk update Face Recognition Service agar support backward compatibility
def update_face_recognition_service_with_backward_compatibility():
    """
    Update existing face recognition service to handle both old and new encodings
    """
    additional_methods = '''
    
    def _detect_encoding_format(self, encoding):
        """Detect if encoding is old format (face_recognition) or new format"""
        try:
            enc_array = np.array(encoding, dtype=np.float32)
            enc_range = np.max(enc_array) - np.min(enc_array)
            
            # Old face_recognition encodings typically have wider range
            if enc_range > 1.0:
                return 'old'
            else:
                return 'new'
        except:
            return 'unknown'
    
    def _normalize_encoding_for_comparison(self, encoding):
        """Normalize encoding to consistent format for comparison"""
        try:
            enc_array = np.array(encoding, dtype=np.float32)
            
            # Detect format
            format_type = self._detect_encoding_format(encoding)
            
            if format_type == 'old':
                # Convert old format to new
                return convert_old_encoding_to_new_format(encoding)
            else:
                # Already new format or unknown, return as-is
                return enc_array
                
        except Exception as e:
            logger.error(f"Error normalizing encoding: {e}")
            return np.array(encoding, dtype=np.float32)
    
    def recognize_face_with_backward_compatibility(self, face_encoding):
        """Enhanced recognize_face with backward compatibility"""
        if len(self.known_face_encodings) == 0:
            return None, None
        
        try:
            # Normalize input encoding
            normalized_input = self._normalize_encoding_for_comparison(face_encoding)
            
            best_similarity = -1
            best_match_index = -1
            
            for i, known_encoding in enumerate(self.known_face_encodings):
                # Normalize known encoding
                normalized_known = self._normalize_encoding_for_comparison(known_encoding)
                
                # Calculate similarity
                similarity = self._cosine_similarity(normalized_input, normalized_known)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = i
            
            # Use lower threshold for converted encodings
            adaptive_threshold = self.threshold * 0.8  # Reduce threshold by 20%
            
            if best_similarity > adaptive_threshold:
                guest_id = self.known_face_ids[best_match_index]
                guest_name = self.known_face_names[best_match_index]
                logger.info(f"Face recognized (backward compat): {guest_name} (similarity: {best_similarity:.3f})")
                return guest_id, guest_name
            else:
                logger.info(f"Face not recognized (similarity: {best_similarity:.3f}, threshold: {adaptive_threshold:.3f})")
                return None, None
                
        except Exception as e:
            logger.error(f"Error in backward compatible recognition: {e}")
            return None, None
    '''
    
    return additional_methods