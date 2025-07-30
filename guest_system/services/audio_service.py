import os
import sys
import io
import threading
import time
from contextlib import redirect_stdout, redirect_stderr

class AudioService:
    def __init__(self):
        """Initialize audio service with lazy pygame loading"""
        self.pygame = None
        self.audio_enabled = False
        self._init_attempted = False
        self._last_speech_time = 0
        self._min_speech_interval = 2.0  # Minimum 2 seconds between speeches
        
        # Set pygame environment variables but don't import yet
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        print("🔊 Audio service initialized (pygame will load when needed)")
    
    def _lazy_init_pygame(self):
        """Lazy initialization of pygame when first needed"""
        if self._init_attempted:
            return self.audio_enabled
            
        self._init_attempted = True
        
        try:
            # Suppress pygame output during import
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                # Import pygame only when needed
                import pygame
                self.pygame = pygame
                
                # Initialize mixer
                self.pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
                self.pygame.mixer.init()
                
            self.audio_enabled = True
            print("🔊 Pygame audio system loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Audio system initialization failed: {e}")
            self.audio_enabled = False
            
        return self.audio_enabled
    
    def _can_speak(self):
        """Check if enough time has passed since last speech"""
        current_time = time.time()
        if current_time - self._last_speech_time < self._min_speech_interval:
            return False
        return True
    
    def speak(self, text, language='id'):
        """Convert text to speech and play it"""
        # Check if we can speak (prevent double audio)
        if not self._can_speak():
            print(f"🔇 Audio blocked (too frequent) - Would speak: {text}")
            return
            
        # Initialize pygame only when speak is called
        if not self._lazy_init_pygame():
            print(f"🔇 Audio disabled - Would speak: {text}")
            return
            
        def _speak():
            try:
                # Import gTTS only when needed
                from gtts import gTTS
                
                # Generate TTS audio
                tts = gTTS(text=text, lang=language, slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                # Play audio using pygame
                self.pygame.mixer.music.load(audio_buffer)
                self.pygame.mixer.music.play()
                
                # Wait for playback to complete
                while self.pygame.mixer.music.get_busy():
                    self.pygame.time.wait(100)
                    
                print(f"🔊 Spoke: {text}")
                    
            except Exception as e:
                print(f"🔇 Text-to-speech error: {e}")
        
        # Update last speech time
        self._last_speech_time = time.time()
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def welcome_and_instruct_returning_guest(self, guest_name):
        """Welcome returning guest with gesture instruction (SINGLE MESSAGE)"""
        message = f"Selamat datang kembali, {guest_name}. Silakan tunjukkan gestur: angka 1 untuk pelayanan data, angka 2 untuk menemui pegawai."
        self.speak(message)
    
    def welcome_and_instruct_new_guest(self):
        """Welcome new guest and ask for registration (SINGLE MESSAGE)"""
        message = "Selamat datang! Anda adalah tamu baru. Silakan isi data diri Anda terlebih dahulu."
        self.speak(message)
    
    def instruct_gesture_only(self):
        """Give gesture instruction only (for after registration)"""
        message = "Silakan tunjukkan gestur tangan: angka 1 untuk pelayanan data, angka 2 untuk menemui pegawai."
        self.speak(message)
    
    def direct_to_pst(self):
        """Direct to PST room"""
        message = "Terima kasih. Silakan menuju ruang PST untuk pelayanan data."
        self.speak(message)
    
    def direct_to_staff(self):
        """Direct to staff"""
        message = "Terima kasih. Silakan langsung menemui pegawai atau kepala kantor."
        self.speak(message)
    
    def registration_success(self):
        """Confirm registration success"""
        message = "Pendaftaran berhasil. Sekarang silakan tunjukkan gestur tangan untuk memilih layanan."
        self.speak(message)
    
    def guestbook_complete(self, purpose):
        """Confirm guest book completion"""
        if purpose == 'data_service':
            message = "Buku tamu telah diisi. Silakan menuju ruang PST untuk pelayanan data."
        else:
            message = "Buku tamu telah diisi. Silakan menemui pegawai atau kepala kantor."
        self.speak(message)
    
    # Legacy methods (DEPRECATED - use new single-message methods above)
    def welcome_guest(self, guest_name):
        """DEPRECATED: Use welcome_and_instruct_returning_guest instead"""
        print(f"⚠️ DEPRECATED: welcome_guest called. Use welcome_and_instruct_returning_guest instead")
        self.welcome_and_instruct_returning_guest(guest_name)
    
    def welcome_new_guest(self):
        """DEPRECATED: Use welcome_and_instruct_new_guest instead"""
        print(f"⚠️ DEPRECATED: welcome_new_guest called. Use welcome_and_instruct_new_guest instead")
        self.welcome_and_instruct_new_guest()
    
    def give_gesture_instruction(self):
        """DEPRECATED: This was causing double audio"""
        print(f"⚠️ DEPRECATED: give_gesture_instruction called. This causes double audio!")
        # Don't actually speak - this was causing the double audio issue
        pass
    
    def __del__(self):
        """Cleanup pygame mixer"""
        try:
            if self.audio_enabled and self.pygame:
                self.pygame.mixer.quit()
        except:
            pass

# Alternative: Mock Audio Service for development/testing
class MockAudioService:
    """Mock audio service that just prints messages (for development)"""
    
    def __init__(self):
        print("🔇 Mock audio service initialized (no actual sound)")
    
    def speak(self, text, language='id'):
        print(f"🔊 [MOCK AUDIO] Would speak: {text}")
    
    def welcome_and_instruct_returning_guest(self, guest_name):
        self.speak(f"Selamat datang kembali, {guest_name}. Silakan tunjukkan gestur untuk memilih layanan.")
    
    def welcome_and_instruct_new_guest(self):
        self.speak("Selamat datang! Silakan isi data diri Anda terlebih dahulu.")
    
    def instruct_gesture_only(self):
        self.speak("Silakan tunjukkan gestur tangan untuk memilih layanan.")
    
    def direct_to_pst(self):
        self.speak("Silakan menuju ruang PST.")
    
    def direct_to_staff(self):
        self.speak("Silakan menemui pegawai.")
    
    def registration_success(self):
        self.speak("Pendaftaran berhasil.")
    
    def guestbook_complete(self, purpose):
        self.speak("Buku tamu telah diisi.")
    
    # Legacy methods
    def welcome_guest(self, guest_name):
        self.welcome_and_instruct_returning_guest(guest_name)
    
    def welcome_new_guest(self):
        self.welcome_and_instruct_new_guest()
    
    def give_gesture_instruction(self):
        pass  # Do nothing to prevent double audio

# Auto-select audio service based on environment
def get_audio_service():
    """Get audio service - mock for development, real for production"""
    import os
    
    # Check environment variable for audio preference
    audio_mode = os.environ.get('AUDIO_MODE', 'auto').lower()
    
    if audio_mode == 'mock':
        return MockAudioService()
    elif audio_mode == 'real':
        return AudioService()
    else:  # auto mode
        # Use mock for development, real for production
        debug_mode = os.environ.get('DJANGO_DEBUG', 'True').lower() == 'true'
        if debug_mode:
            return MockAudioService()
        else:
            return AudioService()

# Export the service
audio_service = get_audio_service()