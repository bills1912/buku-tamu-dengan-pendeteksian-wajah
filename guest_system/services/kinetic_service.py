import mediapipe as mp
import cv2
import numpy as np

class KineticRecognitionService:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_gesture(self, frame):
        """Detect hand gesture from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                
                # Recognize gesture
                gesture = self._recognize_number_gesture(landmarks)
                return gesture, hand_landmarks
        
        return None, None
    
    def _recognize_number_gesture(self, landmarks):
        """Recognize number gestures (1 or 2)"""
        if len(landmarks) < 21:
            return None
        
        # Finger tip and PIP joint indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        # Check which fingers are extended
        fingers_up = []
        
        # Thumb (special case - check x coordinate)
        if landmarks[finger_tips[0]][0] > landmarks[finger_pips[0]][0]:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers (check y coordinate)
        for i in range(1, 5):
            if landmarks[finger_tips[i]][1] < landmarks[finger_pips[i]][1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Count extended fingers
        total_fingers = sum(fingers_up)
        
        # Recognize specific gestures
        if total_fingers == 1 and fingers_up[1] == 1:  # Only index finger
            return 1
        elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:  # Index and middle
            return 2
        
        return None