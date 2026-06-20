import cv2
import mediapipe as mp

# MediaPipe solution selection
try:
    # Try the most common export location
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except:
    try:
        # Try the shorthand
        import mediapipe.solutions.pose as mp_pose
        import mediapipe.solutions.drawing_utils as mp_drawing
    except:
        # Fallback to the top-level solutions attribute
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

import numpy as np
from typing import List, Tuple, Optional, Dict

class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp_drawing

    def find_pose(self, img: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """Detects pose landmarks and returns the image and center point."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        center_point = None
        
        if results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            h, w, c = img.shape
            
            # Key landmarks for center calculation: shoulders and hips
            # MediaPipe indices: L_Shoulder=11, R_Shoulder=12, L_Hip=23, R_Hip=24
            indices = [11, 12, 23, 24]
            points = []
            for idx in indices:
                lm = landmarks[idx]
                points.append((lm.x * w, lm.y * h))
            
            points = np.array(points)
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            # Calculate bounding box size for auto-zoom (approximate)
            all_x = [lm.x * w for lm in landmarks]
            all_y = [lm.y * h for lm in landmarks]
            bbox_w = max(all_x) - min(all_x)
            bbox_h = max(all_y) - min(all_y)
            
            center_point = {
                "x": center_x,
                "y": center_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "confidence": results.pose_landmarks.landmark[0].visibility # Use nose visibility as proxy
            }
            
        return img, center_point

    def close(self):
        self.pose.close()
