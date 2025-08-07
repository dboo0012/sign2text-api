"""
Pose estimation processor using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional
from .models import ProcessingResult, Keypoints, FrameInfo
from .logger import setup_logger
from .config import settings

logger = setup_logger(__name__)

class PoseProcessor:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self):
        """Initialize MediaPipe Holistic processor"""
        try:
            self.holistic = mp.solutions.holistic.Holistic(
                **settings.MEDIAPIPE_CONFIG
            )
            logger.info("MediaPipe Holistic processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def extract_keypoints(self, results) -> Keypoints:
        """
        Extract keypoints from MediaPipe results
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            Keypoints object containing all landmark data
        """
        keypoints = Keypoints()
        
        # Extract pose landmarks
        if results.pose_landmarks:
            keypoints.pose = [
                [lm.x, lm.y, lm.z, lm.visibility] 
                for lm in results.pose_landmarks.landmark
            ]
        
        # Extract face landmarks
        if results.face_landmarks:
            keypoints.face = [
                [lm.x, lm.y, lm.z] 
                for lm in results.face_landmarks.landmark
            ]
        
        # Extract left hand landmarks
        if results.left_hand_landmarks:
            keypoints.left_hand = [
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ]
        
        # Extract right hand landmarks
        if results.right_hand_landmarks:
            keypoints.right_hand = [
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ]
        
        return keypoints
    
    def create_frame_info(self, frame: np.ndarray, results) -> FrameInfo:
        """
        Create frame information from processing results
        
        Args:
            frame: Input frame
            results: MediaPipe results
            
        Returns:
            FrameInfo object with frame metadata
        """
        height, width = frame.shape[:2]
        
        return FrameInfo(
            width=width,
            height=height,
            has_pose=results.pose_landmarks is not None,
            has_face=results.face_landmarks is not None,
            has_left_hand=results.left_hand_landmarks is not None,
            has_right_hand=results.right_hand_landmarks is not None
        )
    
    def process_frame(self, frame: np.ndarray) -> ProcessingResult:
        """
        Process a single frame and extract keypoints
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            ProcessingResult with keypoints and metadata
        """
        try:
            # Validate input
            if frame is None:
                return ProcessingResult(
                    success=False,
                    error="Input frame is None"
                )
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            # Extract keypoints
            keypoints = self.extract_keypoints(results)
            
            # Create frame info
            frame_info = self.create_frame_info(frame, results)
            
            return ProcessingResult(
                success=True,
                keypoints=keypoints,
                frame_info=frame_info
            )
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    def decode_base64_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """
        Decode base64 encoded image data
        
        Args:
            frame_data: Base64 encoded image string
            
        Returns:
            Decoded frame as numpy array or None if failed
        """
        try:
            import base64
            
            # Remove data URL prefix if present
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            # Decode base64
            img_bytes = base64.b64decode(frame_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to decode base64 frame: {e}")
            return None
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        try:
            if hasattr(self, 'holistic'):
                self.holistic.close()
                logger.info("MediaPipe resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 