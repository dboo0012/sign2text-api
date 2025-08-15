"""
Keypoints processing module for sign language recognition
This module handles the core business logic for analyzing keypoints data
"""
from typing import Dict, Any, Optional
from .models import Keypoints, FrameInfo, ProcessingResult
from .logger import setup_logger
from .config import settings

logger = setup_logger(__name__)

class KeypointsProcessor:
    """
    Core processor for analyzing keypoints data from client-side MediaPipe extraction
    
    This class contains the main business logic for:
    - Sign language recognition
    - Gesture analysis
    - Temporal pattern detection
    - Machine learning inference
    """
    
    def __init__(self):
        """Initialize the keypoints processor"""
        self.processed_sequences = 0
        self.session_start_time = None
        logger.info("KeypointsProcessor initialized")
    
    async def process_keypoints(
        self, 
        keypoints: Keypoints, 
        frame_info: Optional[FrameInfo] = None,
        timestamp: Optional[float] = None
    ) -> ProcessingResult:
        """
        Main processing method for keypoints analysis
        
        Args:
            keypoints: Keypoints data from client-side processing
            frame_info: Optional frame metadata
            timestamp: Optional timestamp of the data
            
        Returns:
            ProcessingResult with analysis outcomes
        """
        try:
            self.processed_sequences += 1
            
            # Log incoming data for debugging
            self._log_keypoints_info(keypoints, frame_info)
            
            # Validate keypoints data
            validation_result = self._validate_keypoints(keypoints)
            if not validation_result["valid"]:
                return ProcessingResult(
                    success=False,
                    error=f"Keypoints validation failed: {validation_result['error']}"
                )
            
            # Core processing logic
            analysis_result = await self._analyze_keypoints(keypoints, frame_info, timestamp)
            
            # Check if any meaningful data was processed
            if not analysis_result:
                return ProcessingResult(
                    success=False,
                    error="No meaningful keypoints data to process"
                )
            
            return ProcessingResult(
                success=True,
                processed_keypoints=keypoints,
                analysis_result=analysis_result
            )
            
        except Exception as e:
            logger.error(f"Error processing keypoints: {e}")
            return ProcessingResult(
                success=False,
                error=f"Processing error: {str(e)}"
            )
    
    async def _analyze_keypoints(
        self, 
        keypoints: Keypoints, 
        frame_info: Optional[FrameInfo] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Core analysis logic for keypoints data
        
        This is where you'll implement your main business logic:
        - Sign language recognition algorithms
        - Gesture classification
        - Temporal sequence analysis
        - Machine learning model inference
        
        Args:
            keypoints: Validated keypoints data
            frame_info: Optional frame metadata
            timestamp: Optional timestamp
            
        Returns:
            Dictionary containing analysis results
        """
        analysis_result = {
            "processing_info": {
                "sequence_id": self.processed_sequences,
                "timestamp": timestamp,
                "keypoints_detected": {
                    "pose": len(keypoints.pose) > 0,
                    "face": len(keypoints.face) > 0,
                    "left_hand": len(keypoints.left_hand) > 0,
                    "right_hand": len(keypoints.right_hand) > 0
                }
            }
        }
        
        # Add frame info if available
        if frame_info:
            analysis_result["frame_info"] = {
                "dimensions": f"{frame_info.width}x{frame_info.height}",
                "detection_summary": {
                    "pose": frame_info.has_pose,
                    "face": frame_info.has_face,
                    "hands": frame_info.has_left_hand or frame_info.has_right_hand
                }
            }
        
        # Analyze pose data if available
        if keypoints.pose:
            pose_analysis = await self._analyze_pose_keypoints(keypoints.pose)
            analysis_result["pose_analysis"] = pose_analysis
        
        # Analyze hand data if available
        if keypoints.left_hand or keypoints.right_hand:
            hand_analysis = await self._analyze_hand_keypoints(
                keypoints.left_hand, 
                keypoints.right_hand
            )
            analysis_result["hand_analysis"] = hand_analysis
        
        # Analyze face data if available
        if keypoints.face:
            face_analysis = await self._analyze_face_keypoints(keypoints.face)
            analysis_result["face_analysis"] = face_analysis
        
        # TODO: Add your advanced processing logic here:
        # - Sign language recognition models
        # - Gesture classification
        # - Temporal pattern analysis
        # - Machine learning inference
        
        logger.info(f"Processed keypoints sequence #{self.processed_sequences}")
        return analysis_result
    
    async def _analyze_pose_keypoints(self, pose_landmarks) -> Dict[str, Any]:
        """
        Analyze pose landmarks for body posture and movement
        
        Args:
            pose_landmarks: List of pose landmark coordinates
            
        Returns:
            Dictionary with pose analysis results
        """
        # Example pose analysis - replace with your logic
        pose_analysis = {
            "landmarks_count": len(pose_landmarks),
            "confidence_stats": {},
            "body_position": "unknown"
        }
        
        if pose_landmarks:
            # Calculate average confidence (visibility values)
            confidences = [landmark[3] for landmark in pose_landmarks if len(landmark) >= 4]
            if confidences:
                pose_analysis["confidence_stats"] = {
                    "average": sum(confidences) / len(confidences),
                    "min": min(confidences),
                    "max": max(confidences)
                }
            
            # TODO: Add your pose analysis logic:
            # - Body position detection
            # - Movement analysis
            # - Posture classification
            
        return pose_analysis
    
    async def _analyze_hand_keypoints(self, left_hand_landmarks, right_hand_landmarks) -> Dict[str, Any]:
        """
        Analyze hand landmarks for gesture recognition
        
        Args:
            left_hand_landmarks: Left hand landmark coordinates
            right_hand_landmarks: Right hand landmark coordinates
            
        Returns:
            Dictionary with hand analysis results
        """
        hand_analysis = {
            "left_hand": {
                "detected": len(left_hand_landmarks) > 0,
                "landmarks_count": len(left_hand_landmarks)
            },
            "right_hand": {
                "detected": len(right_hand_landmarks) > 0,
                "landmarks_count": len(right_hand_landmarks)
            },
            "gesture_detected": None
        }
        
        # TODO: Add your hand gesture analysis logic:
        # - Hand shape classification
        # - Finger position analysis
        # - Sign language gesture recognition
        # - Hand movement tracking
        
        return hand_analysis
    
    async def _analyze_face_keypoints(self, face_landmarks) -> Dict[str, Any]:
        """
        Analyze face landmarks for facial expressions and lip reading
        
        Args:
            face_landmarks: Face landmark coordinates
            
        Returns:
            Dictionary with face analysis results
        """
        face_analysis = {
            "landmarks_count": len(face_landmarks),
            "facial_expression": "neutral",
            "mouth_movement": "unknown"
        }
        
        # TODO: Add your facial analysis logic:
        # - Facial expression recognition
        # - Lip reading for sign language
        # - Eye gaze direction
        # - Head pose estimation
        
        return face_analysis
    
    def _validate_keypoints(self, keypoints: Keypoints) -> Dict[str, Any]:
        """
        Validate keypoints data structure and content
        
        Args:
            keypoints: Keypoints data to validate
            
        Returns:
            Validation result with success status and error message
        """
        try:
            # Check basic structure
            if not isinstance(keypoints.pose, list):
                return {"valid": False, "error": "Pose landmarks must be a list"}
            
            if not isinstance(keypoints.face, list):
                return {"valid": False, "error": "Face landmarks must be a list"}
            
            if not isinstance(keypoints.left_hand, list):
                return {"valid": False, "error": "Left hand landmarks must be a list"}
            
            if not isinstance(keypoints.right_hand, list):
                return {"valid": False, "error": "Right hand landmarks must be a list"}
            
            # Validate landmark counts if present
            if keypoints.pose and len(keypoints.pose) > settings.KEYPOINTS_CONFIG["max_pose_landmarks"]:
                return {"valid": False, "error": f"Too many pose landmarks: {len(keypoints.pose)}"}
            
            if keypoints.face and len(keypoints.face) > settings.KEYPOINTS_CONFIG["max_face_landmarks"]:
                return {"valid": False, "error": f"Too many face landmarks: {len(keypoints.face)}"}
            
            if keypoints.left_hand and len(keypoints.left_hand) > settings.KEYPOINTS_CONFIG["max_hand_landmarks"]:
                return {"valid": False, "error": f"Too many left hand landmarks: {len(keypoints.left_hand)}"}
            
            if keypoints.right_hand and len(keypoints.right_hand) > settings.KEYPOINTS_CONFIG["max_hand_landmarks"]:
                return {"valid": False, "error": f"Too many right hand landmarks: {len(keypoints.right_hand)}"}
            
            # Validate landmark coordinate format
            validation_errors = []
            
            # Check pose landmarks format [x, y, z, visibility]
            for i, landmark in enumerate(keypoints.pose):
                if len(landmark) != 4:
                    validation_errors.append(f"Pose landmark {i} should have 4 coordinates [x,y,z,visibility]")
            
            # Check face landmarks format [x, y, z]
            for i, landmark in enumerate(keypoints.face):
                if len(landmark) != 3:
                    validation_errors.append(f"Face landmark {i} should have 3 coordinates [x,y,z]")
            
            # Check hand landmarks format [x, y, z]
            for i, landmark in enumerate(keypoints.left_hand):
                if len(landmark) != 3:
                    validation_errors.append(f"Left hand landmark {i} should have 3 coordinates [x,y,z]")
            
            for i, landmark in enumerate(keypoints.right_hand):
                if len(landmark) != 3:
                    validation_errors.append(f"Right hand landmark {i} should have 3 coordinates [x,y,z]")
            
            if validation_errors:
                return {"valid": False, "error": "; ".join(validation_errors[:3])}  # Limit error messages
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _log_keypoints_info(self, keypoints: Keypoints, frame_info: Optional[FrameInfo] = None):
        """Log keypoints information for debugging"""
        logger.info(
            f"Processing keypoints - Pose: {len(keypoints.pose)} points, "
            f"Face: {len(keypoints.face)} points, "
            f"Left hand: {len(keypoints.left_hand)} points, "
            f"Right hand: {len(keypoints.right_hand)} points"
        )
        
        if frame_info:
            logger.info(f"Frame info: {frame_info.width}x{frame_info.height}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "sequences_processed": self.processed_sequences,
            "session_start_time": self.session_start_time,
            "processor_status": "active"
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processed_sequences = 0
        logger.info("Processing statistics reset")
