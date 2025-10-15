"""
Keypoints processing module for sign language recognition
This module handles the core business logic for analyzing keypoints data
"""
from typing import Dict, Any, Optional, List
from .models import ProcessingResult, StructuredKeypoints
from .logger import setup_logger
from .model_utils import run_model_inference
import warnings

logger = setup_logger(__name__)

class KeypointsProcessor:
    """Simple processor that buffers keypoints and runs model inference"""
    
    def __init__(self):
        """Initialize the keypoints processor"""
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        self.processed_sequences = 0
        
        # Keypoints buffer - stores raw OpenPose format dicts
        self.keypoints_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 32
        self.min_sequence_length = 16
        
        logger.info(f"KeypointsProcessor initialized - buffer size: {self.max_buffer_size}, min sequence: {self.min_sequence_length}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "sequences_processed": self.processed_sequences,
            "processor_status": "active",
            "buffer_size": len(self.keypoints_buffer),
            "min_sequence_length": self.min_sequence_length
        }
    
    async def process_keypoints(
        self, 
        keypoint_data: Dict[str, Any],
        frame_info: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> ProcessingResult:
        """
        Process keypoints: add to buffer and run inference when ready
        
        Args:
            keypoint_data: Raw OpenPose format dict {"version": 1.3, "people": [...]}
            frame_info: Optional frame metadata
            timestamp: Optional timestamp
            
        Returns:
            ProcessingResult with analysis outcomes
        """
        try:
            self.processed_sequences += 1
            
            # Add to buffer
            self.keypoints_buffer.append(keypoint_data)
            
            # Keep buffer size manageable (rolling window)
            if len(self.keypoints_buffer) > self.max_buffer_size:
                self.keypoints_buffer.pop(0)
            
            # Get detection summary for logging
            detection_summary = self._get_detection_summary(keypoint_data)
            
            # Calculate feature vector size (x, y for each valid point)
            feature_vector_size = detection_summary.get("total_points", 0) * 2
            
            # Build analysis result matching expected format
            analysis_result = {
                "processing_info": {
                    "sequence_id": self.processed_sequences,
                    "timestamp": timestamp,
                    "detection_summary": detection_summary
                },
                "feature_vector_size": feature_vector_size,
                "model_ready": True,
                "buffer_size": len(self.keypoints_buffer) - self.min_sequence_length if len(self.keypoints_buffer) >= self.min_sequence_length else len(self.keypoints_buffer)
            }
            
            # Run model inference if buffer is full
            if len(self.keypoints_buffer) >= self.min_sequence_length:
                try:
                    # Use the last N frames for prediction
                    sequence = self.keypoints_buffer[-self.min_sequence_length:]
                    prediction_result = run_model_inference(sequence)
                    
                    analysis_result["prediction"] = prediction_result
                    
                    if prediction_result.get("success"):
                        logger.info(f"✅ Prediction: {prediction_result.get('text', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"Model inference error: {e}")
                    analysis_result["prediction"] = {
                        "success": False,
                        "text": "",
                        "confidence": 0.0,
                        "error": str(e)
                    }
            else:
                # Still buffering
                analysis_result["prediction"] = {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "status": "buffering",
                    "frames_needed": self.min_sequence_length - len(self.keypoints_buffer)
                }
            
            return ProcessingResult(
                success=True,
                analysis_result=analysis_result
            )
            
        except Exception as e:
            logger.error(f"Error processing keypoints: {e}")
            return ProcessingResult(
                success=False,
                error=f"Processing error: {str(e)}"
            )
    
    def _get_detection_summary(self, keypoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract detection summary from raw OpenPose data
        
        Args:
            keypoint_data: Raw OpenPose format dict
            
        Returns:
            Detection summary dict
        """
        try:
            if not keypoint_data.get("people"):
                return {
                    "pose_points": 0,
                    "face_points": 0,
                    "left_hand_points": 0,
                    "right_hand_points": 0,
                    "total_points": 0,
                    "has_pose": False,
                    "has_face": False,
                    "has_hands": False
                }
            
            person = keypoint_data["people"][0]
            
            # Count non-zero keypoints (groups of 3: x, y, confidence)
            def count_valid_points(kp_list):
                if not kp_list:
                    return 0
                count = 0
                for i in range(0, len(kp_list), 3):
                    if i + 2 < len(kp_list) and kp_list[i+2] > 0.1:  # confidence threshold
                        count += 1
                return count
            
            pose_points = count_valid_points(person.get("pose_keypoints_2d", []))
            face_points = count_valid_points(person.get("face_keypoints_2d", []))
            left_hand_points = count_valid_points(person.get("hand_left_keypoints_2d", []))
            right_hand_points = count_valid_points(person.get("hand_right_keypoints_2d", []))
            
            return {
                "pose_points": pose_points,
                "face_points": face_points,
                "left_hand_points": left_hand_points,
                "right_hand_points": right_hand_points,
                "total_points": pose_points + face_points + left_hand_points + right_hand_points,
                "has_pose": pose_points > 0,
                "has_face": face_points > 0,
                "has_hands": left_hand_points > 0 or right_hand_points > 0
            }
        except Exception as e:
            logger.warning(f"Error getting detection summary: {e}")
            return {
                "pose_points": 0,
                "face_points": 0,
                "left_hand_points": 0,
                "right_hand_points": 0,
                "total_points": 0,
                "has_pose": False,
                "has_face": False,
                "has_hands": False
            }
