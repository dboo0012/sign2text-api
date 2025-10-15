# class KeypointsProcessor:
#     def __init__(self):
#         # ...existing code...
#         self.keypoints_buffer: List[OpenPoseData] = []
#         self.buffer_update_interval = 5  # Only update buffer every 5 frames
#         self.frames_since_update = 0
#         # ...existing code...

#     async def process_keypoints(
#         self, 
#         keypoint_data: Union[Keypoints, StructuredKeypoints, OpenPoseData, Dict[str, Any]], 
#         frame_info: Optional[FrameInfo] = None,
#         timestamp: Optional[float] = None
#     ) -> ProcessingResult:
#         """Process keypoints with batching optimization"""
#         try:
#             self.processed_sequences += 1
            
#             # Add to buffer
#             if isinstance(keypoint_data, OpenPoseData):
#                 self.keypoints_buffer.append(keypoint_data)
#                 self.frames_since_update += 1
                
#                 # Keep buffer size manageable
#                 if len(self.keypoints_buffer) > self.max_buffer_size:
#                     self.keypoints_buffer.pop(0)
                
#                 # Only run model inference periodically
#                 if self.frames_since_update >= self.buffer_update_interval and \
#                    len(self.keypoints_buffer) >= self.min_sequence_length:
#                     self.frames_since_update = 0
                    
#                     # Run model inference
#                     prediction_result = run_model_inference(
#                         self.keypoints_buffer[-self.min_sequence_length:]
#                     )
                    
#                     return ProcessingResult(
#                         success=True,
#                         analysis_result={
#                             "prediction": prediction_result,
#                             "buffer_size": len(self.keypoints_buffer)
#                         }
#                     )
            
#             # Return basic acknowledgment without running model
#             return ProcessingResult(
#                 success=True,
#                 analysis_result={
#                     "buffered": True,
#                     "buffer_size": len(self.keypoints_buffer),
#                     "frames_until_inference": self.buffer_update_interval - self.frames_since_update
#                 }
#             )
            
#         except Exception as e:
#             logger.error(f"Error in process_keypoints: {e}")
#             return ProcessingResult(success=False, error=str(e))
"""
Keypoints processing module for sign language recognition
This module handles the core business logic for analyzing keypoints data
"""
from typing import Dict, Any, Optional, Union, List
from .models import (
    Keypoints, StructuredKeypoints, OpenPoseData, 
    FrameInfo, ProcessingResult
)
from .logger import setup_logger
from .config import settings
from .model_utils import run_model_inference
import os
import warnings

logger = setup_logger(__name__)

class KeypointsProcessor:
    """
    Core processor for analyzing keypoints data from client-side keypoints extraction
    
    This class contains the main business logic for:
    - Sign language recognition
    - Gesture analysis
    - Temporal pattern detection
    - Machine learning inference
    """
    
    def __init__(self):
        """Initialize the keypoints processor"""
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        self.processed_sequences = 0
        self.session_start_time = None
        
        # Keypoints buffer for sequence processing
        self.keypoints_buffer: List[OpenPoseData] = []
        self.max_buffer_size = 32
        self.min_sequence_length = 16
        
        logger.info("KeypointsProcessor initialized - model will be loaded on demand")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "sequences_processed": self.processed_sequences,
            "session_start_time": self.session_start_time,
            "processor_status": "active",
            "buffer_size": len(self.keypoints_buffer),
            "min_sequence_length": self.min_sequence_length
        }
    
    async def process_keypoints(
        self, 
        keypoint_data: Union[Keypoints, StructuredKeypoints, OpenPoseData, Dict[str, Any]], 
        frame_info: Optional[FrameInfo] = None,
        timestamp: Optional[float] = None
    ) -> ProcessingResult:
        """
        Main processing method for keypoints analysis
        
        Args:
            keypoint_data: Keypoints data (legacy, structured, or raw OpenPose format)
            frame_info: Optional frame metadata
            timestamp: Optional timestamp of the data
            
        Returns:
            ProcessingResult with analysis outcomes
        """
        try:
            self.processed_sequences += 1
            
            # Convert input to structured format for consistent processing
            structured_keypoints = self._convert_to_structured(keypoint_data)
            if not structured_keypoints:
                return ProcessingResult(
                    success=False,
                    error="Failed to convert keypoints to structured format"
                )
            
            # Log incoming data for debugging
            self._log_keypoints_info(structured_keypoints, frame_info)
            
            # Validate keypoints data
            validation_result = self._validate_structured_keypoints(structured_keypoints)
            if not validation_result["valid"]:
                return ProcessingResult(
                    success=False,
                    error=f"Keypoints validation failed: {validation_result['error']}"
                )
            
            # Add to keypoints buffer for sequence processing
            if isinstance(keypoint_data, OpenPoseData):
                self.keypoints_buffer.append(keypoint_data)
                
                # Keep buffer size manageable
                if len(self.keypoints_buffer) > self.max_buffer_size:
                    self.keypoints_buffer.pop(0)
            
            # Core processing logic
            analysis_result = await self._analyze_structured_keypoints(
                structured_keypoints, frame_info, timestamp
            )
            
            # Check if any meaningful data was processed
            if not analysis_result:
                return ProcessingResult(
                    success=False,
                    error="No meaningful keypoints data to process"
                )
            
            return ProcessingResult(
                success=True,
                processed_keypoints=structured_keypoints,
                analysis_result=analysis_result
            )
            
        except Exception as e:
            logger.error(f"Error processing keypoints: {e}")
            return ProcessingResult(
                success=False,
                error=f"Processing error: {str(e)}"
            )
    
    def _convert_to_structured(
        self, 
        keypoint_data: Union[Keypoints, StructuredKeypoints, OpenPoseData, Dict[str, Any]]
    ) -> Optional[StructuredKeypoints]:
        """
        Convert various keypoint formats to structured format
        
        Args:
            keypoint_data: Input keypoints in various formats
            
        Returns:
            StructuredKeypoints or None if conversion fails
        """
        try:
            # Already structured
            if isinstance(keypoint_data, StructuredKeypoints):
                return keypoint_data
            
            # OpenPose format
            if isinstance(keypoint_data, OpenPoseData):
                if keypoint_data.people:
                    return StructuredKeypoints.from_openpose_person(keypoint_data.people[0])
                return StructuredKeypoints()  # Empty if no people
            
            # Dictionary (raw JSON)
            if isinstance(keypoint_data, dict):
                if "people" in keypoint_data:
                    openpose_data = OpenPoseData(**keypoint_data)
                    if openpose_data.people:
                        return StructuredKeypoints.from_openpose_person(openpose_data.people[0])
                return StructuredKeypoints()
            
            # Legacy Keypoints format
            if isinstance(keypoint_data, Keypoints):
                # Convert legacy format to structured
                from .models import PoseKeypoint, KeypointWithConfidence
                
                pose_points = []
                for landmark in keypoint_data.pose:
                    if len(landmark) >= 4:
                        pose_points.append(PoseKeypoint(
                            x=landmark[0], y=landmark[1], 
                            z=landmark[2], visibility=landmark[3]
                        ))
                
                face_points = []
                for landmark in keypoint_data.face:
                    if len(landmark) >= 3:
                        face_points.append(KeypointWithConfidence(
                            x=landmark[0], y=landmark[1], confidence=landmark[2]
                        ))
                
                left_hand_points = []
                for landmark in keypoint_data.left_hand:
                    if len(landmark) >= 3:
                        left_hand_points.append(KeypointWithConfidence(
                            x=landmark[0], y=landmark[1], confidence=landmark[2]
                        ))
                
                right_hand_points = []
                for landmark in keypoint_data.right_hand:
                    if len(landmark) >= 3:
                        right_hand_points.append(KeypointWithConfidence(
                            x=landmark[0], y=landmark[1], confidence=landmark[2]
                        ))
                
                return StructuredKeypoints(
                    pose=pose_points,
                    face=face_points,
                    left_hand=left_hand_points,
                    right_hand=right_hand_points
                )
            
            logger.warning(f"Unknown keypoint data format: {type(keypoint_data)}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting keypoints: {e}")
            return None
    
    async def _analyze_structured_keypoints(
        self, 
        keypoints: StructuredKeypoints, 
        frame_info: Optional[FrameInfo] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Core analysis logic for structured keypoints data
        
        Args:
            keypoints: Structured keypoints data
            frame_info: Optional frame metadata
            timestamp: Optional timestamp
            
        Returns:
            Dictionary containing analysis results
        """
        # Get detection summary
        detection_summary = keypoints.get_detection_summary()
        
        analysis_result = {
            "processing_info": {
                "sequence_id": self.processed_sequences,
                "timestamp": timestamp,
                "detection_summary": detection_summary
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
        
        # Prepare features for model input
        feature_vector = keypoints.to_numpy_features()
        analysis_result["feature_vector_size"] = len(feature_vector)
        
        # Model prediction logic
        if len(self.keypoints_buffer) >= self.min_sequence_length:
            try:
                # Use the last sequence for prediction
                sequence_for_prediction = self.keypoints_buffer[-self.min_sequence_length:]
                
                # Call your existing model inference
                prediction_result = run_model_inference(sequence_for_prediction)
                
                analysis_result["prediction"] = prediction_result
                analysis_result["model_ready"] = True
                analysis_result["buffer_size"] = len(self.keypoints_buffer)
                
                if prediction_result.get("success"):
                    logger.info(f"Model prediction: {prediction_result.get('text', 'N/A')}")
                else:
                    logger.warning(f"Model prediction failed: {prediction_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                analysis_result["prediction"] = {
                    "success": False,
                    "error": f"Model prediction failed: {str(e)}",
                    "text": ""
                }
                analysis_result["model_ready"] = False
        else:
            # Insufficient buffer
            analysis_result["model_ready"] = True
            analysis_result["buffer_size"] = len(self.keypoints_buffer)
            analysis_result["prediction"] = {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "status": "buffering",
                "frames_needed": self.min_sequence_length - len(self.keypoints_buffer)
            }
        
        logger.info(f"Processed structured keypoints sequence #{self.processed_sequences}")
        return analysis_result
    
    def _validate_structured_keypoints(self, keypoints: StructuredKeypoints) -> Dict[str, Any]:
        """
        Validate structured keypoints data
        
        Args:
            keypoints: Structured keypoints to validate
            
        Returns:
            Validation result with success status and error message
        """
        try:
            detection_summary = keypoints.get_detection_summary()
            
            # Check if we have any meaningful data
            if detection_summary["total_points"] == 0:
                return {"valid": False, "error": "No keypoints detected"}
            
            # Check reasonable bounds for keypoint counts
            if detection_summary["pose_points"] > settings.KEYPOINTS_CONFIG["max_pose_landmarks"]:
                return {"valid": False, "error": f"Too many pose landmarks: {detection_summary['pose_points']}"}
            
            if detection_summary["face_points"] > settings.KEYPOINTS_CONFIG["max_face_landmarks"]:
                return {"valid": False, "error": f"Too many face landmarks: {detection_summary['face_points']}"}
            
            if detection_summary["left_hand_points"] > settings.KEYPOINTS_CONFIG["max_hand_landmarks"]:
                return {"valid": False, "error": f"Too many left hand landmarks: {detection_summary['left_hand_points']}"}
            
            if detection_summary["right_hand_points"] > settings.KEYPOINTS_CONFIG["max_hand_landmarks"]:
                return {"valid": False, "error": f"Too many right hand landmarks: {detection_summary['right_hand_points']}"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _log_keypoints_info(self, keypoints: StructuredKeypoints, frame_info: Optional[FrameInfo] = None):
        """Log structured keypoints information for debugging"""
        summary = keypoints.get_detection_summary()
        logger.info(
            f"Processing structured keypoints - "
            f"Pose: {summary['pose_points']} points, "
            f"Face: {summary['face_points']} points, "
            f"Left hand: {summary['left_hand_points']} points, "
            f"Right hand: {summary['right_hand_points']} points, "
            f"Total: {summary['total_points']} points"
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
            "processor_status": "active",
            "buffer_size": len(self.keypoints_buffer),
            "min_sequence_length": self.min_sequence_length
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processed_sequences = 0
        logger.info("Processing statistics reset")
