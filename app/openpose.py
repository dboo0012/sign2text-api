"""
OpenPose keypoint extraction module
Handles initialization and processing of frames using OpenPose
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any
from .logger import setup_logger
from .models import OpenPoseData, OpenPosePerson
from .config import settings as config
from openpose import pyopenpose as op

logger = setup_logger(__name__)

class OpenPoseExtractor:
    """
    Manages OpenPose initialization and keypoint extraction from frames
    """
    
    def __init__(self):
        """Initialize OpenPose wrapper"""
        self.op_wrapper = None
        self.is_initialized = False
        self._initialize_openpose()
    
    def _initialize_openpose(self):
        """Initialize OpenPose with configuration from settings"""
        try:
            logger.info("Initializing OpenPose...")
            
            # Configure OpenPose using settings
            params = dict(config.OPENPOSE_CONFIG)
            
            logger.info(f"OpenPose config: {params}")
            
            # Create OpenPose wrapper
            self.op_wrapper = op.WrapperPython()
            self.op_wrapper.configure(params)
            self.op_wrapper.start()
            
            self.is_initialized = True
            logger.info("✓ OpenPose initialized successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ OpenPose not available - import failed: {e}")
            logger.warning("WebSocket will still work but keypoint extraction will be disabled")
            self.is_initialized = False
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize OpenPose: {e}")
            logger.warning("WebSocket will still work but keypoint extraction will be disabled")
            self.is_initialized = False
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[OpenPoseData]:
        """
        Extract keypoints from a frame using OpenPose
        
        Args:
            frame: OpenCV image frame (BGR format)
            
        Returns:
            OpenPoseData object with extracted keypoints, or None if extraction fails
        """
        if not self.is_initialized:
            logger.warning("OpenPose is not initialized")
            return None
        
        if frame is None:
            logger.warning("Invalid frame provided to OpenPose")
            return None
        
        try:
            # Create Datum object and process the frame
            datum = op.Datum()
            datum.cvInputData = frame
            self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Check if any people were detected
            if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
                return OpenPoseData(version=1.3, people=[])

            # TO DO: Remove debug logs in production
            logger.info("✓ Extracted keypoints: " + str(datum.poseKeypoints.shape))
            
            # Optimize: Only flatten and convert to list when necessary
            # Pre-check arrays exist before processing
            pose_kp = datum.poseKeypoints[0].flatten().tolist() if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0 else []
            face_kp = datum.faceKeypoints[0].flatten().tolist() if datum.faceKeypoints is not None and len(datum.faceKeypoints) > 0 else []
            hand_left_kp = datum.handKeypoints[0][0].flatten().tolist() if datum.handKeypoints is not None and len(datum.handKeypoints) > 0 else []
            hand_right_kp = datum.handKeypoints[1][0].flatten().tolist() if datum.handKeypoints is not None and len(datum.handKeypoints) > 1 else []
            
            return OpenPoseData(
                version=1.3, 
                people=[OpenPosePerson(
                    person_id=[0],
                    pose_keypoints_2d=pose_kp,
                    face_keypoints_2d=face_kp,
                    hand_left_keypoints_2d=hand_left_kp,
                    hand_right_keypoints_2d=hand_right_kp,
                    pose_keypoints_3d=[],
                    face_keypoints_3d=[],
                    hand_left_keypoints_3d=[],
                    hand_right_keypoints_3d=[]
                )]
            )
            
        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get OpenPose extractor statistics
        
        Returns:
            Dictionary with extractor stats
        """
        return {
            "initialized": self.is_initialized,
            "available": self.op_wrapper is not None
        }
