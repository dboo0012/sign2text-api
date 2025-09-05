"""
Model utilities for sign language recognition
Simple wrapper to call the existing run_inference function
"""
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional
from .models import OpenPoseData
from .logger import setup_logger

logger = setup_logger(__name__)

def _get_model_function():
    """
    Dynamically import the model function with proper path handling
    This is called only when needed to avoid import errors at module level
    """
    try:
        # Get the absolute path to the model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        model_dir = os.path.join(base_dir, 'model')
        
        # Add the model directory to Python path if not already there
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
            logger.info(f"Added {model_dir} to Python path")
        
        # Now import the function
        from run_model_eval import run_inference_with_data
        logger.info("Successfully imported run_inference_with_data function")
        return run_inference_with_data
        
    except ImportError as e:
        logger.error(f"Failed to import run_inference_with_data: {e}")
        return None

def convert_openpose_to_numpy(keypoints_sequence: List[OpenPoseData]) -> Optional[np.ndarray]:
    """
    Convert OpenPose sequence to numpy array format expected by your model
    
    Args:
        keypoints_sequence: List of OpenPose frames
        
    Returns:
        Numpy array in format (Time, Vertices, Channels) or None if conversion fails
    """
    try:
        frame_data = []
        
        for frame in keypoints_sequence:
            if frame.people:
                person = frame.people[0]  # Use first person
                
                # Extract keypoints in the exact same order as your model expects
                pose_data = []
                for key in ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'face_keypoints_2d']:
                    keypoints = getattr(person, key, [])
                    if keypoints:
                        pose_data.extend(keypoints)
                
                # Reshape to (vertices, channels) - should give (137, 3)
                if pose_data:
                    frame_array = np.array(pose_data).reshape(-1, 3)
                    frame_data.append(frame_array)
        
        if not frame_data:
            logger.warning("No valid keypoints found in sequence")
            return None
            
        # Stack frames: (Time, Vertices, Channels)
        joints = np.stack(frame_data)
        logger.info(f"Converted OpenPose sequence to shape: {joints.shape}")
        
        return joints
        
    except Exception as e:
        logger.error(f"Error converting OpenPose to numpy: {e}")
        return None

def run_model_inference(keypoints_sequence: List[OpenPoseData]) -> Dict[str, Any]:
    """
    Run inference using the existing model by passing keypoints data directly
    
    Args:
        keypoints_sequence: List of OpenPose frames
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Get the model function dynamically
        run_inference_with_data = _get_model_function()
        if run_inference_with_data is None:
            return {
                "success": False,
                "error": "Model function not available - import failed",
                "text": "",
                "frames_processed": 0
            }
        
        # Convert OpenPose to numpy format
        joints = convert_openpose_to_numpy(keypoints_sequence)
        if joints is None:
            return {
                "success": False,
                "error": "Failed to convert keypoints to numpy format",
                "text": "",
                "frames_processed": 0
            }
        
        logger.info(f"Running model inference with joints shape: {joints.shape}")
        
        # Call your model with the keypoints data
        result = run_inference_with_data(joints)
        
        return {
            "success": True,
            "text": result[0] if result else "",
            "confidence": 0.95,  # You can modify your model to return confidence
            "frames_processed": len(keypoints_sequence)
        }
        
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "frames_processed": 0
        }
