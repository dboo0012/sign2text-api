"""
Model utilities for sign language recognition
Simple wrapper to call the existing run_inference function
"""
import torch
import numpy as np
import os
import json
import sys
from typing import List, Dict, Any, Optional
from .models import OpenPoseData
from .logger import setup_logger
from run_model_eval import run_inference_with_data

logger = setup_logger(__name__)

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
        # Convert OpenPose to numpy format
        joints = convert_openpose_to_numpy(keypoints_sequence)
        if joints is None:
            return {
                "success": False,
                "error": "Failed to convert keypoints to numpy format",
                "text": "",
                "frames_processed": 0
            }
        
        # Add model directory to Python path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'model')
        
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
            logger.info(f"Added {model_dir} to Python path")
        
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
