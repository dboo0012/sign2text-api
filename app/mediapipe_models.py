"""
MediaPipe data models for the Sign Language Recognition API
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class MediaPipeMessage(BaseModel):
    """MediaPipe WebSocket message"""
    type: str
    sequence_id: str = Field(description="Unique identifier for the frame")
    keypoints: Dict[str, Any] = Field(description="Raw MediaPipe keypoints data")
    timestamp: Optional[float] = None

class MediaPipeResponse(BaseModel):
    """MediaPipe response message"""
    type: str = "mediapipe_response"
    success: bool
    sequence_id: str
    received_keypoints: Dict[str, Any]
    timestamp: Optional[float] = None

class MediaPipeError(BaseModel):
    """MediaPipe error message"""
    type: str = "mediapipe_error"
    message: str
    timestamp: Optional[float] = None
