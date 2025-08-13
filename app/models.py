"""
Data models for the Sign Language Recognition API
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class FrameInfo(BaseModel):
    """Information about a processed frame"""
    width: int
    height: int
    has_pose: bool
    has_face: bool
    has_left_hand: bool
    has_right_hand: bool

class Keypoints(BaseModel):
    """Landmark keypoints for different body parts"""
    pose: List[List[float]] = Field(default_factory=list, description="Pose landmarks [x, y, z, visibility]")
    face: List[List[float]] = Field(default_factory=list, description="Face landmarks [x, y, z]")
    left_hand: List[List[float]] = Field(default_factory=list, description="Left hand landmarks [x, y, z]")
    right_hand: List[List[float]] = Field(default_factory=list, description="Right hand landmarks [x, y, z]")

class ProcessingResult(BaseModel):
    """Result of frame processing"""
    success: bool
    keypoints: Keypoints = Field(default_factory=Keypoints)
    frame_info: FrameInfo = Field(default_factory=FrameInfo)
    error: Optional[str] = None

class WebSocketMessage(BaseModel):
    """Base WebSocket message model"""
    type: str
    timestamp: Optional[float] = None

class FrameMessage(WebSocketMessage):
    """Message containing frame data"""
    type: str = "frame"
    frame: str = Field(..., description="Base64 encoded image data")

class PingMessage(WebSocketMessage):
    """Ping message for connection health check"""
    type: str = "ping"

class KeypointsMessage(WebSocketMessage):
    """Message containing processed keypoints"""
    type: str = "keypoints"
    data: ProcessingResult

class PongMessage(WebSocketMessage):
    """Pong response message"""
    type: str = "pong"

class ErrorMessage(WebSocketMessage):
    """Error message"""
    type: str = "error"
    message: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    mediapipe_available: bool
    active_connections: int
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)

class RootResponse(BaseModel):
    """Root endpoint response"""
    message: str
    status: str
    active_connections: int
    version: str 