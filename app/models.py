"""
Data models for the Sign Language Recognition API
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime

class FrameInfo(BaseModel):
    """Information about a processed frame"""
    width: int
    height: int
    has_pose: bool
    has_face: bool
    has_left_hand: bool
    has_right_hand: bool

# Keypoint data types based on OpenPose/MediaPipe format
class Point2D(BaseModel):
    """2D coordinate point"""
    x: float
    y: float

class Point3D(BaseModel):
    """3D coordinate point"""
    x: float
    y: float
    z: float = 0.0

class KeypointWithConfidence(BaseModel):
    """Keypoint with confidence/visibility score"""
    x: float
    y: float
    confidence: float
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'KeypointWithConfidence':
        """Create from flat list [x, y, confidence]"""
        if len(coords) >= 3:
            return cls(x=coords[0], y=coords[1], confidence=coords[2])
        elif len(coords) == 2:
            return cls(x=coords[0], y=coords[1], confidence=1.0)
        else:
            raise ValueError(f"Invalid keypoint format: {coords}")

class PoseKeypoint(BaseModel):
    """Pose keypoint with visibility"""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'PoseKeypoint':
        """Create from flat list [x, y, z, visibility] or [x, y, visibility]"""
        if len(coords) >= 4:
            return cls(x=coords[0], y=coords[1], z=coords[2], visibility=coords[3])
        elif len(coords) == 3:
            return cls(x=coords[0], y=coords[1], visibility=coords[2])
        else:
            raise ValueError(f"Invalid pose keypoint format: {coords}")

# OpenPose person data structure
class OpenPosePerson(BaseModel):
    """Single person detection from OpenPose"""
    person_id: List[int] = Field(default_factory=list)
    pose_keypoints_2d: List[float] = Field(default_factory=list)
    face_keypoints_2d: List[float] = Field(default_factory=list)
    hand_left_keypoints_2d: List[float] = Field(default_factory=list)
    hand_right_keypoints_2d: List[float] = Field(default_factory=list)
    pose_keypoints_3d: List[float] = Field(default_factory=list)
    face_keypoints_3d: List[float] = Field(default_factory=list)
    hand_left_keypoints_3d: List[float] = Field(default_factory=list)
    hand_right_keypoints_3d: List[float] = Field(default_factory=list)

class OpenPoseData(BaseModel):
    """Complete OpenPose JSON structure"""
    version: float
    people: List[OpenPosePerson] = Field(default_factory=list)
    
    @validator('people')
    def validate_people(cls, v):
        """Ensure at least one person is detected"""
        return v

# Structured keypoints for processing
class StructuredKeypoints(BaseModel):
    """Structured keypoints extracted from OpenPose data"""
    pose: List[PoseKeypoint] = Field(default_factory=list)
    face: List[KeypointWithConfidence] = Field(default_factory=list)
    left_hand: List[KeypointWithConfidence] = Field(default_factory=list)
    right_hand: List[KeypointWithConfidence] = Field(default_factory=list)
    
    @classmethod
    def from_openpose_person(cls, person: OpenPosePerson) -> 'StructuredKeypoints':
        """Convert OpenPose person data to structured keypoints"""
        # Convert pose keypoints (groups of 3 or 4: x, y, [z], confidence)
        pose_points = []
        pose_data = person.pose_keypoints_2d
        if pose_data:
            # Determine if it's 3D (x,y,z,confidence) or 2D (x,y,confidence)
            stride = 4 if len(pose_data) % 4 == 0 else 3
            for i in range(0, len(pose_data), stride):
                if i + stride <= len(pose_data):
                    coords = pose_data[i:i + stride]
                    # Filter out zero confidence points
                    if coords[-1] > 0.1:  # confidence threshold
                        try:
                            pose_points.append(PoseKeypoint.from_list(coords))
                        except ValueError:
                            continue
        
        # Convert face keypoints (groups of 3: x, y, confidence)
        face_points = []
        face_data = person.face_keypoints_2d
        if face_data:
            for i in range(0, len(face_data), 3):
                if i + 3 <= len(face_data):
                    coords = face_data[i:i + 3]
                    if coords[2] > 0.1:  # confidence threshold
                        try:
                            face_points.append(KeypointWithConfidence.from_list(coords))
                        except ValueError:
                            continue
        
        # Convert hand keypoints (groups of 3: x, y, confidence)
        left_hand_points = []
        left_hand_data = person.hand_left_keypoints_2d
        if left_hand_data:
            for i in range(0, len(left_hand_data), 3):
                if i + 3 <= len(left_hand_data):
                    coords = left_hand_data[i:i + 3]
                    if coords[2] > 0.1:  # confidence threshold
                        try:
                            left_hand_points.append(KeypointWithConfidence.from_list(coords))
                        except ValueError:
                            continue
        
        right_hand_points = []
        right_hand_data = person.hand_right_keypoints_2d
        if right_hand_data:
            for i in range(0, len(right_hand_data), 3):
                if i + 3 <= len(right_hand_data):
                    coords = right_hand_data[i:i + 3]
                    if coords[2] > 0.1:  # confidence threshold
                        try:
                            right_hand_points.append(KeypointWithConfidence.from_list(coords))
                        except ValueError:
                            continue
        
        return cls(
            pose=pose_points,
            face=face_points,
            left_hand=left_hand_points,
            right_hand=right_hand_points
        )
    
    def to_numpy_features(self) -> List[float]:
        """Convert to flat feature vector for model input"""
        features = []
        
        # Add pose features (x, y coordinates only)
        for point in self.pose:
            features.extend([point.x, point.y])
        
        # Add hand features
        for point in self.left_hand:
            features.extend([point.x, point.y])
        for point in self.right_hand:
            features.extend([point.x, point.y])
        
        # Optionally add face features (subset for important facial landmarks)
        # Using only first 10 face points to avoid noise
        for point in self.face[:10]:
            features.extend([point.x, point.y])
        
        return features
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detected keypoints"""
        return {
            "pose_points": len(self.pose),
            "face_points": len(self.face),
            "left_hand_points": len(self.left_hand),
            "right_hand_points": len(self.right_hand),
            "total_points": len(self.pose) + len(self.face) + len(self.left_hand) + len(self.right_hand),
            "has_pose": len(self.pose) > 0,
            "has_face": len(self.face) > 0,
            "has_hands": len(self.left_hand) > 0 or len(self.right_hand) > 0
        }

# Legacy compatibility - keeping original Keypoints for backward compatibility
class Keypoints(BaseModel):
    """Legacy keypoints model - use StructuredKeypoints for new code"""
    pose: List[List[float]] = Field(default_factory=list, description="Pose landmarks [x, y, z, visibility]")
    face: List[List[float]] = Field(default_factory=list, description="Face landmarks [x, y, z]")
    left_hand: List[List[float]] = Field(default_factory=list, description="Left hand landmarks [x, y, z]")
    right_hand: List[List[float]] = Field(default_factory=list, description="Right hand landmarks [x, y, z]")
    
    @classmethod
    def from_structured(cls, structured: StructuredKeypoints) -> 'Keypoints':
        """Convert from StructuredKeypoints to legacy format"""
        pose_legacy = [[p.x, p.y, p.z, p.visibility] for p in structured.pose]
        face_legacy = [[f.x, f.y, f.confidence] for f in structured.face]
        left_hand_legacy = [[h.x, h.y, h.confidence] for h in structured.left_hand]
        right_hand_legacy = [[h.x, h.y, h.confidence] for h in structured.right_hand]
        
        return cls(
            pose=pose_legacy,
            face=face_legacy,
            left_hand=left_hand_legacy,
            right_hand=right_hand_legacy
        )

class ProcessingResult(BaseModel):
    """Result of keypoints processing"""
    success: bool
    processed_keypoints: Optional[Union[Keypoints, StructuredKeypoints]] = Field(default=None)
    analysis_result: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = None

class WebSocketMessage(BaseModel):
    """Base WebSocket message model"""
    type: str
    timestamp: Optional[float] = None

class KeypointsInputMessage(WebSocketMessage):
    """Message containing keypoints data from client with new packet structure"""
    type: str = "keypoint_sequence"
    sequence_id: str = Field(description="Unique identifier for the sequence frame, e.g., 'demo_1724312345678_frame_42'")
    keypoints: OpenPoseData = Field(description="The raw OpenPose data structure")
    format: str = Field(default="openpose_raw", description="Indicates the data format")
    frame_info: Optional[FrameInfo] = Field(default=None, description="Optional frame metadata")
    
    @validator('format')
    def validate_format(cls, v):
        """Ensure format is supported"""
        supported_formats = ["openpose_raw"]
        if v not in supported_formats:
            raise ValueError(f"Unsupported format: {v}. Supported formats: {supported_formats}")
        return v
    
    @validator('keypoints')
    def validate_keypoints(cls, v):
        """Ensure keypoints data is provided"""
        if not v:
            raise ValueError("keypoints data must be provided")
        return v

class PingMessage(WebSocketMessage):
    """Ping message for connection health check"""
    type: str = "ping"

class ProcessingResponseMessage(WebSocketMessage):
    """Message containing processing response/acknowledgment"""
    type: str = "success"
    success: bool
    message: Optional[str] = None
    processed_data: Optional[Dict[str, Any]] = Field(default=None, description="Any processed results")

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