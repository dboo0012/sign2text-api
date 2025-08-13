"""
Configuration settings for the Sign Language Recognition API
"""
import os
from typing import List

class Settings:
    """Application settings and configuration"""
    
    # API Configuration
    API_TITLE = "sign2text API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Real-time sign language recognition using MediaPipe and WebSocket"
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    RELOAD = os.getenv("RELOAD", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*"  # In production, remove this and specify exact origins
    ]
    
    # MediaPipe Configuration
    MEDIAPIPE_CONFIG = {
        "static_image_mode": False,
        "model_complexity": 1,
        "enable_segmentation": False,
        "refine_face_landmarks": True,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5
    }
    
    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL = 20  # seconds
    WEBSOCKET_PING_TIMEOUT = 20   # seconds
    
    # Logging Configuration
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL_STR = "INFO"

# Global settings instance
settings = Settings() 