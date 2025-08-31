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
    
    # Processing Configuration (for client-side keypoints)
    KEYPOINTS_CONFIG = {
        "max_pose_landmarks": 33,
        "max_face_landmarks": 468,
        "max_hand_landmarks": 21,
        "required_confidence": 0.5
    }
    
    # Model Configuration
    MODEL_CONFIG = {
        "model_dir": os.getenv("MODEL_DIR", "model"),
        "weights_path": os.getenv("MODEL_WEIGHTS_PATH", "model/how2sign/vn_model/glofe_vn_how2sign_0224.pt"),
        "tokenizer_path": os.getenv("MODEL_TOKENIZER_PATH", "model/notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased"),
        "config_path": os.getenv("MODEL_CONFIG_PATH", "model/how2sign/vn_model/exp_config.json"),
        "clip_length": int(os.getenv("MODEL_CLIP_LENGTH", "16")),
        "max_gen_tks": int(os.getenv("MODEL_MAX_GEN_TKS", "35")),
        "num_beams": int(os.getenv("MODEL_NUM_BEAMS", "5")),
        "buffer_size": int(os.getenv("MODEL_BUFFER_SIZE", "32")),
        "min_sequence_length": int(os.getenv("MODEL_MIN_SEQUENCE_LENGTH", "16"))
    }
    
    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL = 20  # seconds
    WEBSOCKET_PING_TIMEOUT = 20   # seconds
    
    # Logging Configuration
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL_STR = "INFO"

# Global settings instance
settings = Settings() 