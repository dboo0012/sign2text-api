"""
MediaPipe processor - just logs and echoes back keypoints
"""
from typing import Dict, Any
from .logger import setup_logger

logger = setup_logger(__name__)

class MediaPipeProcessor:
    """
    Simple processor that just logs MediaPipe keypoints and echoes them back
    """
    
    def __init__(self):
        """Initialize the processor"""
        self.processed_count = 0
        logger.info("MediaPipeProcessor initialized")
    
    def process_keypoints(self, sequence_id: str, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process MediaPipe keypoints - just log and return them
        
        Args:
            sequence_id: Frame sequence identifier
            keypoints: Raw MediaPipe keypoints data
            
        Returns:
            Dictionary with the received keypoints
        """
        self.processed_count += 1
        
        # Log what we received
        logger.info(f"Received MediaPipe keypoints for sequence: {sequence_id}")
        logger.info(f"Keypoints keys: {list(keypoints.keys())}")
        
        # Log counts if the data has arrays
        for key, value in keypoints.items():
            if isinstance(value, list):
                logger.info(f"  {key}: {len(value)} items")
            elif value is not None:
                logger.info(f"  {key}: {type(value).__name__}")
        
        logger.info(f"Total sequences processed: {self.processed_count}")
        
        # Just return what we received
        return {
            "sequence_id": sequence_id,
            "processed_count": self.processed_count,
            "received_keypoints": keypoints
        }
