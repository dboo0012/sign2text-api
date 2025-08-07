"""
WebSocket connection manager for handling multiple client connections
"""
from typing import Dict, List, Optional
from fastapi import WebSocket
from .pose_processor import PoseProcessor
from .logger import setup_logger

logger = setup_logger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and their associated pose processors"""
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: List[WebSocket] = []
        self.processors: Dict[WebSocket, PoseProcessor] = {}
        self.connection_info: Dict[WebSocket, Dict] = {}
        logger.info("Connection manager initialized")
    
    async def connect(self, websocket: WebSocket) -> bool:
        """
        Accept new WebSocket connection and initialize processor
        
        Args:
            websocket: WebSocket connection object
            
        Returns:
            True if connection was established successfully
        """
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            
            # Initialize pose processor for this connection
            processor = PoseProcessor()
            self.processors[websocket] = processor
            
            # Store connection metadata
            self.connection_info[websocket] = {
                'connected_at': self._get_timestamp(),
                'messages_processed': 0,
                'last_activity': self._get_timestamp()
            }
            
            logger.info(f"New connection established. Total connections: {len(self.active_connections)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish connection: {e}")
            return False
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove WebSocket connection and cleanup resources
        
        Args:
            websocket: WebSocket connection to remove
        """
        try:
            # Remove from active connections
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            # Cleanup processor
            if websocket in self.processors:
                processor = self.processors[websocket]
                del self.processors[websocket]
                # Processor cleanup is handled by __del__ method
            
            # Remove connection info
            if websocket in self.connection_info:
                del self.connection_info[websocket]
            
            logger.info(f"Connection disconnected. Total connections: {len(self.active_connections)}")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def get_processor(self, websocket: WebSocket) -> Optional[PoseProcessor]:
        """
        Get pose processor for a specific connection
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            PoseProcessor instance or None if not found
        """
        return self.processors.get(websocket)
    
    def update_activity(self, websocket: WebSocket) -> None:
        """
        Update last activity timestamp for a connection
        
        Args:
            websocket: WebSocket connection
        """
        if websocket in self.connection_info:
            self.connection_info[websocket]['last_activity'] = self._get_timestamp()
            self.connection_info[websocket]['messages_processed'] += 1
    
    def get_connection_stats(self) -> Dict:
        """
        Get statistics about active connections
        
        Returns:
            Dictionary with connection statistics
        """
        total_connections = len(self.active_connections)
        total_messages = sum(
            info.get('messages_processed', 0) 
            for info in self.connection_info.values()
        )
        
        return {
            'total_connections': total_connections,
            'total_messages_processed': total_messages,
            'active_processors': len(self.processors)
        }
    
    def is_connected(self, websocket: WebSocket) -> bool:
        """
        Check if a WebSocket is still connected
        
        Args:
            websocket: WebSocket connection to check
            
        Returns:
            True if connection is active
        """
        return websocket in self.active_connections
    
    def get_connection_info(self, websocket: WebSocket) -> Optional[Dict]:
        """
        Get information about a specific connection
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Connection information dictionary or None
        """
        return self.connection_info.get(websocket)
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def cleanup_inactive_connections(self) -> int:
        """
        Clean up inactive connections (for future use)
        
        Returns:
            Number of connections cleaned up
        """
        # This could be implemented to remove stale connections
        # For now, we'll rely on WebSocket disconnect events
        return 0 