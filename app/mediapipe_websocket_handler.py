"""
MediaPipe WebSocket handler for single connection processing
"""
from fastapi import WebSocket, WebSocketDisconnect
from .logger import setup_logger
from .mediapipe_models import MediaPipeMessage, MediaPipeResponse, MediaPipeError
from .mediapipe_processor import MediaPipeProcessor
import time
from typing import Optional
from pydantic import ValidationError

logger = setup_logger(__name__)

class MediaPipeWebSocketHandler:
    """MediaPipe WebSocket handler for single connection"""
    
    def __init__(self):
        """Initialize the handler"""
        self.current_websocket: Optional[WebSocket] = None
        self.processor: Optional[MediaPipeProcessor] = None
        self.connected_at: Optional[float] = None
        self.messages_processed: int = 0
        logger.info("MediaPipeWebSocketHandler initialized")

    async def handle_connection(self, websocket: WebSocket):
        """Handle a MediaPipe WebSocket connection"""
        # If there's already an active connection, reject the new one
        if self.current_websocket is not None:
            logger.warning("MediaPipe connection denied, connection already exists.")
            await websocket.close(code=1008, reason="Only one MediaPipe connection allowed")
            return

        try:
            # Accept the connection
            await websocket.accept()
            self.current_websocket = websocket
            self.processor = MediaPipeProcessor()
            self.connected_at = time.time()
            self.messages_processed = 0
            
            logger.info("MediaPipe WebSocket connection established")
            
            # Handle messages until disconnection
            while True:
                message_data = await websocket.receive_json()
                await self._process_message(message_data)
                
        except WebSocketDisconnect:
            logger.info("MediaPipe WebSocket connection closed by client")
        except Exception as e:
            logger.error(f"MediaPipe WebSocket error: {e}")
        finally:
            self._cleanup_connection()

    def _cleanup_connection(self):
        """Clean up the connection"""
        if self.current_websocket:
            logger.info(f"MediaPipe connection cleanup - processed {self.messages_processed} messages")
        
        self.current_websocket = None
        self.processor = None
        self.connected_at = None
        self.messages_processed = 0

    async def _process_message(self, message_data: dict):
        """Process incoming MediaPipe message"""
        self.messages_processed += 1
        
        try:
            msg_type = message_data.get('type')
            
            if msg_type == 'mediapipe_keypoints':
                # Validate and process
                message = MediaPipeMessage(**message_data)
                await self._handle_keypoints(message)
            else:
                await self._send_error(f"Unknown MediaPipe message type: {msg_type}")
                
        except ValidationError as e:
            logger.error(f"MediaPipe message validation error: {e}")
            await self._send_error(f"Invalid message format: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing MediaPipe message: {e}")
            await self._send_error(f"Processing error: {str(e)}")

    async def _handle_keypoints(self, message: MediaPipeMessage):
        """Handle MediaPipe keypoints - just log and echo back"""
        if not self.processor:
            await self._send_error("Processor not initialized")
            return
            
        try:
            # Process (just log and return)
            result = self.processor.process_keypoints(message.sequence_id, message.keypoints)
            
            # Send back the keypoints
            response = MediaPipeResponse(
                success=True,
                sequence_id=message.sequence_id,
                received_keypoints=result["received_keypoints"],
                timestamp=message.timestamp
            )
            await self._send_response(response)
            
        except Exception as e:
            logger.error(f"Error processing keypoints: {e}")
            await self._send_error(f"Keypoints processing error: {str(e)}")

    async def _send_error(self, error_message: str):
        """Send error message"""
        error_response = MediaPipeError(message=error_message, timestamp=time.time())
        await self._send_response(error_response)

    async def _send_response(self, response):
        """Send response to client"""
        if self.current_websocket:
            try:
                await self.current_websocket.send_json(response.model_dump())
            except Exception as e:
                logger.error(f"Failed to send MediaPipe response: {e}")

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.current_websocket is not None

    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        return {
            'connected': self.current_websocket is not None,
            'connected_at': self.connected_at,
            'messages_processed': self.messages_processed,
            'uptime_seconds': time.time() - self.connected_at if self.connected_at else 0,
            'connection_type': 'mediapipe'
        }
