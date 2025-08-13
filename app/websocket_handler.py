"""
Simplified WebSocket handler for single connection processing
"""
from fastapi import WebSocket, WebSocketDisconnect
from .logger import setup_logger
from .pose_processor import PoseProcessor
import json
import time
from typing import Optional

logger = setup_logger(__name__)

class WebSocketHandler:
    """Simplified WebSocket handler that manages a single connection at a time"""
    
    def __init__(self):
        """Initialize the handler with no active connection"""
        self.current_websocket: Optional[WebSocket] = None
        self.processor: Optional[PoseProcessor] = None
        self.connected_at: Optional[float] = None
        self.messages_processed: int = 0
        logger.info("WebSocket handler initialized for single connection")

    async def handle_connection(self, websocket: WebSocket):
        """
        Handle a single WebSocket connection
        
        Args:
            websocket: WebSocket connection object
        """
        # If there's already an active connection, reject the new one
        if self.current_websocket is not None:
            logger.warning("Connection denied, websocket connection already exists.")
            await websocket.close(code=1008, reason="Only one connection allowed at a time")
            return

        try:
            # Accept the connection
            await websocket.accept()
            self.current_websocket = websocket
            self.processor = PoseProcessor()
            self.connected_at = time.time()
            self.messages_processed = 0
            
            logger.info("WebSocket connection established")
            
            # Handle messages until disconnection
            while True:
                data = await websocket.receive_text()
                await self._process_message(data)
                
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed by client")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self._cleanup_connection()

    def _cleanup_connection(self):
        """Clean up the current connection and associated resources"""
        if self.current_websocket:
            logger.info(f"Connection cleanup - processed {self.messages_processed} messages")
        
        self.current_websocket = None
        self.processor = None
        self.connected_at = None
        self.messages_processed = 0

    async def _process_message(self, data: str):
        """
        Process incoming WebSocket message
        
        Args:
            data: Raw message data as string
        """
        self.messages_processed += 1
        
        try:
            message = json.loads(data)
            msg_type = message.get('type')
            
            if msg_type == 'frame':
                await self._handle_frame(message)
            elif msg_type == 'ping':
                await self._handle_ping(message)
            else:
                await self._send_error(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            await self._send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error(f"Message processing error: {str(e)}")

    async def _handle_frame(self, message):
        """
        Handle frame processing message
        
        Args:
            message: Parsed message dictionary
        """
        frame_data = message.get('frame')
        timestamp = message.get('timestamp')
        
        if not frame_data:
            await self._send_error("No frame data provided")
            return
            
        if not self.processor:
            await self._send_error("Pose processor not initialized")
            return
            
        # Decode and process the frame
        frame = self.processor.decode_base64_frame(frame_data)
        if frame is None:
            await self._send_error("Failed to decode image")
            return
            
        result = self.processor.process_frame(frame)
        
        # Send the result back to the client
        await self._send_response({
            'type': 'keypoints',
            'timestamp': timestamp,
            'data': result.model_dump()
        })

    async def _handle_ping(self, message):
        """
        Handle ping message
        
        Args:
            message: Parsed message dictionary
        """
        await self._send_response({
            'type': 'pong',
            'timestamp': message.get('timestamp')
        })

    async def _send_error(self, error_message: str):
        """
        Send error message to client
        
        Args:
            error_message: Error message to send
        """
        await self._send_response({
            'type': 'error',
            'message': error_message
        })

    async def _send_response(self, response: dict):
        """
        Send response to the current WebSocket connection
        
        Args:
            response: Response dictionary to send
        """
        if self.current_websocket:
            try:
                await self.current_websocket.send_text(json.dumps(response))
            except Exception as e:
                logger.error(f"Failed to send response: {e}")

    def is_connected(self) -> bool:
        """
        Check if there's an active connection
        
        Returns:
            True if there's an active connection
        """
        return self.current_websocket is not None

    def get_connection_stats(self) -> dict:
        """
        Get statistics about the current connection
        
        Returns:
            Dictionary with connection statistics
        """
        if self.current_websocket:
            return {
                'connected': True,
                'connected_at': self.connected_at,
                'messages_processed': self.messages_processed,
                'uptime_seconds': time.time() - self.connected_at if self.connected_at else 0
            }
        else:
            return {
                'connected': False,
                'connected_at': None,
                'messages_processed': 0,
                'uptime_seconds': 0
            }