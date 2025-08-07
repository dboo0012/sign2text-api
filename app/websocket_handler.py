"""
WebSocket message handler for processing client requests
"""
from fastapi import WebSocket, WebSocketDisconnect
from .logger import setup_logger
from .connection_manager import ConnectionManager
import json

logger = setup_logger(__name__)

class WebSocketHandler:
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager

    async def handle_connection(self, websocket: WebSocket):
        await self.connection_manager.connect(websocket)
        processor = self.connection_manager.get_processor(websocket)
        try:
            while self.connection_manager.is_connected(websocket):
                data = await websocket.receive_text()
                await self._process_message(websocket, data, processor)
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed by client")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connection_manager.disconnect(websocket)

    async def _process_message(self, websocket: WebSocket, data: str, processor):
        try:
            message = json.loads(data)
            msg_type = message.get('type')
            self.connection_manager.update_activity(websocket)
            if msg_type == 'frame':
                await self._handle_frame(websocket, message, processor)
            elif msg_type == 'ping':
                await self._handle_ping(websocket, message)
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error(websocket, f"Message processing error: {str(e)}")

    async def _handle_frame(self, websocket: WebSocket, message, processor):
        frame_data = message.get('frame')
        timestamp = message.get('timestamp')
        if not frame_data:
            await self._send_error(websocket, "No frame data provided")
            return
        frame = processor.decode_base64_frame(frame_data)
        if frame is None:
            await self._send_error(websocket, "Failed to decode image")
            return
        result = processor.process_frame(frame)
        await websocket.send_text(json.dumps({
            'type': 'keypoints',
            'timestamp': timestamp,
            'data': result.model_dump()
        }))

    async def _handle_ping(self, websocket: WebSocket, message):
        await websocket.send_text(json.dumps({
            'type': 'pong',
            'timestamp': message.get('timestamp')
        }))

    async def _send_error(self, websocket: WebSocket, message: str):
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': message
        }))