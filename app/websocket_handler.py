"""
Simplified WebSocket handler for single connection processing using FastAPI built-in features
"""
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from .logger import setup_logger
from .models import KeypointsInputMessage, PingMessage, FrameMessage, ProcessingResponseMessage, PongMessage, ErrorMessage, ProcessingResult, StructuredKeypoints
from .keypoints_processor import KeypointsProcessor
from .openpose import OpenPoseExtractor
import time
import json
import numpy as np
import cv2
from typing import Optional, Union
from pydantic import ValidationError

logger = setup_logger(__name__)

class WebSocketHandler:
    """Simplified WebSocket handler that manages a single connection at a time"""
    
    def __init__(self):
        """Initialize the handler with no active connection"""
        self.current_websocket: Optional[WebSocket] = None
        self.keypoints_processor: Optional[KeypointsProcessor] = None
        self.openpose_extractor: Optional[OpenPoseExtractor] = None
        self.connected_at: Optional[float] = None
        self.messages_processed: int = 0
        self.frames_processed: int = 0
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
            # Accept the connection first
            await websocket.accept()
            self.current_websocket = websocket
            self.connected_at = time.time()
            self.messages_processed = 0
            self.frames_processed = 0
            
            logger.info("WebSocket connection established")
            
            # Initialize processors (non-blocking)
            try:
                self.keypoints_processor = KeypointsProcessor()
                logger.info("✓ Keypoints processor initialized")
            except Exception as e:
                logger.warning(f"⚠️ Keypoints processor initialization failed: {e}")
                self.keypoints_processor = None
            
            # Initialize OpenPose (can be slow, but shouldn't block connection)
            try:
                self.openpose_extractor = OpenPoseExtractor()
                if self.openpose_extractor.is_initialized:
                    logger.info("✓ OpenPose extractor ready for keypoint extraction")
                else:
                    logger.warning("⚠️ OpenPose extractor not available - will accept frames but not extract keypoints")
            except Exception as e:
                logger.warning(f"⚠️ OpenPose extractor initialization failed: {e}")
                self.openpose_extractor = None
            
            # Handle messages until disconnection
            while True:
                # Use FastAPI's built-in JSON receiving
                message_data = await websocket.receive_json()
                await self._process_message(message_data)
                
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed by client")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self._cleanup_connection()

    def _cleanup_connection(self):
        """Clean up the current connection and associated resources"""
        if self.current_websocket:
            logger.info(f"Connection cleanup - processed {self.messages_processed} messages, {self.frames_processed} frames")
        
        self.current_websocket = None
        self.keypoints_processor = None
        self.openpose_extractor = None
        self.connected_at = None
        self.messages_processed = 0
        self.frames_processed = 0

    async def _process_message(self, message_data: dict):
        """
        Process incoming WebSocket message using FastAPI's built-in JSON parsing
        
        Args:
            message_data: Already parsed message dictionary from websocket.receive_json()
        """
        self.messages_processed += 1
        
        try:
            msg_type = message_data.get('type')
            
            if msg_type == 'keypoint_sequence':
                # Validate using Pydantic model
                keypoints_msg = KeypointsInputMessage(**message_data)
                await self._handle_keypoints_input(keypoints_msg)
            elif msg_type == 'ping':
                # Validate using Pydantic model
                ping_msg = PingMessage(**message_data)
                await self._handle_ping(ping_msg)
            elif msg_type == 'frame':
                # Validate using Pydantic model
                frame_msg = FrameMessage(**message_data)
                await self._handle_frame_input(frame_msg)
            else:
                await self._send_error(f"Unknown message type: {msg_type}")
                
        except ValidationError as e:
            logger.error(f"Message validation error: {e}")
            await self._send_error(f"Invalid message format: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error(f"Message processing error: {str(e)}")

    async def _handle_keypoints_input(self, message: KeypointsInputMessage):
        """
        Handle keypoints input message using validated Pydantic model with new packet structure
        
        Args:
            message: Validated KeypointsInputMessage model with sequence_id and OpenPose data
        """
        if not self.keypoints_processor:
            await self._send_error("Keypoints processor not initialized")
            return
            
        try:
            logger.info(f"Processing keypoint sequence: {message.sequence_id}")
            
            # Process keypoints using the dedicated processor
            result = await self.keypoints_processor.process_keypoints(
                keypoint_data=message.keypoints,  # This is now the OpenPoseData structure
                frame_info=message.frame_info,
                timestamp=message.timestamp
            )
            
            # Create and send response using Pydantic model
            response_data = result.analysis_result if result.success else None
            if response_data:
                # Add sequence_id to the response for correlation
                response_data["sequence_id"] = message.sequence_id
                response_data["format"] = message.format
            
            response = ProcessingResponseMessage(
                timestamp=message.timestamp,
                success=result.success,
                message=result.error if not result.success else f"Sequence {message.sequence_id} processed successfully",
                prediction=response_data.get('prediction') if response_data else None,
                processed_data=response_data
            )
            await self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Error processing keypoints for sequence {message.sequence_id}: {e}")
            await self._send_error(f"Keypoints processing error for {message.sequence_id}: {str(e)}")

    async def _handle_frame_input(self, message: FrameMessage):
        """
        Handle frame input message using validated Pydantic model
        Extract keypoints using OpenPose and process them
        
        Args:
            message: Validated FrameMessage model containing frame data
        """
        try:            
            # Convert the list of integers back to bytes
            frame_bytes = bytes(message.data)
            
            # Decode the frame using OpenCV
            np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning("⚠️ Failed to decode frame")
                await self._send_error("Failed to decode frame image")
                return
            
            height, width = frame.shape[:2]
            self.frames_processed += 1
            
            # Save frame for debugging (overwrites latest.jpg each time)
            # cv2.imwrite("latest.jpg", frame)
            
            # Check if OpenPose is available
            if not self.openpose_extractor or not self.openpose_extractor.is_initialized:
                logger.debug("⚠️ OpenPose extractor not available - skipping keypoint extraction")
                response = ProcessingResponseMessage(
                    timestamp=message.timestamp,
                    success=True,
                    message=f"Frame received (OpenPose not available): {width}x{height}",
                    prediction=None,
                    processed_data={
                        "frame_width": width,
                        "frame_height": height,
                        "format": message.format,
                        "openpose_available": False,
                        "frame_number": self.frames_processed
                    }
                )
                await self._send_json_response(response)
                return
            
            # Extract keypoints from the frame
            extracted_keypoints = self.openpose_extractor.extract_keypoints(frame)
            
            if extracted_keypoints is None:
                logger.warning("⚠️ Failed to extract keypoints from frame")
                response = ProcessingResponseMessage(
                    timestamp=message.timestamp,
                    success=True,
                    message=f"Frame received but no keypoints detected: {width}x{height}",
                    prediction=None,
                    processed_data={
                        "frame_width": width,
                        "frame_height": height,
                        "format": message.format,
                        "people_detected": 0,
                        "keypoints_extracted": False
                    }
                )
                await self._send_json_response(response)
                return
            
            # Process extracted keypoints
            num_people = len(extracted_keypoints.people)
            logger.info(f"✓ Extracted keypoints for {num_people} person(s)")
            
            # if num_people > 0:
            person = extracted_keypoints.people[0]
            
            # Send raw keypoints to the model for inference via keypoints processor
            # The processor will buffer frames and run inference when enough frames are collected
            if self.keypoints_processor:
                # logger.info(f"Passing keypoint into Keypoint processor == {extracted_keypoints.model_dump()}")

                result = await self.keypoints_processor.process_keypoints(
                    keypoint_data=extracted_keypoints.model_dump(),  # Pass raw OpenPoseData
                    frame_info=None,
                    timestamp=message.timestamp
                )

                # Create and send response using Pydantic model
                response_data = result.analysis_result if result.success else None
                # logger.info(f"Returned result: {result}")
                # logger.info(f"SENDING RESULTS (response_data): {response_data}")
                
                if response_data:
                    # Add sequence_id for correlation (using frames_processed as sequence)
                    response_data["sequence_id"] = self.frames_processed
                    response_data["format"] = message.format
                
                response = ProcessingResponseMessage(
                    timestamp=message.timestamp,
                    success=result.success,  # Overall processing success
                    message=result.error if not result.success else f"Sequence {self.frames_processed} processed",
                    prediction=response_data.get('prediction') if response_data else None,
                    processed_data=response_data
                )
                # logger.info(f"Response to ws: {response}")
                await self._send_json_response(response)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            await self._send_error(f"Frame processing error: {str(e)}")

    async def _handle_ping(self, message: PingMessage):
        """
        Handle ping message using validated Pydantic model
        
        Args:
            message: Validated PingMessage model
        """
        response = PongMessage(timestamp=message.timestamp)
        await self._send_json_response(response)

    async def _send_error(self, error_message: str):
        """
        Send error message to client using Pydantic model
        
        Args:
            error_message: Error message to send
        """
        error_response = ErrorMessage(message=error_message)
        await self._send_json_response(error_response)

    async def _send_json_response(self, response: Union[ProcessingResponseMessage, PongMessage, ErrorMessage]):
        """
        Send response to the current WebSocket connection using FastAPI's send_json
        
        Args:
            response: Pydantic model response to send
        """
        if self.current_websocket:
            try:
                # Use FastAPI's built-in JSON sending with Pydantic model
                await self.current_websocket.send_json(response.model_dump())
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
        stats = {
            'connected': self.current_websocket is not None,
            'connected_at': self.connected_at,
            'messages_processed': self.messages_processed,
            'frames_processed': self.frames_processed,
            'uptime_seconds': time.time() - self.connected_at if self.connected_at else 0
        }
        
        # Add keypoints processor stats if available
        if self.keypoints_processor:
            processor_stats = self.keypoints_processor.get_processing_stats()
            stats['keypoints_processing'] = processor_stats
        
        # Add OpenPose extractor stats if available
        if self.openpose_extractor:
            openpose_stats = self.openpose_extractor.get_stats()
            stats['openpose_extractor'] = openpose_stats
            
        return stats