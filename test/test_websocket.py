"""
Pytest-based tests for simplified WebSocket functionality
"""
import sys
import os
import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path so we can import the app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.websocket_handler import WebSocketHandler


class TestWebSocketHandler:
    """Test simplified WebSocketHandler functionality"""
    
    @pytest.fixture
    def handler(self):
        """Create a WebSocketHandler instance for testing"""
        return WebSocketHandler()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    def test_handler_initialization(self, handler):
        """Test WebSocketHandler initialization"""
        assert handler.current_websocket is None, "Should start with no connection"
        assert handler.connected_at is None, "Should start with no connection time"
        assert handler.messages_processed == 0, "Should start with 0 messages processed"
        assert handler.keypoints_processor is None, "Should start with no keypoints processor"
    
    def test_is_connected_false_initially(self, handler):
        """Test is_connected returns False initially"""
        assert not handler.is_connected(), "Should not be connected initially"
    
    def test_get_connection_stats_empty(self, handler):
        """Test getting connection stats when no connection"""
        stats = handler.get_connection_stats()
        
        assert stats['connected'] is False, "Should show not connected"
        assert stats['connected_at'] is None, "Should have no connection time"
        assert stats['messages_processed'] == 0, "Should have 0 messages processed"
        assert stats['uptime_seconds'] == 0, "Should have 0 uptime"
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, handler):
        """Test handling ping message"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        
        timestamp = 1234567890.0
        ping_message = {
            "type": "ping",
            "timestamp": timestamp
        }
        
        await handler._process_message(ping_message)
        
        # Verify pong response was sent using send_json
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "pong", "Should respond with pong"
        assert call_args["timestamp"] == timestamp, "Should echo timestamp"
    
    @pytest.mark.asyncio
    async def test_handle_keypoints_message_no_data(self, handler):
        """Test handling keypoints message without keypoints data"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        
        keypoints_message = {
            "type": "keypoint_sequence",  # Updated to new message type
            "timestamp": 1234567890.0
            # Missing 'keypoints' field
        }
        
        await handler._process_message(keypoints_message)
        
        # Should send error response for validation error
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "error", "Should respond with error"
        assert "Invalid message format" in call_args["message"], "Should indicate validation error"
    
    @pytest.mark.asyncio
    async def test_handle_keypoints_message_success(self, handler):
        """Test successful keypoints processing"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        
        keypoints_message = {
            "type": "keypoint_sequence",  # Updated to new message type
            "timestamp": 1234567890.0,
            "keypoints": {
                "pose": [[0.5, 0.5, 0.0, 0.9]],
                "face": [[0.5, 0.5, 0.0]],
                "left_hand": [[0.3, 0.5, 0.0]],
                "right_hand": [[0.7, 0.5, 0.0]]
            }
        }
        
        await handler._process_message(keypoints_message)
        
        # Should send success response
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "processing_response", "Should respond with processing_response"
        assert call_args["success"] == True, "Should indicate success"
    
    @pytest.mark.asyncio
    async def test_handle_keypoints_message_empty_data(self, handler):
        """Test handling keypoints message with empty keypoints"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        
        keypoints_message = {
            "type": "keypoint_sequence",  # Updated to new message type
            "timestamp": 1234567890.0,
            "keypoints": {
                "pose": [],
                "face": [],
                "left_hand": [],
                "right_hand": []
            }
        }
        
        await handler._process_message(keypoints_message)
        
        # Should send response indicating no data
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "processing_response", "Should respond with processing_response"
        assert call_args["success"] == False, "Should indicate no data processed"
    
    @pytest.mark.asyncio
    async def test_send_error(self, handler):
        """Test sending error message"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        
        error_message = "Test error message"
        
        await handler._send_error(error_message)
        
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "error", "Should be error type"
        assert call_args["message"] == error_message, "Should contain error message"
    
    @pytest.mark.asyncio
    async def test_send_response_no_connection(self, handler):
        """Test sending response when no connection is active"""
        handler.current_websocket = None
        
        # Should not raise an error
        from app.models import ErrorMessage
        test_msg = ErrorMessage(message="test")
        await handler._send_json_response(test_msg)
    
    @pytest.mark.asyncio
    async def test_process_invalid_message(self, handler):
        """Test processing invalid message structure"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        handler.messages_processed = 0
        
        # Invalid message structure (missing required fields)
        invalid_message = {"invalid": "structure"}
        
        await handler._process_message(invalid_message)
        
        # Should send error response
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "error", "Should respond with error"
        assert "Unknown message type" in call_args["message"], "Should indicate unknown type"
        assert handler.messages_processed == 1, "Should increment message counter even on error"
    
    @pytest.mark.asyncio
    async def test_process_unknown_message_type(self, handler):
        """Test processing unknown message type"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        handler.messages_processed = 0
        
        unknown_message = {
            "type": "unknown_type",
            "data": "some data"
        }
        
        await handler._process_message(unknown_message)
        
        # Should send error response
        handler.current_websocket.send_json.assert_called_once()
        call_args = handler.current_websocket.send_json.call_args[0][0]
        
        assert call_args["type"] == "error", "Should respond with error"
        assert "Unknown message type" in call_args["message"], "Should indicate unknown type"
        assert handler.messages_processed == 1, "Should increment message counter"
    
    @pytest.mark.asyncio
    async def test_reject_second_connection(self, handler, mock_websocket):
        """Test that second connection is rejected"""
        # Set up a connection manually to simulate an active connection
        handler.current_websocket = Mock()
        handler.connected_at = time.time()
        
        second_websocket = Mock()
        second_websocket.close = AsyncMock()
        
        # Try second connection - should be rejected
        await handler.handle_connection(second_websocket)
        
        # Second connection should be rejected
        second_websocket.close.assert_called_once_with(
            code=1008, 
            reason="Only one connection allowed at a time"
        )
    
    def test_cleanup_connection(self, handler):
        """Test connection cleanup"""
        # Set up a mock connection
        handler.current_websocket = Mock()
        handler.connected_at = time.time()
        handler.messages_processed = 5
        
        # Cleanup
        handler._cleanup_connection()
        
        # Verify cleanup
        assert handler.current_websocket is None, "Should clear websocket"
        assert handler.connected_at is None, "Should clear connection time"
        assert handler.messages_processed == 0, "Should reset message counter"
        assert handler.keypoints_processor is None, "Should clear keypoints processor"
    
    @pytest.mark.asyncio
    async def test_get_connection_stats_with_connection(self, handler):
        """Test getting connection stats with active connection"""
        start_time = time.time()
        handler.current_websocket = Mock()
        handler.connected_at = start_time
        handler.messages_processed = 10
        
        # Add small delay to test uptime
        await asyncio.sleep(0.01)
        
        stats = handler.get_connection_stats()
        
        assert stats['connected'] is True, "Should show connected"
        assert stats['connected_at'] == start_time, "Should show correct connection time"
        assert stats['messages_processed'] == 10, "Should show correct message count"
        assert stats['uptime_seconds'] > 0, "Should show positive uptime"


class TestIntegration:
    """Integration tests for simplified WebSocket components"""
    
    @pytest.fixture
    def handler(self):
        """Create a WebSocketHandler instance for testing"""
        return WebSocketHandler()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_message_processing_cycle(self, handler):
        """Test complete message processing cycle"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_json = AsyncMock()
        
        # Test ping - now pass dict directly
        ping_data = {"type": "ping", "timestamp": 12345}
        await handler._process_message(ping_data)
        
        # Test keypoints - now pass dict directly  
        keypoints_data = {
            "type": "keypoint_sequence",  # Updated to new message type
            "keypoints": {
                "pose": [[0.5, 0.5, 0.0, 0.9]],
                "face": [],
                "left_hand": [],
                "right_hand": []
            },
            "timestamp": 67890
        }
        await handler._process_message(keypoints_data)
        
        # Verify both messages were processed
        assert handler.messages_processed == 2, "Should have processed 2 messages"
        assert handler.current_websocket.send_json.call_count == 2, "Should have sent 2 responses"
    
    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_single_connection_enforcement(self, handler):
        """Test that only one connection is allowed at a time"""
        # This test verifies the core requirement of single connection
        # Set up first connection manually 
        handler.current_websocket = Mock()
        handler.connected_at = time.time()
        handler.messages_processed = 0
        
        # Verify first connection is active
        assert handler.is_connected(), "First connection should be active"
        
        # Try to establish second connection
        second_ws = Mock()
        second_ws.close = AsyncMock()
        
        await handler.handle_connection(second_ws)
        
        # Second connection should be rejected
        second_ws.close.assert_called_once_with(
            code=1008, 
            reason="Only one connection allowed at a time"
        )
        
        # First connection should still be active
        assert handler.is_connected(), "First connection should still be active"


class TestKeypointsProcessor:
    """Test keypoints processor functionality"""
    
    @pytest.mark.asyncio
    async def test_keypoints_processor_integration(self):
        """Test that the keypoints processor is properly integrated"""
        from app.keypoints_processor import KeypointsProcessor
        from app.models import Keypoints, FrameInfo
        
        processor = KeypointsProcessor()
        
        # Test with sample keypoints
        keypoints = Keypoints(
            pose=[[0.5, 0.5, 0.0, 0.9]],
            face=[[0.5, 0.4, 0.0]],
            left_hand=[[0.3, 0.6, 0.0]],
            right_hand=[[0.7, 0.6, 0.0]]
        )
        
        frame_info = FrameInfo(
            width=640,
            height=480,
            has_pose=True,
            has_face=True,
            has_left_hand=True,
            has_right_hand=True
        )
        
        result = await processor.process_keypoints(keypoints, frame_info, 1234567890.0)
        
        assert result.success is True, "Should successfully process keypoints"
        assert result.analysis_result is not None, "Should return analysis results"
        assert "processing_info" in result.analysis_result, "Should contain processing info"