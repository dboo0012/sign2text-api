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
from app.pose_processor import PoseProcessor


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
    
    @pytest.fixture
    def mock_processor_result(self):
        """Create a mock processor result"""
        result = Mock()
        result.model_dump = Mock(return_value={
            "keypoints": [],
            "confidence": 0.95,
            "timestamp": time.time()
        })
        return result
    
    def test_handler_initialization(self, handler):
        """Test WebSocketHandler initialization"""
        assert handler.current_websocket is None, "Should start with no connection"
        assert handler.processor is None, "Should start with no processor"
        assert handler.connected_at is None, "Should start with no connection time"
        assert handler.messages_processed == 0, "Should start with 0 messages processed"
    
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
        handler.current_websocket.send_text = AsyncMock()
        
        timestamp = 1234567890.0
        ping_message = {
            "type": "ping",
            "timestamp": timestamp
        }
        
        await handler._handle_ping(ping_message)
        
        # Verify pong response was sent
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "pong", "Should respond with pong"
        assert response["timestamp"] == timestamp, "Should echo timestamp"
    
    @pytest.mark.asyncio
    async def test_handle_frame_message_no_data(self, handler):
        """Test handling frame message without frame data"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        
        frame_message = {
            "type": "frame",
            "timestamp": 1234567890.0
            # Missing 'frame' field
        }
        
        await handler._handle_frame(frame_message)
        
        # Should send error response
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "No frame data provided" in response["message"], "Should indicate missing frame data"
    
    @pytest.mark.asyncio
    async def test_handle_frame_message_no_processor(self, handler):
        """Test handling frame message when processor is not initialized"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        handler.processor = None  # No processor
        
        frame_message = {
            "type": "frame",
            "frame": "test_data",
            "timestamp": 1234567890.0
        }
        
        await handler._handle_frame(frame_message)
        
        # Should send error response
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Pose processor not initialized" in response["message"], "Should indicate processor error"
    
    @pytest.mark.asyncio
    async def test_handle_frame_message_decode_failure(self, handler, mock_processor_result):
        """Test handling frame message when decoding fails"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        
        # Mock processor
        handler.processor = Mock()
        handler.processor.decode_base64_frame = Mock(return_value=None)  # Decode failure
        
        frame_message = {
            "type": "frame",
            "frame": "invalid_base64_data",
            "timestamp": 1234567890.0
        }
        
        await handler._handle_frame(frame_message)
        
        # Should send error response
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Failed to decode image" in response["message"], "Should indicate decode failure"
    
    @pytest.mark.asyncio
    async def test_handle_frame_message_success(self, handler, mock_processor_result):
        """Test successful frame processing"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        
        # Mock processor
        handler.processor = Mock()
        mock_frame = Mock()
        handler.processor.decode_base64_frame = Mock(return_value=mock_frame)
        handler.processor.process_frame = Mock(return_value=mock_processor_result)
        
        timestamp = 1234567890.0
        frame_message = {
            "type": "frame",
            "frame": "valid_base64_data",
            "timestamp": timestamp
        }
        
        await handler._handle_frame(frame_message)
        
        # Verify frame processing was called
        handler.processor.decode_base64_frame.assert_called_once_with("valid_base64_data")
        handler.processor.process_frame.assert_called_once_with(mock_frame)
        
        # Verify response was sent
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "keypoints", "Should respond with keypoints"
        assert response["timestamp"] == timestamp, "Should echo timestamp"
        assert "data" in response, "Should include processed data"
    
    @pytest.mark.asyncio
    async def test_send_error(self, handler):
        """Test sending error message"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        
        error_message = "Test error message"
        
        await handler._send_error(error_message)
        
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should be error type"
        assert response["message"] == error_message, "Should contain error message"
    
    @pytest.mark.asyncio
    async def test_send_response_no_connection(self, handler):
        """Test sending response when no connection is active"""
        handler.current_websocket = None
        
        # Should not raise an error
        await handler._send_response({"type": "test"})
    
    @pytest.mark.asyncio
    async def test_process_invalid_json(self, handler):
        """Test processing invalid JSON message"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        handler.messages_processed = 0
        
        invalid_json = "{ invalid json }"
        
        await handler._process_message(invalid_json)
        
        # Should send error response
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Invalid JSON format" in response["message"], "Should indicate JSON error"
        assert handler.messages_processed == 1, "Should increment message counter even on error"
    
    @pytest.mark.asyncio
    async def test_process_unknown_message_type(self, handler):
        """Test processing unknown message type"""
        handler.current_websocket = Mock()
        handler.current_websocket.send_text = AsyncMock()
        handler.messages_processed = 0
        
        unknown_message = json.dumps({
            "type": "unknown_type",
            "data": "some data"
        })
        
        await handler._process_message(unknown_message)
        
        # Should send error response
        handler.current_websocket.send_text.assert_called_once()
        call_args = handler.current_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Unknown message type" in response["message"], "Should indicate unknown type"
        assert handler.messages_processed == 1, "Should increment message counter"
    
    @pytest.mark.asyncio
    async def test_reject_second_connection(self, handler, mock_websocket):
        """Test that second connection is rejected"""
        # Set up a connection manually to simulate an active connection
        handler.current_websocket = Mock()
        handler.processor = Mock()
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
        handler.processor = Mock()
        handler.connected_at = time.time()
        handler.messages_processed = 5
        
        # Cleanup
        handler._cleanup_connection()
        
        # Verify cleanup
        assert handler.current_websocket is None, "Should clear websocket"
        assert handler.processor is None, "Should clear processor"
        assert handler.connected_at is None, "Should clear connection time"
        assert handler.messages_processed == 0, "Should reset message counter"
    
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
        handler.current_websocket.send_text = AsyncMock()
        
        # Initialize processor
        handler.processor = Mock()
        handler.processor.decode_base64_frame = Mock(return_value=Mock())
        
        mock_result = Mock()
        mock_result.model_dump = Mock(return_value={"test": "data"})
        handler.processor.process_frame = Mock(return_value=mock_result)
        
        # Test ping
        ping_data = json.dumps({"type": "ping", "timestamp": 12345})
        await handler._process_message(ping_data)
        
        # Test frame
        frame_data = json.dumps({
            "type": "frame", 
            "frame": "base64data", 
            "timestamp": 67890
        })
        await handler._process_message(frame_data)
        
        # Verify both messages were processed
        assert handler.messages_processed == 2, "Should have processed 2 messages"
        assert handler.current_websocket.send_text.call_count == 2, "Should have sent 2 responses"
    
    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_single_connection_enforcement(self, handler):
        """Test that only one connection is allowed at a time"""
        # This test verifies the core requirement of single connection
        # Set up first connection manually 
        handler.current_websocket = Mock()
        handler.processor = Mock()
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