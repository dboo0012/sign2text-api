"""
Pytest-based tests for WebSocket functionality
"""
import sys
import os
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock

# Add parent directory to path so we can import the app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.connection_manager import ConnectionManager
from app.websocket_handler import WebSocketHandler
from app.pose_processor import PoseProcessor


class TestConnectionManager:
    """Test ConnectionManager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create a ConnectionManager instance for testing"""
        return ConnectionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        return websocket
    
    def test_manager_initialization(self, manager):
        """Test ConnectionManager initialization"""
        assert manager.active_connections == [], "Should start with no connections"
        assert manager.processors == {}, "Should start with no processors"
        assert manager.connection_info == {}, "Should start with no connection info"
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, manager, mock_websocket):
        """Test connecting a WebSocket"""
        result = await manager.connect(mock_websocket)
        
        assert result is True, "Connection should succeed"
        assert mock_websocket in manager.active_connections, "WebSocket should be in active connections"
        assert mock_websocket in manager.processors, "WebSocket should have associated processor"
        assert mock_websocket in manager.connection_info, "WebSocket should have connection info"
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, manager, mock_websocket):
        """Test disconnecting a WebSocket"""
        # First connect
        await manager.connect(mock_websocket)
        assert mock_websocket in manager.active_connections
        
        # Then disconnect
        manager.disconnect(mock_websocket)
        
        assert mock_websocket not in manager.active_connections, "WebSocket should be removed from active connections"
        assert mock_websocket not in manager.processors, "Processor should be cleaned up"
        assert mock_websocket not in manager.connection_info, "Connection info should be cleaned up"
    
    def test_get_processor(self, manager, mock_websocket):
        """Test getting processor for a connection"""
        # Before connection
        processor = manager.get_processor(mock_websocket)
        assert processor is None, "Should return None for non-existent connection"
    
    @pytest.mark.asyncio
    async def test_get_processor_after_connect(self, manager, mock_websocket):
        """Test getting processor after connection"""
        await manager.connect(mock_websocket)
        
        processor = manager.get_processor(mock_websocket)
        assert processor is not None, "Should return processor for connected WebSocket"
        assert isinstance(processor, PoseProcessor), "Should return PoseProcessor instance"
    
    @pytest.mark.asyncio
    async def test_update_activity(self, manager, mock_websocket):
        """Test updating connection activity"""
        import time
        
        await manager.connect(mock_websocket)
        
        initial_info = manager.get_connection_info(mock_websocket)
        initial_messages = initial_info['messages_processed']
        
        # Add small delay to ensure timestamp difference
        time.sleep(0.001)
        manager.update_activity(mock_websocket)
        
        updated_info = manager.get_connection_info(mock_websocket)
        assert updated_info['messages_processed'] == initial_messages + 1, "Message count should increment"
        assert updated_info['last_activity'] >= initial_info['last_activity'], "Last activity should be updated or same"
    
    @pytest.mark.asyncio
    async def test_connection_stats(self, manager, mock_websocket):
        """Test getting connection statistics"""
        # Empty stats
        stats = manager.get_connection_stats()
        assert stats['total_connections'] == 0, "Should start with 0 connections"
        assert stats['total_messages_processed'] == 0, "Should start with 0 messages"
        
        # After connection
        await manager.connect(mock_websocket)
        manager.update_activity(mock_websocket)
        
        stats = manager.get_connection_stats()
        assert stats['total_connections'] == 1, "Should have 1 connection"
        assert stats['total_messages_processed'] == 1, "Should have 1 message processed"
        assert stats['active_processors'] == 1, "Should have 1 active processor"
    
    @pytest.mark.asyncio
    async def test_is_connected(self, manager, mock_websocket):
        """Test checking if WebSocket is connected"""
        assert not manager.is_connected(mock_websocket), "Should not be connected initially"
        
        await manager.connect(mock_websocket)
        assert manager.is_connected(mock_websocket), "Should be connected after connect"
        
        manager.disconnect(mock_websocket)
        assert not manager.is_connected(mock_websocket), "Should not be connected after disconnect"


class TestWebSocketHandler:
    """Test WebSocketHandler functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create a ConnectionManager instance for testing"""
        return ConnectionManager()
    
    @pytest.fixture
    def handler(self, manager):
        """Create a WebSocketHandler instance for testing"""
        return WebSocketHandler(manager)
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        return websocket
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock PoseProcessor for testing"""
        processor = Mock()
        processor.decode_base64_frame = Mock(return_value=None)
        processor.process_frame = Mock()
        return processor
    
    def test_handler_initialization(self, handler, manager):
        """Test WebSocketHandler initialization"""
        assert handler.connection_manager is manager, "Handler should reference the connection manager"
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, handler, mock_websocket):
        """Test handling ping message"""
        timestamp = 1234567890.0
        ping_message = {
            "type": "ping",
            "timestamp": timestamp
        }
        
        await handler._handle_ping(mock_websocket, ping_message)
        
        # Verify pong response was sent
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "pong", "Should respond with pong"
        assert response["timestamp"] == timestamp, "Should echo timestamp"
    
    @pytest.mark.asyncio
    async def test_handle_frame_message_no_data(self, handler, mock_websocket):
        """Test handling frame message without frame data"""
        frame_message = {
            "type": "frame",
            "timestamp": 1234567890.0
            # Missing 'frame' field
        }
        
        await handler._handle_frame(mock_websocket, frame_message, Mock())
        
        # Should send error response
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "No frame data provided" in response["message"], "Should indicate missing frame data"
    
    @pytest.mark.asyncio
    async def test_handle_frame_message_decode_failure(self, handler, mock_websocket, mock_processor):
        """Test handling frame message when decoding fails"""
        frame_message = {
            "type": "frame",
            "frame": "invalid_base64_data",
            "timestamp": 1234567890.0
        }
        
        # Mock processor to return None (decode failure)
        mock_processor.decode_base64_frame.return_value = None
        
        await handler._handle_frame(mock_websocket, frame_message, mock_processor)
        
        # Should send error response
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Failed to decode image" in response["message"], "Should indicate decode failure"
    
    @pytest.mark.asyncio
    async def test_send_error(self, handler, mock_websocket):
        """Test sending error message"""
        error_message = "Test error message"
        
        await handler._send_error(mock_websocket, error_message)
        
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should be error type"
        assert response["message"] == error_message, "Should contain error message"
    
    @pytest.mark.asyncio
    async def test_process_invalid_json(self, handler, mock_websocket, mock_processor):
        """Test processing invalid JSON message"""
        invalid_json = "{ invalid json }"
        
        await handler._process_message(mock_websocket, invalid_json, mock_processor)
        
        # Should send error response
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Invalid JSON format" in response["message"], "Should indicate JSON error"
    
    @pytest.mark.asyncio
    async def test_process_unknown_message_type(self, handler, mock_websocket, mock_processor, manager):
        """Test processing unknown message type"""
        unknown_message = json.dumps({
            "type": "unknown_type",
            "data": "some data"
        })
        
        # Mock manager methods
        manager.update_activity = Mock()
        manager.is_connected = Mock(return_value=True)
        
        await handler._process_message(mock_websocket, unknown_message, mock_processor)
        
        # Should send error response
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        response = json.loads(call_args)
        
        assert response["type"] == "error", "Should respond with error"
        assert "Unknown message type" in response["message"], "Should indicate unknown type"


class TestIntegration:
    """Integration tests for WebSocket components"""
    
    @pytest.fixture
    def manager(self):
        """Create a ConnectionManager instance for testing"""
        return ConnectionManager()
    
    @pytest.fixture
    def handler(self, manager):
        """Create a WebSocketHandler instance for testing"""
        return WebSocketHandler(manager)
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        return websocket
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_connection_lifecycle(self, manager, handler, mock_websocket):
        """Test complete connection lifecycle"""
        # Connect
        await manager.connect(mock_websocket)
        assert manager.is_connected(mock_websocket), "Should be connected"
        
        # Process activity
        manager.update_activity(mock_websocket)
        info = manager.get_connection_info(mock_websocket)
        assert info['messages_processed'] == 1, "Should have processed 1 message"
        
        # Disconnect
        manager.disconnect(mock_websocket)
        assert not manager.is_connected(mock_websocket), "Should be disconnected"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_connections(self, manager):
        """Test handling multiple simultaneous connections"""
        websockets = []
        num_connections = 3
        
        # Create multiple mock websockets
        for i in range(num_connections):
            ws = Mock()
            ws.accept = AsyncMock()
            websockets.append(ws)
        
        # Connect all
        for ws in websockets:
            await manager.connect(ws)
        
        # Verify all connected
        stats = manager.get_connection_stats()
        assert stats['total_connections'] == num_connections, f"Should have {num_connections} connections"
        assert stats['active_processors'] == num_connections, f"Should have {num_connections} processors"
        
        # Disconnect all
        for ws in websockets:
            manager.disconnect(ws)
        
        # Verify all disconnected
        stats = manager.get_connection_stats()
        assert stats['total_connections'] == 0, "Should have 0 connections after disconnect"
        assert stats['active_processors'] == 0, "Should have 0 processors after disconnect"
