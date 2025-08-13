"""
Shared pytest fixtures and configuration
"""
import sys
import os
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, AsyncMock

# Add parent directory to path so we can import the app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pose_processor import PoseProcessor
from app.connection_manager import ConnectionManager
from app.websocket_handler import WebSocketHandler


@pytest.fixture(scope="session")
def test_frame():
    """Create a reusable test frame for all tests"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some simple shapes to make it interesting for MediaPipe
    cv2.rectangle(frame, (200, 150), (440, 330), (100, 150, 200), -1)  # Body-like rectangle
    cv2.circle(frame, (320, 120), 40, (150, 200, 100), -1)  # Head-like circle
    cv2.rectangle(frame, (180, 200), (220, 280), (120, 180, 150), -1)  # Left arm
    cv2.rectangle(frame, (420, 200), (460, 280), (120, 180, 150), -1)  # Right arm
    
    return frame


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for testing"""
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def connection_manager():
    """Create a fresh ConnectionManager for each test"""
    return ConnectionManager()


@pytest.fixture
def websocket_handler(connection_manager):
    """Create a WebSocketHandler with a fresh ConnectionManager"""
    return WebSocketHandler(connection_manager)


@pytest.fixture
def pose_processor():
    """Create a PoseProcessor for tests that need MediaPipe functionality"""
    return PoseProcessor()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "mediapipe: Tests requiring MediaPipe"
    )
    config.addinivalue_line(
        "markers", "websocket: WebSocket related tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/content"""
    for item in items:
        # Mark MediaPipe tests
        if "mediapipe" in item.nodeid.lower() or any("mediapipe" in marker.name for marker in item.iter_markers()):
            item.add_marker(pytest.mark.mediapipe)
        
        # Mark WebSocket tests
        if "websocket" in item.nodeid.lower() or any("websocket" in marker.name for marker in item.iter_markers()):
            item.add_marker(pytest.mark.websocket)
        
        # Mark slow tests
        if any("slow" in marker.name for marker in item.iter_markers()):
            item.add_marker(pytest.mark.slow)


# Skip MediaPipe tests if not available
def pytest_runtest_setup(item):
    """Skip tests if required dependencies are not available"""
    markers = [marker.name for marker in item.iter_markers()]
    
    if "mediapipe" in markers:
        try:
            import mediapipe
        except ImportError:
            pytest.skip("MediaPipe not available - install with: pip install mediapipe")
    
    # Skip slow tests by default unless explicitly requested
    if "slow" in markers and not item.config.getoption("--run-slow", default=False):
        pytest.skip("Slow test skipped - use --run-slow to run")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
