"""
Shared pytest fixtures and configuration
"""
import sys
import os
import pytest
from unittest.mock import Mock, AsyncMock

# Add parent directory to path so we can import the app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.websocket_handler import WebSocketHandler

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
def websocket_handler():
    """Create a WebSocketHandler for testing"""
    return WebSocketHandler()


@pytest.fixture
def sample_keypoints():
    """Create sample keypoints data for testing"""
    return {
        "pose": [[0.5, 0.5, 0.0, 0.9]] * 33,  # 33 pose landmarks
        "face": [[0.5, 0.5, 0.0]] * 468,       # 468 face landmarks
        "left_hand": [[0.3, 0.5, 0.0]] * 21,   # 21 left hand landmarks
        "right_hand": [[0.7, 0.5, 0.0]] * 21   # 21 right hand landmarks
    }


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
