# Testing Guide

This directory contains the test suite for the sign2text API using pytest.

## Quick Start

1. **Install dependencies** (including test dependencies):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run all tests**:

   ```bash
   python run_tests.py
   ```

3. **Run specific test categories**:

   ```bash
   # Fast unit tests only
   pytest -v -m "not slow"

   # MediaPipe tests only
   pytest -v -m "mediapipe"

   # WebSocket tests only
   pytest -v -m "websocket"

   # Integration tests
   pytest -v -m "integration"
   ```

## Test Structure

- **`test_mediapipe_pytest.py`** - MediaPipe functionality tests
- **`test_websocket.py`** - WebSocket connection and message handling tests
- **`conftest.py`** - Shared fixtures and pytest configuration
- **`__init__.py`** - Makes test directory a Python package

## Test Categories

Tests are organized with markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.mediapipe` - Tests requiring MediaPipe
- `@pytest.mark.websocket` - WebSocket-related tests

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test/test_mediapipe_pytest.py

# Run specific test function
pytest test/test_mediapipe_pytest.py::TestPoseProcessor::test_processor_initialization
```

### Advanced Options

```bash
# Skip slow tests
pytest -m "not slow"

# Run only MediaPipe tests
pytest -m "mediapipe"

# Run with coverage
pytest --cov=app --cov-report=html

# Run tests matching pattern
pytest -k "test_websocket"

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l
```

## Test Requirements

- **OpenCV**: `pip install opencv-python`
- **MediaPipe**: `pip install mediapipe`
- **pytest**: `pip install pytest`
- **pytest-asyncio**: For async test support
- **pytest-mock**: For mocking capabilities

## Adding New Tests

1. Create test files with `test_*.py` naming
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Follow the existing fixture patterns in `conftest.py`
4. Add docstrings to describe test purpose

Example:

```python
@pytest.mark.unit
def test_my_function():
    """Test description"""
    # Test implementation
    assert result == expected
```

## Fixtures Available

- `test_frame` - Pre-created test image frame
- `mock_websocket` - Mock WebSocket for testing
- `connection_manager` - Fresh ConnectionManager instance
- `websocket_handler` - WebSocketHandler with manager
- `pose_processor` - MediaPipe processor instance

## Troubleshooting

**MediaPipe tests failing**: Install MediaPipe with `pip install mediapipe`

**Import errors**: Ensure you're running from the project root directory

**Slow tests**: Use `-m "not slow"` to skip performance tests

**Coverage reports**: Install with `pip install pytest-cov`
