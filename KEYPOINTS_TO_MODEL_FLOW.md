# Keypoints to Model Inference Flow

## Overview
This document explains how extracted keypoints are passed from OpenPose extraction to the sign language recognition model for inference.

## Data Flow Architecture

```
Frame Input (WebSocket)
    ↓
OpenPoseExtractor.extract_keypoints()
    ↓
Raw OpenPoseData (unedited)
    ↓
KeypointsProcessor.process_keypoints()
    ↓
Buffering (16 frames minimum)
    ↓
model_utils.run_model_inference()
    ↓
Model Prediction Result
    ↓
WebSocket Response to Client
```

## Implementation Details

### 1. Frame Reception (`websocket_handler.py`)
- WebSocket receives frame data from the client
- Frame is decoded from JPEG/PNG bytes to OpenCV image format
- OpenPoseExtractor extracts keypoints from the frame

### 2. Keypoint Extraction (`openpose.py`)
- OpenPose processes the frame and extracts:
  - Body pose keypoints (25 points)
  - Face keypoints (70 points)
  - Left hand keypoints (21 points)
  - Right hand keypoints (21 points)
- Returns raw `OpenPoseData` object with unedited keypoint coordinates

### 3. Keypoint Processing (`websocket_handler.py` → `keypoints_processor.py`)
**Key Changes:**
```python
# Pass raw OpenPoseData directly to the processor
if self.keypoints_processor:
    processing_result = await self.keypoints_processor.process_keypoints(
        keypoint_data=extracted_keypoints,  # Raw, unedited OpenPoseData
        frame_info=None,
        timestamp=message.timestamp
    )
```

### 4. Frame Buffering (`keypoints_processor.py`)
- Raw OpenPoseData frames are stored in a buffer: `self.keypoints_buffer`
- Buffer configuration:
  - **Maximum buffer size**: 32 frames
  - **Minimum sequence length**: 16 frames (required for inference)
- Oldest frames are discarded when buffer exceeds maximum size

### 5. Model Inference (`model_utils.py`)
When buffer has sufficient frames (≥16):
1. Extract last 16 frames from buffer
2. Convert OpenPoseData sequence to numpy array format
3. Call `run_inference_with_data(joints)` with raw keypoint data
4. Return prediction result to WebSocket handler

### 6. Response to Client (`websocket_handler.py`)
The response includes:
```json
{
  "type": "processing_response",
  "success": true,
  "message": "Frame processed with keypoint extraction",
  "processed_data": {
    "keypoints_extracted": true,
    "keypoints_summary": {...},
    "keypoints": {...},
    "model_inference": {
      "prediction": {
        "success": true,
        "text": "PREDICTED_SIGN",
        "confidence": 0.95,
        "status": "complete"
      },
      "buffer_size": 16,
      "model_ready": true
    }
  }
}
```

## Key Features

### ✅ Raw Keypoint Data
- **No preprocessing or modification** of keypoint coordinates
- Maintains original OpenPose output format
- Preserves all coordinate values (x, y, confidence)

### ✅ Buffering Strategy
- Accumulates frames until minimum sequence length is reached
- Status messages during buffering phase:
  ```json
  {
    "prediction": {
      "success": false,
      "status": "buffering",
      "frames_needed": 5
    }
  }
  ```

### ✅ Logging
The implementation includes detailed logging:
- `🔍` Keypoints extracted per frame
- `📊` Buffering progress
- `🤖` Model predictions with confidence scores
- `⚠️` Warnings for failures

## Testing

### Manual Testing
Send frame data via WebSocket and observe:
1. Keypoint extraction logs
2. Buffer accumulation progress
3. Model inference when buffer is full
4. Prediction results in response

### Example Log Output
```
🔍 Keypoints processed - Pose: 25, Face: 70, Left Hand: 21, Right Hand: 21
📊 Buffering frames... need 10 more frames
📊 Buffering frames... need 5 more frames
🤖 Model prediction: 'HELLO' (confidence: 0.95)
```

## Configuration

### Buffer Settings (`keypoints_processor.py`)
```python
self.keypoints_buffer: List[OpenPoseData] = []
self.max_buffer_size = 32
self.min_sequence_length = 16
```

### Adjusting Buffer Size
To change how many frames are required for inference:
```python
# In KeypointsProcessor.__init__()
self.min_sequence_length = 20  # Require 20 frames instead of 16
```

## Error Handling

### No Keypoints Processor Available
```python
if not self.keypoints_processor:
    logger.warning("⚠️ Keypoints processor not available")
```

### Processing Failure
```python
if not processing_result.success:
    logger.warning(f"⚠️ Keypoints processing failed: {processing_result.error}")
```

### Model Inference Failure
```python
if not prediction_result.get("success"):
    logger.warning(f"Model prediction failed: {prediction_result.get('error')}")
```

## Future Enhancements

1. **Sliding Window**: Implement overlapping frame windows for continuous predictions
2. **Real-time Optimization**: Process frames at variable intervals (e.g., every 3rd frame)
3. **Multiple Person Support**: Handle keypoints from multiple detected people
4. **Confidence Filtering**: Only process frames with high-confidence keypoint detections
5. **Buffer Management**: Implement FIFO queue with time-based expiration

## Summary

✅ **Implemented**: Raw OpenPoseData is now passed directly to the keypoints processor  
✅ **Implemented**: Buffering system accumulates frames until sufficient data is available  
✅ **Implemented**: Model inference runs automatically when buffer is full  
✅ **Implemented**: Results are returned to client via WebSocket responses  
✅ **Maintained**: No preprocessing or editing of keypoint data before model inference  
