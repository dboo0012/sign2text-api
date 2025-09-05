# Model Integration Guide

## Overview

This document describes how the sign language recognition model is integrated into the FastAPI backend.

## Architecture

### Simple Integration Approach

Instead of complex model wrappers, we use a straightforward approach:

1. **Model Loading**: Models are loaded directly in `KeypointsProcessor.__init__()`
2. **Data Conversion**: OpenPose data is converted to model format in `model_utils.py`
3. **Inference**: Model prediction runs in the existing processing pipeline

## Key Files

### 1. `app/model_utils.py`

- `convert_openpose_to_model_input()`: Converts OpenPose sequence to model tensor
- `load_sign_model()`: Loads model, tokenizer, and config
- `run_model_inference()`: Runs prediction on preprocessed data

### 2. `app/keypoints_processor.py` (Modified)

- Added model initialization in `__init__()`
- Added keypoints buffering for sequence processing
- Integrated model prediction in `_analyze_structured_keypoints()`

### 3. `app/config.py` (Modified)

- Added `MODEL_CONFIG` section with configurable paths
- Environment variable support for model paths

## Data Flow

```
WebSocket Input (OpenPose JSON)
    ↓
Validation & Conversion to StructuredKeypoints
    ↓
Buffer Management (collect sequence frames)
    ↓
Convert to Model Input Format (model_utils.py)
    ↓
Model Inference (PyTorch)
    ↓
Decode Predictions (Tokenizer)
    ↓
Return Results via WebSocket
```

## Configuration

Set these environment variables or update `app/config.py`:

```bash
MODEL_WEIGHTS_PATH=model/how2sign/vn_model/glofe_vn_how2sign_0224.pt
MODEL_TOKENIZER_PATH=model/notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased
MODEL_CONFIG_PATH=model/how2sign/vn_model/exp_config.json
MODEL_CLIP_LENGTH=16
MODEL_BUFFER_SIZE=32
MODEL_MIN_SEQUENCE_LENGTH=16
```

## Model Requirements

1. **Model Files**: Place your model files in the paths specified in config
2. **Dependencies**: Install requirements with `pip install -r requirements.txt`
3. **GPU Support**: PyTorch will automatically use CUDA if available

## API Response Format

The WebSocket response now includes prediction results:

```json
{
  "type": "success",
  "success": true,
  "processed_data": {
    "prediction": {
      "success": true,
      "text": "hello world",
      "confidence": 0.95,
      "frames_processed": 16
    },
    "model_ready": true,
    "buffer_size": 18
  }
}
```

## Testing

1. **Start the server**: `python main.py`
2. **Connect via WebSocket**: Send OpenPose keypoints data
3. **Check logs**: Model loading and prediction logs will appear
4. **Monitor responses**: Predictions will be included in WebSocket responses

## Troubleshooting

### Model Not Loading

- Check file paths in config
- Verify model files exist
- Check dependencies are installed
- Look at server logs for specific errors

### No Predictions

- Ensure buffer has enough frames (min 16)
- Check OpenPose data format
- Verify keypoints are being detected
- Monitor buffer size in responses

### Performance Issues

- Reduce buffer size for faster responses
- Use GPU if available
- Consider model quantization for production
