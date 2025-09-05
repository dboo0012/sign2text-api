"""
Simple test to verify model integration with simplified approach
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_simple_model_integration():
    """Test the simplified model integration"""
    print("Testing simplified model integration...")
    
    try:
        from app.keypoints_processor import KeypointsProcessor
        from app.models import OpenPoseData, OpenPosePerson
        
        # Initialize processor
        processor = KeypointsProcessor()
        print("✓ KeypointsProcessor initialized")
        
        # Create dummy OpenPose data with proper structure
        dummy_person = OpenPosePerson(
            pose_keypoints_2d=[1.0, 2.0, 0.8] * 25,  # 25 pose points * 3 values
            hand_left_keypoints_2d=[1.0, 2.0, 0.8] * 21,  # 21 hand points * 3 values  
            hand_right_keypoints_2d=[1.0, 2.0, 0.8] * 21,  # 21 hand points * 3 values
            face_keypoints_2d=[1.0, 2.0, 0.8] * 70  # 70 face points * 3 values
        )
        
        dummy_openpose = OpenPoseData(
            version=1.0,
            people=[dummy_person]
        )
        
        print("✓ Created dummy OpenPose data")
        
        # Process multiple frames to build up buffer
        for i in range(18):  # More than min_sequence_length (16)
            result = await processor.process_keypoints(dummy_openpose)
            print(f"Frame {i+1}: success={result.success}, buffer_size={result.analysis_result.get('buffer_size', 0) if result.analysis_result else 0}")
        
        # Check the final result
        if result.analysis_result:
            prediction = result.analysis_result.get('prediction', {})
            print(f"Final prediction status: {prediction.get('status', 'N/A')}")
            if prediction.get('success'):
                print(f"✓ Model prediction: {prediction.get('text', 'N/A')}")
            else:
                print(f"✗ Model prediction failed: {prediction.get('error', 'N/A')}")
        
        print("Test completed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_model_integration())
