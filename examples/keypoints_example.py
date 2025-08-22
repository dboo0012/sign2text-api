"""
Example usage of the improved keypoints data types
"""
import asyncio
import json
from typing import Dict, Any

# Import your models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    OpenPoseData, StructuredKeypoints, 
    KeypointWithConfidence, PoseKeypoint
)
from app.keypoints_processor import KeypointsProcessor

# Sample OpenPose data (from your attachment)
SAMPLE_OPENPOSE_DATA = {
    "version": 1.3,
    "people": [{
        "person_id": [-1],
        "pose_keypoints_2d": [
            666.606, 216.053, 0.87404,
            682.248, 351.145, 0.801615,
            578.438, 349.25, 0.629371,
            484.438, 505.973, 0.743795,
            523.539, 353.198, 0.746404
        ],
        "face_keypoints_2d": [
            625.031, 202.339, 0.801791,
            626.666, 215.422, 0.901775,
            630.482, 227.96, 0.897332
        ],
        "hand_left_keypoints_2d": [
            694.804, 451.279, 0.191344,
            674.989, 441.371, 0.210579,
            649.512, 437.833, 0.164115
        ],
        "hand_right_keypoints_2d": [
            539.124, 308.146, 0.0346953,
            536.442, 294.067, 0.0763414,
            562.588, 292.726, 0.0584684
        ],
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": []
    }]
}

async def demonstrate_keypoint_processing():
    """Demonstrate the new keypoint processing capabilities"""
    
    print("🔧 Creating KeypointsProcessor...")
    processor = KeypointsProcessor()
    
    print("\n📊 Processing OpenPose Data...")
    
    # Method 1: Direct OpenPose data processing
    result1 = await processor.process_keypoints(SAMPLE_OPENPOSE_DATA)
    print(f"✅ Direct processing result: {result1.success}")
    if result1.success:
        analysis = result1.analysis_result
        print(f"   Detection summary: {analysis['processing_info']['detection_summary']}")
        print(f"   Feature vector size: {analysis['feature_vector_size']}")
    
    # Method 2: Convert to structured format first
    print("\n🔄 Converting to structured format...")
    openpose_obj = OpenPoseData(**SAMPLE_OPENPOSE_DATA)
    structured_keypoints = StructuredKeypoints.from_openpose_person(openpose_obj.people[0])
    
    print(f"   Structured keypoints created:")
    summary = structured_keypoints.get_detection_summary()
    for key, value in summary.items():
        print(f"     {key}: {value}")
    
    # Method 3: Process structured keypoints
    result2 = await processor.process_keypoints(structured_keypoints)
    print(f"\n✅ Structured processing result: {result2.success}")
    
    # Method 4: Create manual keypoints
    print("\n🎯 Creating manual keypoints...")
    manual_keypoints = StructuredKeypoints(
        pose=[
            PoseKeypoint(x=100.0, y=200.0, z=0.0, visibility=0.9),
            PoseKeypoint(x=150.0, y=250.0, z=0.0, visibility=0.8)
        ],
        left_hand=[
            KeypointWithConfidence(x=80.0, y=180.0, confidence=0.7),
            KeypointWithConfidence(x=85.0, y=185.0, confidence=0.6)
        ],
        right_hand=[
            KeypointWithConfidence(x=220.0, y=180.0, confidence=0.8)
        ],
        face=[]
    )
    
    result3 = await processor.process_keypoints(manual_keypoints)
    print(f"✅ Manual processing result: {result3.success}")
    
    # Get feature vector for model input
    features = manual_keypoints.to_numpy_features()
    print(f"   Feature vector: {features[:10]}... (length: {len(features)})")
    
    print("\n📈 Processing Stats:")
    stats = processor.get_processing_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

def demonstrate_data_types():
    """Demonstrate the different data type capabilities"""
    
    print("\n🧪 Data Type Demonstrations:")
    
    # 1. Point creation
    point = KeypointWithConfidence(x=100.0, y=200.0, confidence=0.85)
    print(f"   Point: {point}")
    
    # 2. Point from list
    point_from_list = KeypointWithConfidence.from_list([150.0, 250.0, 0.9])
    print(f"   Point from list: {point_from_list}")
    
    # 3. Pose keypoint
    pose_point = PoseKeypoint(x=100.0, y=200.0, z=5.0, visibility=0.95)
    print(f"   Pose point: {pose_point}")
    
    # 4. OpenPose parsing
    openpose_data = OpenPoseData(**SAMPLE_OPENPOSE_DATA)
    print(f"   OpenPose version: {openpose_data.version}")
    print(f"   Number of people: {len(openpose_data.people)}")
    
    if openpose_data.people:
        person = openpose_data.people[0]
        structured = StructuredKeypoints.from_openpose_person(person)
        print(f"   Converted to structured: {structured.get_detection_summary()}")

if __name__ == "__main__":
    print("🚀 Keypoints Data Types Example")
    print("=" * 50)
    
    # Demonstrate data types
    demonstrate_data_types()
    
    # Demonstrate processing
    print("\n" + "=" * 50)
    asyncio.run(demonstrate_keypoint_processing())
    
    print("\n✨ Example completed!")
