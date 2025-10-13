"""
Test script to verify OpenPose integration with the websocket handler
"""
import cv2
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_openpose_extractor():
    """Test OpenPose extractor initialization and keypoint extraction"""
    try:
        from app.openpose_extractor import OpenPoseExtractor
        from app.logger import setup_logger
        
        logger = setup_logger(__name__)
        
        logger.info("="*50)
        logger.info("Testing OpenPose Integration")
        logger.info("="*50)
        
        # Initialize OpenPose extractor
        logger.info("\n1. Initializing OpenPose extractor...")
        extractor = OpenPoseExtractor()
        
        if not extractor.is_initialized:
            logger.error("✗ OpenPose extractor failed to initialize")
            return False
        
        logger.info("✓ OpenPose extractor initialized successfully")
        
        # Test with sample image
        logger.info("\n2. Testing keypoint extraction with sample image...")
        image_path = "/openpose/examples/media/COCO_val2014_000000000192.jpg"
        
        if not os.path.exists(image_path):
            logger.warning(f"⚠️ Sample image not found at {image_path}")
            logger.info("Skipping image test")
        else:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"✗ Could not read image from {image_path}")
                return False
            
            logger.info(f"✓ Loaded sample image: {frame.shape}")
            
            # Extract keypoints
            openpose_data = extractor.extract_keypoints(frame)
            
            if openpose_data is None:
                logger.error("✗ Failed to extract keypoints")
                return False
            
            logger.info(f"✓ Extracted keypoints successfully")
            logger.info(f"  - Number of people detected: {len(openpose_data.people)}")
            
            if len(openpose_data.people) > 0:
                from app.models import StructuredKeypoints
                
                person = openpose_data.people[0]
                structured = StructuredKeypoints.from_openpose_person(person)
                summary = structured.get_detection_summary()
                
                logger.info(f"  - Pose keypoints: {summary['pose_points']}")
                logger.info(f"  - Face keypoints: {summary['face_points']}")
                logger.info(f"  - Left hand keypoints: {summary['left_hand_points']}")
                logger.info(f"  - Right hand keypoints: {summary['right_hand_points']}")
                logger.info(f"  - Total keypoints: {summary['total_points']}")
        
        # Get stats
        logger.info("\n3. OpenPose extractor stats:")
        stats = extractor.get_stats()
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
        
        logger.info("\n" + "="*50)
        logger.info("✓ All tests passed!")
        logger.info("="*50)
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Make sure OpenPose is installed and available")
        return False
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openpose_extractor()
    sys.exit(0 if success else 1)
