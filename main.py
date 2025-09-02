from fastapi import Request, FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.websocket_handler import WebSocketHandler
import logging
from pydantic import BaseModel
from typing import List
import json
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define OpenPose models so WebSocket can parse them if needed
class Person(BaseModel):
    person_id: List[int]
    pose_keypoints_2d: List[float]
    face_keypoints_2d: List[float]
    hand_left_keypoints_2d: List[float]
    hand_right_keypoints_2d: List[float]
    pose_keypoints_3d: List[float]
    face_keypoints_3d: List[float]
    hand_left_keypoints_3d: List[float]
    hand_right_keypoints_3d: List[float]


class OpenPoseData(BaseModel):
    version: float
    people: List[Person]


ws_handler = WebSocketHandler()

# Dependency for getting the WebSocket handler
def get_websocket_handler() -> WebSocketHandler:
    """Dependency injection for WebSocket handler"""
    return ws_handler


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": f"{settings.API_TITLE} is running",
        "status": "healthy",
    }


@app.get("/ping")
async def health():
    """Health check endpoint with connection status"""
    connection_stats = ws_handler.get_connection_stats()
    return {
        "status": "healthy",
        "mediapipe_available": True,
        "connection_active": connection_stats["connected"],
        "messages_processed": connection_stats["messages_processed"],
        "version": settings.API_VERSION,
    }


@app.websocket("/ws/video_stream")
async def websocket_endpoint(
    websocket: WebSocket, handler: WebSocketHandler = Depends(get_websocket_handler)
):
    """WebSocket endpoint for video stream processing"""
    await websocket.accept()
    logger.info("✅ WebSocket connected")

    try:
        while True:
            raw_message = await websocket.receive_text()
            data = json.loads(raw_message)

            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                logger.info("🔄 Received ping → sent pong")

            elif msg_type == "frame":
                frame_bytes = bytes(data["data"])
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    logger.info(
                        f"📷 Received frame {frame.shape} ({data.get('format', 'jpeg')})"
                    )
                    # Example: Save frame (overwrites latest.jpg each time)
                    cv2.imwrite("latest.jpg", frame)
                else:
                    logger.warning("⚠️ Failed to decode frame")

            elif msg_type == "keypoint_sequence":
                logger.info("🧍 Received keypoint sequence")
                # TODO: forward to Mediapipe/OpenPose pipeline

            else:
                logger.warning(f"⚠️ Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("❌ WebSocket disconnected")


@app.post("/openpose")
async def receive_openpose(request: Request):
    """Fallback REST endpoint for OpenPose data"""
    data = await request.json()
    logger.info("✅ Received OpenPose data: %s", data)
    return {"status": "ok", "people_detected": len(data.get("people", []))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL,
    )
