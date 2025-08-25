from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.websocket_handler import WebSocketHandler
from app.mediapipe_websocket_handler import MediaPipeWebSocketHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Global WebSocket handlers
ws_handler = WebSocketHandler()  # Original OpenPose handler
mediapipe_ws_handler = MediaPipeWebSocketHandler()  # MediaPipe handler

# Dependencies for getting WebSocket handlers
def get_websocket_handler() -> WebSocketHandler:
    """Dependency injection for OpenPose WebSocket handler"""
    return ws_handler

def get_mediapipe_websocket_handler() -> MediaPipeWebSocketHandler:
    """Dependency injection for MediaPipe WebSocket handler"""
    return mediapipe_ws_handler

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
    openpose_stats = ws_handler.get_connection_stats()
    mediapipe_stats = mediapipe_ws_handler.get_connection_stats()
    
    return {
        "status": "healthy",
        "mediapipe_available": True,
        "openpose_connection_active": openpose_stats["connected"],
        "mediapipe_connection_active": mediapipe_stats["connected"],
        "openpose_messages_processed": openpose_stats["messages_processed"],
        "mediapipe_messages_processed": mediapipe_stats["messages_processed"],
        "version": settings.API_VERSION
    }

@app.websocket("/ws/video_stream")
async def websocket_endpoint(
    websocket: WebSocket, 
    handler: WebSocketHandler = Depends(get_websocket_handler)
):
    """WebSocket endpoint for OpenPose video stream processing using dependency injection"""
    await handler.handle_connection(websocket)

@app.websocket("/ws/mediapipe_stream")
async def mediapipe_websocket_endpoint(
    websocket: WebSocket, 
    handler: MediaPipeWebSocketHandler = Depends(get_mediapipe_websocket_handler)
):
    """WebSocket endpoint for MediaPipe keypoints processing"""
    await handler.handle_connection(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL
    )