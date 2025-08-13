from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.websocket_handler import WebSocketHandler
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

# Global WebSocket handler (single connection)
ws_handler = WebSocketHandler()

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
        "version": settings.API_VERSION
    }

@app.websocket("/ws/video_stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for video stream processing"""
    await ws_handler.handle_connection(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL
    )