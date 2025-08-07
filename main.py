from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
  return {"message": "Hello World"}

@app.get("/ping")
async def health():
    return {
        "status": "healthy",
        "mediapipe_available": True,
        # "active_connections": len(manager.active_connections),
        "version": "1.0.0"
    }