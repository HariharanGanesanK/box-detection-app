import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

app = FastAPI(title="Test Backend")

# Enable CORS (for frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Simple API routes
# -----------------------

@app.get("/")
async def root():
    return {"message": "✅ Backend is running!"}

@app.get("/ping")
async def ping():
    print("📩 Received request at /ping")
    return {"response": "pong", "status": "ok"}

# -----------------------
# WebSocket Test
# -----------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text("🔌 WebSocket connected!")
    try:
        while True:
            data = await ws.receive_text()
            print(f"📡 Received via WebSocket: {data}")
            await ws.send_text(f"Echo: {data}")
    except Exception:
        print("❌ WebSocket closed")

# -----------------------
# Run with Uvicorn (for Render)
# -----------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
