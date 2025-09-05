from fastapi import FastAPI, WebSocket
import os

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("👋 Hello from backend WebSocket!")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception:
        await websocket.close()
