import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Set

from backend.data_stream import DataStreamer
from backend.inference import load_model, predict_window
from backend.mode import ModeController

app = FastAPI(title="DigitalEpi Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Load model
model, scaler = load_model()
mode_ctrl = ModeController()
streamer = DataStreamer("model/data/stream_source.csv", window_size=30, delay=1.0)

clients: Set[WebSocket] = set()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "mode": mode_ctrl.current_mode(),
        "model_loaded": model is not None
    }

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        async for window in streamer.stream():   # DataStreamer yields rolling window
            score = predict_window(model, scaler, window)
            hr, rr, spo2 = window[-1]

            payload = {
                "mode": mode_ctrl.current_mode(),
                "anomaly_score": float(score),
                "hr": hr,
                "rr": rr,
                "spo2": spo2,
                "window": window  # send entire vitals window
            }

             # debug: see latest rolling window in console
            await ws.send_json(payload)

    except WebSocketDisconnect:
        clients.remove(ws)
