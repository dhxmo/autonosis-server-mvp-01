import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict

import aiofiles
import numpy as np
from fastapi import FastAPI, WebSocket, Request
from pywhispercpp.model import Model
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from model import transcribe_audio_file, ollama_llm

# Store models in a dictionary
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create media folder if it doesnt exist
    Path("media").mkdir(exist_ok=True)

    # Load the models
    models["whisper"] = Model("base.en")

    # --- Warm up the models

    # whisper warm up
    sample_rate = 16000  # Standard sample rate for Whisper
    duration = 3.0  # Duration in seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a sine wave at 440 Hz (A4 note)
    dummy_audio = np.sin(2 * np.pi * 440 * t)
    # Add some noise to make it more realistic
    dummy_audio += 0.5 * np.random.randn(len(dummy_audio))
    # Normalize to [-1, 1] range
    dummy_audio = dummy_audio / np.abs(dummy_audio).max()
    # Reshape to match expected format (batch_size, audio_length)
    dummy_audio = dummy_audio.reshape(1, -1)
    models["whisper"].transcribe(dummy_audio)

    yield

    # Clean up resources
    models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def root():
    return {"message": "pong"}


@app.get("/get_template")
def get_template(modality: str, organ: str):
    with open("assets/findings_template.json", "r") as f:
        json_data = json.load(f)

    return {"findings_template": json_data[modality.lower()][organ.lower()]}


@app.post("/update_text")
async def update_text(request: Request):
    req_body = await request.json()

    audio_file = f"media/{str(req_body['audio_uuid'])}.webm"

    # transcribe audio file
    audio_text = transcribe_audio_file(models["whisper"], audio_file)
    print("audio_text", audio_text)

    # call llm
    updated_text = ollama_llm(
        prev_diagnosis=req_body["curr_text"],
        user_prompt=audio_text,
    )
    print("updated_text", updated_text)

    return {"updated_text": updated_text}


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str] = {}
        self.recording_files: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # Create UUID for this recording session
        audio_uuid = str(uuid.uuid4())
        self.recording_files[client_id] = f"media/{audio_uuid}.webm"

        # Send UUID back to client immediately
        await websocket.send_json({"event_type": "audio_uuid", "uuid": audio_uuid})

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            self.active_connections.pop(client_id)
            if client_id in self.recording_files:
                self.recording_files.pop(client_id)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        await manager.connect(websocket, client_id)

        # Delete previous file if it exists
        if client_id in manager.recording_files:
            try:
                os.remove(manager.recording_files[client_id])
            except OSError:
                pass

        async with aiofiles.open(manager.recording_files[client_id], "wb") as out_file:
            while True:
                data = await websocket.receive_bytes()
                await out_file.write(data)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error in websocket connection: {e}")
        await websocket.close(code=1000, reason="Server error")
