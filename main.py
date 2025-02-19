import json
import uuid
from pathlib import Path
from typing import Dict

import transformers
from fastapi import FastAPI, WebSocket, Request
from starlette.middleware.cors import CORSMiddleware
import aiofiles
from starlette.websockets import WebSocketDisconnect
from pywhispercpp.model import Model

from model import transcribe_audio_file, llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("media").mkdir(exist_ok=True)

# Load models on startup
whisper_model = Model("medium.en")
llm_pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
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
    audio_text = transcribe_audio_file(whisper_model, audio_file)

    # call llm
    updated_text = llm(
        pipeline=llm_pipeline,
        prev_diagnosis=req_body["curr_text"],
        user_prompt=audio_text,
    )

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

        async with aiofiles.open(manager.recording_files[client_id], "wb") as out_file:
            while True:
                data = await websocket.receive_bytes()
                await out_file.write(data)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error in websocket connection: {e}")
        await websocket.close(code=1000, reason="Server error")
