import json

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_template")
def get_template(modality:str, organ:str):
    with open("findings_template.json", "r") as f:
        json_data = json.load(f)

    return {"findings_template": json_data[modality.lower()][organ.lower()]}