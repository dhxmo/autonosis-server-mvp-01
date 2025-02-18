import json
import os
import shutil
from pathlib import Path
from typing import Any

import open_clip
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from ray import serve
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


app = FastAPI()

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create upload directory
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)


async def validate_file_extension(filename: str) -> bool:
    """Validate file extension"""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.2})
@serve.ingress(app)
class CaptionXrayChest:
    def __init__(self):
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="model/chest_xray_findings.pt"
        )

    @app.post("/infer")
    async def infer(
        self, file: UploadFile = File(...), data: str = Form(...)
    ) -> dict[str, str | None]:
        extra_data = json.loads(data)  # Convert JSON string back to dictionary
        print("extra_data", extra_data["template_findings"])

        try:
            contents = file.file.read()
            with open(os.path.join(UPLOAD_FOLDER, file.filename), "wb") as f:
                f.write(contents)
        except Exception:
            raise HTTPException(status_code=500, detail="Something went wrong")
        finally:
            file.file.close()

        caption = self.generate_caption(file.filename)
        return {"updated_findings": caption}

    def generate_caption(self, img_path) -> str:

        im = Image.open(img_path).convert("RGB")
        im = self.transform(im).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(im)

        return (
            open_clip.decode(generated[0])
            .split("<end_of_text>")[0]
            .replace("<start_of_text>", "")
        )


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.8, "num_gpus": 1})
class MedModel:
    def __init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="microsoft/phi-4",
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

    @app.post("/chat")
    async def chat(self, data: str = Form(...)) -> dict[str, Any]:
        extra_data = json.loads(data)
        user_prompt = extra_data["audio_data"]
        prev_findings = extra_data["prev_findings"]

        messages = [
            {
                "role": "system",
                "content": "You need to edit a given finding with the template provided. Edit the "
                "template_finding to accommodate what the new_finding wants to add. "
                "Maintain the structure of template_finding, just update the information from "
                "new_findings into template_findings. Provide the result in the exact same "
                "format as the template_findings.",
            },
            {
                "role": "user",
                "content": f"template_findings: {prev_findings} \n\n new_findings: {user_prompt}",
            },
        ]

        outputs = self.pipeline(messages, max_new_tokens=128)
        response = outputs[0]["generated_text"][-1]

        return {"updated_audio_text": response}


# caption_app = CaptionXrayChest.bind()
caption_app = MedModel.bind()
