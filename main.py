import json
import os
import shutil
from pathlib import Path

import open_clip
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from ray import serve

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

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.6, "num_gpus": 1})
@serve.ingress(app)
class CaptionXrayChest:
    def __init__(self):
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="model/chest_xray_findings.pt"
        )

    @app.post("/infer")
    async def infer(self, file: UploadFile = File(...), data: str = Form(...)) -> dict[str, str | None]:
        extra_data = json.loads(data)  # Convert JSON string back to dictionary
        print("extra_data", extra_data["template_findings"])

        try:
            contents = file.file.read()
            with open(file.filename, 'wb') as f:
                f.write(contents)
        except Exception:
            raise HTTPException(status_code=500, detail='Something went wrong')
        finally:
            file.file.close()

        return {"message": f"Successfully uploaded {file.filename}"}

        # # download multipart image
        #
        # img_path = "model/test.png"
        #
        # caption = self.generate_caption(img_path)
        # print("caption", caption)
        #
        # return caption

    def generate_caption(self, img_path) -> str:

        im = Image.open(img_path).convert("RGB")
        im = self.transform(im).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(im)

        return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")


caption_app = CaptionXrayChest.bind()






