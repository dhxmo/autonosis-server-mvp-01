import open_clip
import torch
from PIL import Image
from fastapi import FastAPI
from ray import serve

app = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.7, "num_gpus": 0})
@serve.ingress(app)
class CaptionXrayChest:
    def __init__(self):
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="model/chest_xray_findings.pt"
        )

    @app.post("/infer")
    def infer(self) -> str:
        # download multipart image

        img_path = "model/test.png"

        caption = self.generate_caption(img_path)
        print("caption", caption)

        return caption

    def generate_caption(self, img_path) -> str:

        im = Image.open(img_path).convert("RGB")
        im = self.transform(im).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(im)

        return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")


caption_app = CaptionXrayChest.bind()






