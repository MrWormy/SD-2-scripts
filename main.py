import random
import time
import torch
import sys
from PIL import Image, PngImagePlugin
# KDPM2AncestralDiscreteScheduler new implementation, waiting distilled !
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

device = "cuda"
repo_id = "stabilityai/stable-diffusion-2-1"

pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16") if device == "cuda" \
    else DiffusionPipeline.from_pretrained(repo_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.requires_safety_checker = False
pipe.to(device)

prompt = sys.argv[1] or "An improvised prompt"
negative_prompt = sys.argv[2] or ""
steps = int(sys.argv[3]) or 20
scale = float(sys.argv[4]) or 9

for i in range(int(sys.argv[5]) or 10):
    seed = random.getrandbits(63)
    generator = torch.Generator(device=device).manual_seed(seed)
    # 3 batch size is max for 20G vram
    images = pipe(prompt, guidance_scale=scale, num_inference_steps=steps, generator=generator,
                  negative_prompt=negative_prompt, num_images_per_prompt=3).images

    for ind, image in enumerate(images):
        name = f"{round(time.time())}_{random.getrandbits(8)}"
        info = PngImagePlugin.PngInfo()
        info.add_text('prompt', prompt)
        info.add_text('nprompt', negative_prompt)
        info.add_text('gscale', f"{scale}")
        info.add_text('steps', f"{steps}")
        info.add_text('seed', f"{seed}")
        info.add_text('nsample', f"{ind}")
        image.save(f"out/{name}.png", pnginfo=info)
