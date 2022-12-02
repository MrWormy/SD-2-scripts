import time
import torch
import sys
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

repo_id = "stabilityai/stable-diffusion-2"
# torch_dtype=torch.float16, revision="fp16"
pipe = DiffusionPipeline.from_pretrained(repo_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator().manual_seed(52)

prompt = sys.argv[1] or "An improvised prompt"
negative_prompt = sys.argv[2] or ""
image = pipe(prompt, guidance_scale=9, num_inference_steps=20, generator=generator, negative_prompt=negative_prompt).images[0]
image.save(f"out{round(time.time())}.png")
