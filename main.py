import random
import time
import torch
import sys
# KDPM2AncestralDiscreteScheduler new implementation, wainting distilled !
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

repo_id = "stabilityai/stable-diffusion-2"
# torch_dtype=torch.float16, revision="fp16"
pipe = DiffusionPipeline.from_pretrained(repo_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator().manual_seed(random.getrandbits(16))

prompt = sys.argv[1] or "An improvised prompt"
negative_prompt = sys.argv[2] or ""
steps = int(sys.argv[3]) or 20
scale = int(sys.argv[4]) or 9
image = pipe(prompt, guidance_scale=scale, num_inference_steps=steps, generator=generator, negative_prompt=negative_prompt).images[0]
image.save(f"out{round(time.time())}.png")
