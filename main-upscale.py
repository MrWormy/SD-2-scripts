from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

device = "cuda"

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# let's open an image
low_res_img = Image.open("./astronaut.png").convert("RGB")

prompt = "a white cat"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled_astronaut.png")
