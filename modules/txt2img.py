# /content/modules/txt2img.py

import gradio as gr
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
import torch
from modules import pipeline as pipe_module
from modules.pipeline import load_pipeline_global
import random
import sys

def txt2img(prompt_t2i, negative_prompt_t2i, height_t2i, width_t2i, num_inference_steps_t2i, guidance_scale_t2i, batch_size_t2i, seed_int):
    if seed_int == "":
        seed = random.randint(0, sys.maxsize)
    else:
        try:
            seed = int(seed_int)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return None
    torch.manual_seed(seed)
    global pipeline
    if pipe_module.pipeline is None:
        return "Pipeline is not loaded. Please click 'Load Pipeline' first."
    images = pipe_module.pipeline(prompt=prompt_t2i, negative_prompt=negative_prompt_t2i, height=height_t2i, width=width_t2i, num_inference_steps=num_inference_steps_t2i, guidance_scale=guidance_scale_t2i, num_images_per_prompt=batch_size_t2i).images
    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]
    return images_pil, seed

