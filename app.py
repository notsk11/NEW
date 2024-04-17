# /content/app.py

import gradio as gr
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
import torch
from modules import txt2img
from modules.txt2img import txt2img
from modules import pipeline
from modules.pipeline import load_pipeline_global
from modules import style
from modules.style import css

from diffusers import (
    PNDMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
)

# Define a dictionary to map scheduler names to their constructors
scheduler_constructors = {
    "PNDM": PNDMScheduler.from_pretrained,
    "DEIS": DEISMultistepScheduler.from_pretrained,
    "UniPC": UniPCMultistepScheduler.from_pretrained,
    "Euler": EulerDiscreteScheduler.from_pretrained,
    "Euler A": EulerAncestralDiscreteScheduler.from_pretrained,
    "LMS": LMSDiscreteScheduler.from_pretrained,
    "LMS-Karras": LMSDiscreteScheduler.from_pretrained,
    "DPM2": KDPM2DiscreteScheduler.from_pretrained,
    "DPM2-Karras": KDPM2DiscreteScheduler.from_pretrained,
    "DPM2-A": KDPM2AncestralDiscreteScheduler.from_pretrained,
    "DPM2-A-Karras": KDPM2AncestralDiscreteScheduler.from_pretrained,
    "DPM-SDE": DPMSolverSinglestepScheduler.from_pretrained,
    "DPM-SDE-Karras": DPMSolverSinglestepScheduler.from_pretrained,
    "DPM-2M": DPMSolverMultistepScheduler.from_pretrained,
    "DPM-2M-Karras": DPMSolverMultistepScheduler.from_pretrained,
    "DPM-2M-SDE": DPMSolverMultistepScheduler.from_pretrained,
    "DPM-2M-SDE-Karras": DPMSolverMultistepScheduler.from_pretrained,
}


def update_scheduler(scheduler_name):
    if scheduler_name in scheduler_constructors:
        # Get the constructor function for the selected scheduler
        constructor = scheduler_constructors[scheduler_name]
        # Load the scheduler using the constructor
        pipeline.scheduler = constructor("notsk007/" + scheduler_name)
        return f"Successfully updated scheduler to {scheduler_name}. Scheduler value: {pipeline.scheduler}"
    else:
        return f"Scheduler '{scheduler_name}' not found."




with gr.Blocks(css=style.css) as demo:
  with gr.Tab("Txt2Img", elem_classes="tab-t2i"):
    with gr.Group():
      with gr.Column():
        with gr.Column():
          prompt_t2i = gr.Textbox(elem_classes="prompt-t2i", lines=3, container=False, placeholder="Place Prompts here....")
          negative_prompt_t2i = gr.Textbox(elem_classes="negative-prompt-t2i", lines=3, container=False, placeholder="Place Negative Prompts here....")
        with gr.Row():
          seed_inp = gr.Textbox(elem_classes="seed-input-t2i", lines=1, container=False, placeholder="Place Seed here....")
        with gr.Row():
          height_t2i = gr.Slider(elem_classes="height-t2i", label="Height", minimum=408, maximum=1600, value=408, step=8)
          width_t2i = gr.Slider(elem_classes="width-t2i", label="Width", minimum=408, maximum=1600, value=408, step=8)
        with gr.Row():
          num_inference_steps_t2i = gr.Slider(elem_classes="num-steps-t2i", label="Sampling Steps", minimum=1, maximum=100, value=20, step=1, scale=8)
          guidance_scale_t2i = gr.Slider(elem_classes="guidance-scale-t2i", label="CFG Scale", minimum=0, maximum=10, value=7.5, step=0.1, scale=5)
        with gr.Row():
          batch_size_t2i = gr.Slider(elem_classes="batch-size-t2i", label="Batch Size", minimum=1, maximum=10, value=1, step=1)
      with gr.Column():
        with gr.Row():
          model_global = gr.Textbox(elem_classes="model_global", label="Model", value="SG161222/Realistic_Vision_V6.0_B1_noVAE", scale=5)
          load_model_global = gr.Button("Load Model", elem_classes="load-model-gloabl", scale=1)
      with gr.Column():
        with gr.Row():
          scheduler = gr.Dropdown(elem_classes="scheduler", choices=list(scheduler_constructors.keys()), label="Sampling Method", value="DPM-SDE-Karras", scale=5)
          load_scheduler = gr.Button("Load Scheduler", scale=1)
        with gr.Column():
          generate_t2i = gr.Button("Generate", elem_classes="generate-t2i", scale=3)
    image_out_t2i = gr.Gallery(elem_classes="image-out-t2i")
    metadata_t2i = gr.Textbox(elem_classes="metadata-t2i", lines=5, container=False, placeholder="Data about generated image here....")
  with gr.Tab("Img2Img", elem_classes="tab-t2i"):
    lol = gr.Textbox()
    load_model_global.click(fn=load_pipeline_global, inputs=[model_global])
    load_scheduler.click(fn=update_scheduler, inputs=[scheduler])
    generate_t2i.click(fn=txt2img, inputs=[prompt_t2i, negative_prompt_t2i, height_t2i, width_t2i, num_inference_steps_t2i, guidance_scale_t2i, batch_size_t2i, seed_inp], outputs=[image_out_t2i, metadata_t2i])
demo.launch(share=True, debug=True)
