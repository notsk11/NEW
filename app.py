#@title main.py
import gradio as gr
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler
from modules.txt2img import load_diffusion_pipeline, generate_images
from modules import style
from modules.style import css
scheduler_choices = {
    "Euler": EulerDiscreteScheduler.from_config,
    "Euler a": EulerAncestralDiscreteScheduler.from_config,
    "LMS": LMSDiscreteScheduler.from_config,
    "DPM++ 2M": DPMSolverMultistepScheduler.from_config,
    "DPM++ SDE": DPMSolverSinglestepScheduler.from_config,
    "LMS Karras": lambda config, use_karras_sigmas=True: LMSDiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
    "DPM2 Karras": lambda config, use_karras_sigmas=True: KDPM2DiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
    "DPM++ 2M Karras": lambda config, use_karras_sigmas=True: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
    "DPM++ SDE Karras": lambda config, use_karras_sigmas=True: DPMSolverSinglestepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
    "DPM++ 2M SDE": lambda config, use_karras_sigmas=False: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas, algorithm_type='sde-dpmsolver++'),
    "DPM++ 2M SDE Karras": lambda config, use_karras_sigmas=True: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas, algorithm_type='sde-dpmsolver++'),
}

modelnames = [
    ("digiplay/Realisian_v5", "digiplay/Realisian_v5"),
    ("SG161222/Realistic_Vision_V6.0_B1_noVAE", "SG161222/Realistic_Vision_V6.0_B1_noVAE"),
    ("segmind/SSD-1B", "segmind/SSD-1B"),
    ("digiplay/RealEpicMajicRevolution_v1", "digiplay/RealEpicMajicRevolution_v1"),
    ("imagepipeline/Realities-Edge-XL", "imagepipeline/Realities-Edge-XL"),
    ("stablediffusionapi/realisian111", "stablediffusionapi/realisian111")
]

pipeline = None  # Initialize pipeline globally

def load_model(models_id):
    global pipeline
    pipeline = load_diffusion_pipeline(models_id)
    pipeline.safety_checker = None

image_info_textbox = gr.Textbox(label="Image Info")  # Create the Textbox component
textbox_holder = [image_info_textbox]

def create_callback():
    def generate_images_callback(prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_int, scheduler_choice):
        global pipeline
        images, seed, scheduler = generate_images(pipeline, prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_int, scheduler_choice)
        image_data = {
            "prompt": prompt_str,
            "negative_prompt": neg_prompt_str,
            "seed": seed,
            "scheduler": pipeline.scheduler
        }
        image_info_textbox = update_image_info(textbox_holder[0], image_data)  # Access the textbox from the list
        textbox_holder[0] = image_info_textbox  # Update the textbox in the list
        return images, textbox_holder[0]  # Return the new Textbox
    return generate_images_callback


def load_model_callback(models_id):
    load_model(models_id)

def update_image_info(image_info_textbox, image_data):
    # Convert image_data to string
    image_data_str = str(image_data)
    new_textbox = gr.Textbox(value=image_data_str)  # Create a new Textbox
    return new_textbox


with gr.Blocks(css=style.css) as demo:
  gr.Markdown("Stable Diffusion")
  image_output_block = gr.Gallery(elem_classes="image")
  prompt_str = gr.Textbox(placeholder="Prompt", lines=2, container=False, elem_classes="prompt")
  neg_prompt_str = gr.Textbox(placeholder="Negative Prompt", lines=2, container=False, elem_classes="neg-prompt")
  height_int = gr.Slider(minimum=408, maximum=1600, value=408, step=8, container=False, elem_classes="height")
  gr.Markdown("Height", elem_classes="height-mark")
  width_int = gr.Slider(minimum=408, maximum=1600, value=408, step=8, container=False, elem_classes="width")
  gr.Markdown("Width", elem_classes="width-mark")
  generate = gr.Button("Generate", elem_classes="gen")

  with gr.Tab("Txt2Img", elem_classes="tab"):
    with gr.Group():
      with gr.Column():
         load_model_button = gr.Button("Load Model", elem_classes="load")
      with gr.Row():
          load_model_button.click(load_model_callback, inputs=[gr.Dropdown(choices=modelnames, elem_classes="model", value="stablediffusionapi/realisian111")])
          schedulers = gr.Dropdown(label="Sampling Methods", elem_classes="model", choices=scheduler_choices)
      with gr.Row():
        num_steps_int = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=10, step=1, container=True, elem_classes="model")
        guid_scale_float = gr.Slider(label="CFG Scale", minimum=1, maximum=10, value=5, step=0.1, container=True, elem_classes="model")
      with gr.Row():
        num_images_int = gr.Slider(label="Batch Count", minimum=1, maximum=10, value=1, step=1, container=True, elem_classes="model")
      with gr.Column():
        seed_inp = gr.Textbox(label="Seed", elem_classes="model")
      with gr.Accordion(label="Details", open=False):
        image_info_textbox = gr.Textbox(show_label=False, placeholder="Info about generated image", elem_classes="details", lines=10)
  with gr.Tab("Img2Img", elem_classes="tab"):
    num_images = gr.Slider(minimum=1, maximum=1, value=1, step=1, container=False)
  with gr.Tab("Img2Vid", elem_classes="tab"):
    num_images = gr.Slider(minimum=1, maximum=1, value=1, step=1, container=False)
  with gr.Tab("UpScale", elem_classes="tab"):
    num_images = gr.Slider(minimum=1, maximum=1, value=1, step=1, container=False)

  callback = create_callback()  # Create the callback
  generate.click(fn=callback, inputs=[prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_inp, schedulers], outputs=[image_output_block, image_info_textbox])

demo.launch(share=True, debug=True)
