# /content/modules/pipeline.py
from diffusers import DiffusionPipeline

pipeline = None
scheduler = None  # Add this line to define the scheduler attribute
def load_pipeline_global(model_id):
  global pipeline
  pipeline = DiffusionPipeline.from_pretrained(model_id).to('cuda')
  pipeline.safety_checker = None
  return pipeline
