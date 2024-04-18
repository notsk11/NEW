# /content/modules/pipeline.py

from diffusers import DiffusionPipeline
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

pipeline = None
def load_pipeline(model_id):
  global pipeline
  pipeline = DiffusionPipeline.from_pretrained(model_id).to('cuda')
  pipeline.safety_checker = None
  return pipeline

def update_scheduler(scheduler):
    if scheduler == "PNDM":
        input_scheduler = PNDMScheduler.from_pretrained("notsk007/PNDM")
    elif scheduler == "DEIS":
        input_scheduler = DEISMultistepScheduler.from_pretrained("notsk007/DEIS")
    elif scheduler == "UniPC":
        input_scheduler = UniPCMultistepScheduler.from_pretrained("notsk007/UniPC")
    elif scheduler == "Euler":
        input_scheduler = EulerDiscreteScheduler.from_pretrained("notsk007/Euler")
    elif scheduler == "Euler-A":
        input_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("notsk007/Euler-A")
    elif scheduler == "LMS":
        input_scheduler = LMSDiscreteScheduler.from_pretrained("notsk007/LMS")
    elif scheduler == "LMS-Karras":
        input_scheduler = LMSDiscreteScheduler.from_pretrained("notsk007/LMS-Karras")
    elif scheduler == "DPM2":
        input_scheduler = KDPM2DiscreteScheduler.from_pretrained("notsk007/DPM2")
    elif scheduler == "DPM2-Karras":
        input_scheduler = KDPM2DiscreteScheduler.from_pretrained("notsk007/DPM2-Karras")
    elif scheduler == "DPM2-A":
        input_scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained("notsk007/DPM2-A")
    elif scheduler == "DPM2-A-Karras":
        input_scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained("notsk007/DPM2-A-Karras")
    elif scheduler == "DPM-SDE":
        input_scheduler = DPMSolverSinglestepScheduler.from_pretrained("notsk007/DPM-SDE")
    elif scheduler == "DPM-SDE-Karras":
        input_scheduler = DPMSolverSinglestepScheduler.from_pretrained("notsk007/DPM-SDE-Karras")
    elif scheduler == "DPM-2M":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M")
    elif scheduler == "DPM-2M-Karras":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M-Karras")
    elif scheduler == "DPM-2M-SDE":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M-SDE")
    elif scheduler == "DPM-2M-SDE-Karras":
        input_scheduler = DPMSolverMultistepScheduler.from_pretrained("notsk007/DPM-2M-SDE-Karras")
    else:
        print("Invalid scheduler selection.")
        return

    pipeline.scheduler = input_scheduler
    print(f"Scheduler updated to: {scheduler}")
    return pipeline.scheduler
