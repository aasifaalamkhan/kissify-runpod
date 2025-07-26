import os
import torch
import uuid
import gc
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel # New
# Import the new pose extraction function
from utils import (
    load_face_images, crop_face, prepare_ip_adapter_inputs,
    export_video_with_imageio, upscale_video, smooth_video,
    extract_pose_sequence
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"
MOTION_TEMPLATE_PATH = "/workspace/motion_template.mp4"

# --- Load Models ---
print("[INFO] Initializing models and pipeline...", flush=True)
device = "cuda"

# --- NEW: Load ControlNet OpenPose Model ---
controlnet_model_id = "thibaud/controlnet-openpose-sd-v1-1"
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float32).to(device)

base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float32
).to(device).eval()
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
motion_adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float32).to(device)

# --- NEW: Inject ControlNet into the pipeline ---
pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet, # Pass the loaded controlnet
    torch_dtype=torch.float32,
).to(device)
pipe.scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
pipe.load_ip_adapter(
    ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin"
)
print("[INFO] Models and pipeline are initialized.", flush=True)

# --- NEW: Pre-process and cache the pose sequence at startup ---
POSE_SEQUENCE = extract_pose_sequence(MOTION_TEMPLATE_PATH)
NUM_FRAMES = len(POSE_SEQUENCE)


# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    # ... (Variable and path definitions are mostly unchanged) ...
    # ...
    try:
        # ... (Steps 1-3: Loading, cropping, and encoding faces are unchanged) ...
        print("🧠 Step 1/5: Loading images...", flush=True)
        # ...

        print("👤 Step 2/5: Detecting and cropping faces...", flush=True)
        # ...

        print("🔍 Step 3/5: Preparing and encoding faces with IP-Adapter...", flush=True)
        # ...

        prompt = "..."
        negative_prompt = "..."
        pipe.set_ip_adapter_scale(1.8)

        print(f"🎨 Step 4/5: Generating ControlNet-guided animation...", flush=True)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=POSE_SEQUENCE, # Pass the pose sequence to the 'image' param for ControlNet
                controlnet_conditioning_scale=0.8, # Set ControlNet strength
                ip_adapter_image_embeds=[ip_embeds],
                num_frames=NUM_FRAMES, # Number of frames must match pose sequence length
                guidance_scale=5.0,
                num_inference_steps=50,
            ).frames[0]
        video_frames = output

        # --- Step 5: Post-Processing Workflow (Unchanged) ---
        print("🚀 Step 5/5: Post-processing video (export, upscale, smooth)...", flush=True)
        # ...
        
        return {"filename": final_filename}

    finally:
        # --- Cleanup (Unchanged) ---
        print("🧹 Cleaning up GPU memory and temporary files...", flush=True)
        # ...
