import os
import torch
import uuid
import gc
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from utils import (
    load_face_images, crop_face,
    export_video_with_imageio, upscale_video, smooth_video,
    extract_pose_sequence,
    save_pose_sequence, load_pose_sequence # Keep these imports for caching
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"
MOTION_TEMPLATE_PATH = "/workspace/kissify-runpod/motion_template.mp4"
CACHED_POSE_PATH = "/workspace/kissify-runpod/cached_pose_sequence.npy" # Path for cached pose sequence


# --- Load Models ---
print("[INFO] Initializing models and pipeline...", flush=True)
device = "cuda"

# --- Use the correct, standard ControlNet model ID ---
controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16).to(device)


base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float16
).to(device).eval()
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
motion_adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16).to(device)

# --- Inject ControlNet into the pipeline ---
pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(device)
pipe.scheduler = DDIMScheduler(beta_schedule="linear", num_train_timesteps=1000) # Use DDIMScheduler with default params if not specified in base model

# ==============================================================================
# 🚨🚨🚨 CRITICAL FIX STARTS HERE 🚨🚨🚨
# LOAD IP-ADAPTER TWICE TO HAVE TWO SEPARATE IP-ADAPTERS IN THE PIPELINE
# Each call to load_ip_adapter *adds* an adapter.
# ==============================================================================
pipe.load_ip_adapter(
    ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin"
)
pipe.load_ip_adapter( # <--- THIS IS THE SECOND CALL TO LOAD THE SECOND IP-ADAPTER
    ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin"
)
# ==============================================================================
# 🚨🚨🚨 CRITICAL FIX ENDS HERE 🚨🚨🚨
# ==============================================================================

print("[INFO] Models and pipeline are initialized.", flush=True)

# --- Pre-process and cache the pose sequence at startup ---
POSE_SEQUENCE = load_pose_sequence(CACHED_POSE_PATH)
if POSE_SEQUENCE is None:
    POSE_SEQUENCE = extract_pose_sequence(MOTION_TEMPLATE_PATH)
    save_pose_sequence(POSE_SEQUENCE, CACHED_POSE_PATH) # Save after extraction
NUM_FRAMES = len(POSE_SEQUENCE)


# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video based on two input face images.
    """
    raw_video_path = None
    upscaled_video_path = None

    try:
        # --- Define unique filenames and paths for this request ---
        unique_id = str(uuid.uuid4())
        raw_filename = f"{unique_id}_raw.mp4"
        upscaled_filename = f"{unique_id}_upscaled.mp4"
        final_filename = f"{unique_id}_final.mp4"

        raw_video_path = os.path.join(OUTPUT_DIR, raw_filename)
        upscaled_video_path = os.path.join(OUTPUT_DIR, upscaled_filename)
        final_video_path = os.path.join(OUTPUT_DIR, final_filename)

        # --- Step 1: Loading images ---
        print("🧠 Step 1/5: Loading images from b64...", flush=True)
        face_images_b64 = [input_data['face_image1'], input_data['face_image2']]
        pil_images = load_face_images(face_images_b64)

        # --- Step 2: Detecting and cropping faces ---
        print("👤 Step 2/5: Detecting and cropping faces...", flush=True)
        face1_cropped = crop_face(pil_images[0])
        face2_cropped = crop_face(pil_images[1])

        # --- Step 3: Preparing faces for IP-Adapter (passing two distinct images) ---
        print("🔍 Step 3/5: Preparing faces for IP-Adapter...", flush=True)
        # We now have two IP-Adapters loaded in the pipe.
        # So we can pass the two cropped PIL images directly as a list.
        # The pipeline will use the first image for the first IP-Adapter,
        # and the second image for the second IP-Adapter.
        ip_adapter_images_for_pipeline = [face1_cropped, face2_cropped]


        prompt = "a man and a woman kissing, best quality, realistic, masterpiece, high resolution"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo"

        # --- Step 4: Generating ControlNet-guided animation ---
        print(f"🎨 Step 4/5: Generating ControlNet-guided animation ({NUM_FRAMES} frames)...", flush=True)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=POSE_SEQUENCE,
                controlnet_conditioning_scale=0.8,
                # ==============================================================================
                # 🚨🚨🚨 CRITICAL FIX: Ensure `ip_adapter_image` is used with the list of PIL images 🚨🚨🚨
                # ==============================================================================
                ip_adapter_image=ip_adapter_images_for_pipeline,
                # You can specify a list of scales if you want different weights for each IP-Adapter:
                # ip_adapter_scale=[1.8, 1.8], # Example: same scale for both
                ip_adapter_scale=1.8, # This applies 1.8 to both loaded IP-Adapters
                num_frames=NUM_FRAMES,
                guidance_scale=5.0,
                num_inference_steps=50,
            ).frames[0]
        video_frames = output

        # --- Step 5: Post-Processing Workflow ---
        print("🚀 Step 5/5: Post-processing video (export, upscale, smooth)...", flush=True)
        export_video_with_imageio(video_frames, raw_video_path, fps=8)
        upscale_video(raw_video_path, upscaled_video_path)
        smooth_video(upscaled_video_path, final_video_path, target_fps=48)

        return {"filename": final_filename}

    finally:
        # --- Cleanup: Free GPU memory and temporary files ---
        print("🧹 Cleaning up GPU memory and temporary files...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        if raw_video_path and os.path.exists(raw_video_path):
            os.remove(raw_video_path)
            print(f"Removed temporary file: {raw_video_path}", flush=True)
        if upscaled_video_path and os.path.exists(upscaled_video_path):
            os.remove(upscaled_video_path)
            print(f"Removed temporary file: {upscaled_video_path}", flush=True)
