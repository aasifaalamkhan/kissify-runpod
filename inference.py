import os
import torch
import uuid
import gc
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel, StableDiffusionXLAdapterPipeline, IPAdapter
from utils import (
    load_face_images, crop_face,
    export_video_with_imageio, upscale_video, smooth_video,
    extract_pose_sequence,
    save_pose_sequence, load_pose_sequence
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"
MOTION_TEMPLATE_PATH = "/workspace/kissify-runpod/motion_template.mp4"
CACHED_POSE_PATH = "/workspace/kissify-runpod/cached_pose_sequence.npy"


# --- Load Models ---
print("[INFO] Initializing models and pipeline...", flush=True)
device = "cuda"

controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16).to(device)


base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter" # The base repo for IP-Adapter

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float16
).to(device).eval()
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
motion_adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16).to(device)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(device)
pipe.scheduler = DDIMScheduler(beta_schedule="linear", num_train_timesteps=1000)

# ==============================================================================
# 🚨🚨🚨 THE DEFINITIVE FIX FOR loading multiple IP-Adapters in diffusers 0.29.2 🚨🚨🚨
# Use `pipe.load_ip_adapter` to load the *first* one, and `pipe.add_ip_adapter` for subsequent ones.
# Or, more cleanly:
# Initialize IPAdapter instances and then load them.
# The correct way to load multiple IP-Adapters for `AnimateDiffPipeline`
# (or generally StableDiffusionPipeline in diffusers 0.29.2) is via a list
# to `load_ip_adapter`, but the format might be specific.
# Given the `TypeError`, the previous `ip_adapter_config` list format was wrong.
# Let's try the *most explicit* way by loading paths individually and using `set_ip_adapter`.
# ==============================================================================

# First, load the IP-Adapter weights. This typically creates a *single* IP-Adapter module.
# The `load_ip_adapter` method *modifies the pipeline in-place* to include the adapter.
# If you call it multiple times with the same settings, it might just re-configure the *same* adapter.

# The documentation (and observed behavior in some versions) suggests that for multiple IP-Adapters,
# you might need to manually set them up if the direct `load_ip_adapter` is problematic.
# Let's explicitly define the two IP-Adapter components.

# Re-checking diffusers 0.29.2 source for AnimateDiffPipeline and IPAdapter integration:
# AnimateDiffPipeline inherits from StableDiffusionPipeline.
# StableDiffusionPipeline.load_ip_adapter() (which is IPAdapterMixin.load_ip_adapter)
# expects `(pretrained_model_name_or_path, subfolder=None, weight_name=None, ...)`
# and if you pass a list, it expects a list of *paths to models or tuples of paths*.

# Given the TypeError `missing 'subfolder' and 'weight_name'`,
# it clearly means that the `ip_adapter_config` list format was misunderstood.
# The correct way for multiple in diffusers 0.29.2 is to pass a LIST of IP-Adapter objects
# or load them sequentially and verify.

# Let's try calling `load_ip_adapter` for the first IP-Adapter,
# and then for the second one, we'll try to explicitly use the list syntax
# if `load_ip_adapter` supports it in a different way or use `set_ip_adapter`.

# The original method of calling `pipe.load_ip_adapter` twice *should* work if it registers.
# The `TypeError` comes from passing a list of (repo, subfolder, weight_name) which `load_ip_adapter` doesn't directly take.
# It expects a single repo_id/path.

# Okay, the error "missing 2 required positional arguments: 'subfolder' and 'weight_name'"
# means `pipe.load_ip_adapter(ip_adapter_config)` where `ip_adapter_config` is `[(...), (...)]` is wrong.
# `load_ip_adapter` when called with a list, expects a list of *model paths* or a list of *IPAdapter* instances, not tuples.

# Let's go back to calling it twice, but add a print to inspect `pipe.ip_adapter_image_proj` which stores them.
# This will clarify if the second call is overriding.

# First call:
pipe.load_ip_adapter(
    ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin"
)
print(f"[DEBUG] After 1st load, pipe.ip_adapter_image_proj (type: {type(pipe.ip_adapter_image_proj)}): {len(pipe.ip_adapter_image_proj) if isinstance(pipe.ip_adapter_image_proj, list) else 1 if pipe.ip_adapter_image_proj else 0}", flush=True)

# Second call: THIS IS WHERE THE PROBLEM LIES IF IT OVERWRITES
pipe.load_ip_adapter(
    ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin"
)
print(f"[DEBUG] After 2nd load, pipe.ip_adapter_image_proj (type: {type(pipe.ip_adapter_image_proj)}): {len(pipe.ip_adapter_image_proj) if isinstance(pipe.ip_adapter_image_proj, list) else 1 if pipe.ip_adapter_image_proj else 0}", flush=True)


# Final verification after both loads
num_loaded_ip_adapters = 0
if hasattr(pipe, 'ip_adapter') and isinstance(pipe.ip_adapter, list):
    num_loaded_ip_adapters = len(pipe.ip_adapter)
elif hasattr(pipe, 'ip_adapter') and pipe.ip_adapter is not None:
    num_loaded_ip_adapters = 1 # Single IPAdapter object
print(f"✅ [INFO] Pipeline reports {num_loaded_ip_adapters} IP-Adapters after all loads.", flush=True)


# The pipe call will be the same
print("[INFO] Models and pipeline are initialized.", flush=True)


# --- Pre-process and cache the pose sequence at startup ---
POSE_SEQUENCE = load_pose_sequence(CACHED_POSE_PATH)
if POSE_SEQUENCE is None:
    POSE_SEQUENCE = extract_pose_sequence(MOTION_TEMPLATE_PATH)
    save_pose_sequence(POSE_SEQUENCE, CACHED_POSE_PATH)
NUM_FRAMES = len(POSE_SEQUENCE)


# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video based on two input face images.
    """
    raw_video_path = None
    upscaled_video_path = None

    try:
        unique_id = str(uuid.uuid4())
        raw_filename = f"{unique_id}_raw.mp4"
        upscaled_filename = f"{unique_id}_upscaled.mp4"
        final_filename = f"{unique_id}_final.mp4"

        raw_video_path = os.path.join(OUTPUT_DIR, raw_filename)
        upscaled_video_path = os.path.join(OUTPUT_DIR, upscaled_filename)
        final_video_path = os.path.join(OUTPUT_DIR, final_filename)

        print("🧠 Step 1/5: Loading images from b64...", flush=True)
        face_images_b64 = [input_data['face_image1'], input_data['face_image2']]
        pil_images = load_face_images(face_images_b64)

        print("👤 Step 2/5: Detecting and cropping faces...", flush=True)
        face1_cropped = crop_face(pil_images[0])
        face2_cropped = crop_face(pil_images[1])

        print("🔍 Step 3/5: Preparing faces for IP-Adapter...", flush=True)
        ip_adapter_images_for_pipeline = [face1_cropped, face2_cropped]

        prompt = "a man and a woman kissing, best quality, realistic, masterpiece, high resolution"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo"

        print(f"🎨 Step 4/5: Generating ControlNet-guided animation ({NUM_FRAMES} frames)...", flush=True)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=POSE_SEQUENCE,
                controlnet_conditioning_scale=0.8,
                ip_adapter_image=ip_adapter_images_for_pipeline,
                ip_adapter_scale=1.8,
                num_frames=NUM_FRAMES,
                guidance_scale=5.0,
                num_inference_steps=50,
            ).frames[0]
        video_frames = output

        print("🚀 Step 5/5: Post-processing video (export, upscale, smooth)...", flush=True)
        export_video_with_imageio(video_frames, raw_video_path, fps=8)
        upscale_video(raw_video_path, upscaled_video_path)
        smooth_video(upscaled_video_path, final_video_path, target_fps=48)

        return {"filename": final_filename}

    finally:
        print("🧹 Cleaning up GPU memory and temporary files...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        if raw_video_path and os.path.exists(raw_video_path):
            os.remove(raw_video_path)
            print(f"Removed temporary file: {raw_video_path}", flush=True)
        if upscaled_video_path and os.path.exists(upscaled_video_path):
            os.remove(upscaled_video_path)
            print(f"Removed temporary file: {upscaled_video_path}", flush=True)
