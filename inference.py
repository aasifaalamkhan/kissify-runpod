import os
import torch
import uuid
import gc
import time
from PIL import Image
# Import the new DPM++ scheduler
from diffusers import AnimateDiffPipeline, MotionAdapter, ControlNetModel, DPMSolverMultistepScheduler
import numpy as np

from utils import (
    load_face_images, crop_face,
    export_video_with_imageio, upscale_video, smooth_video,
    extract_pose_sequence, save_pose_sequence, load_pose_sequence
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"
MOTION_TEMPLATE_PATH = "/workspace/kissify-runpod/motion_template.mp4"
CACHED_POSE_PATH = "/workspace/kissify-runpod/cached_pose_sequence.npy"


# --- Load Models ---
print("[INFO] Initializing all models (including ControlNet)...", flush=True)
device = "cuda"

controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16).to(device)

base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"

motion_adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16).to(device)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(device)

# FIX: Add the required 'final_sigmas_type' parameter to resolve the error.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, 
    use_karras_sigmas=True,
    final_sigmas_type='sigma_min' 
)


print("[INFO] Loading ONE IP-Adapter.", flush=True)
pipe.load_ip_adapter(ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin")

# Load and combine BOTH Kissing LoRAs
print("[INFO] Loading and combining two Kissing LoRAs...", flush=True)
pipe.load_lora_weights("Remade-AI/kissing", weight_name="Kissing.safetensors", adapter_name="style")
pipe.load_lora_weights("ighoshsubho/Wan-I2V-LoRA-Kiss", weight_name="wan-i2v-lora-kiss.safetensors", adapter_name="motion")

pipe.set_adapters(["style", "motion"], adapter_weights=[0.6, 0.6])

pipe.fuse_lora()


POSE_SEQUENCE = load_pose_sequence(CACHED_POSE_PATH)
if POSE_SEQUENCE is None:
    POSE_SEQUENCE = extract_pose_sequence(MOTION_TEMPLATE_PATH)
    save_pose_sequence(POSE_SEQUENCE, CACHED_POSE_PATH)
print("✅ All models and pose data are ready.", flush=True)


# ========= Video Generation Logic (MODIFIED TO BE A GENERATOR) =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video.
    NOW YIELDS its progress logs and returns the final filename.
    """
    try:
        unique_id = str(uuid.uuid4())
        final_filename = f"{unique_id}_final.mp4"
        final_video_path = os.path.join(OUTPUT_DIR, final_filename)

        yield "🧠 Step 1/5: Loading and preparing images..."
        face_images_b64 = [input_data['face_image1'], input_data['face_image2']]
        pil_images = load_face_images(face_images_b64)
        
        face1_cropped = crop_face(pil_images[0]).resize((224, 224))
        face2_cropped = crop_face(pil_images[1]).resize((224, 224))
        composite_image = Image.new('RGB', (448, 224))
        composite_image.paste(face1_cropped, (0, 0))
        composite_image.paste(face2_cropped, (224, 0))
        
        composite_filename = f"{unique_id}_composite.jpg"
        composite_image_path = os.path.join(OUTPUT_DIR, composite_filename)
        composite_image.save(composite_image_path)
        yield {'composite_filename': composite_filename}


        prompt = "photo of a man and a woman kissing, faces of the people from the reference image, best quality, realistic, masterpiece, high resolution"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, two heads, multiple people, deformed"

        # --- Sliding Window & Speed Parameters ---
        window_size = 32
        stride = 16
        generation_steps = 25
        total_frames = len(POSE_SEQUENCE)
        all_frames = []
        
        yield f"🎨 Step 2/5: Starting sliding window generation for {total_frames} frames ({generation_steps} steps/chunk)..."
        with torch.inference_mode():
            for i in range(0, total_frames - window_size + stride, stride):
                start_index = i
                end_index = i + window_size

                if end_index > total_frames:
                    start_index = max(0, total_frames - window_size)
                    end_index = total_frames

                chunk_poses = POSE_SEQUENCE[start_index:end_index]

                if len(chunk_poses) < window_size:
                    padding_needed = window_size - len(chunk_poses)
                    chunk_poses.extend([chunk_poses[-1]] * padding_needed)

                yield f"  -> Generating chunk for frames {start_index} to {end_index-1}..."
                output_chunk = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=chunk_poses,
                    controlnet_conditioning_scale=0.8,
                    ip_adapter_image=composite_image,
                    ip_adapter_scale=0.7,
                    num_frames=window_size,
                    guidance_scale=7.0,
                    num_inference_steps=generation_steps,
                ).frames[0]

                if i == 0:
                    all_frames.extend(output_chunk)
                else:
                    all_frames.extend(output_chunk[-stride:])

                if end_index >= total_frames:
                    break

        video_frames = all_frames[:total_frames]
        yield f"✅ Step 3/5: Finished generation. Total frames: {len(video_frames)}"

        yield "🚀 Step 4/5: Post-processing (exporting video)..."
        
        export_video_with_imageio(video_frames, final_video_path, fps=8)
        
        yield "✅ Post-processing finished."
        yield "✅ Step 5/5: Done!"
        
        yield {"filename": final_filename}

    finally:
        yield "🧹 Cleaning up..."
        gc.collect()
        torch.cuda.empty_cache()