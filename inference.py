import os
import torch
import uuid
import gc
import time  # Import the time module
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
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
pipe.scheduler = DDIMScheduler(beta_schedule="linear", num_train_timesteps=1000)

print("[INFO] WORKAROUND: Loading only ONE IP-Adapter.", flush=True)
pipe.load_ip_adapter(ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin")

POSE_SEQUENCE = load_pose_sequence(CACHED_POSE_PATH)
if POSE_SEQUENCE is None:
    POSE_SEQUENCE = extract_pose_sequence(MOTION_TEMPLATE_PATH)
    save_pose_sequence(POSE_SEQUENCE, CACHED_POSE_PATH)
print("✅ All models and pose data are ready.", flush=True)


# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video based on two input face images.
    Uses a sliding window approach and now times the final export step.
    """
    raw_video_path = None
    upscaled_video_path = None

    try:
        unique_id = str(uuid.uuid4())
        raw_filename = f"{unique_id}_raw.mp4"
        final_filename = f"{unique_id}_final.mp4"

        raw_video_path = os.path.join(OUTPUT_DIR, raw_filename)
        final_video_path = os.path.join(OUTPUT_DIR, final_filename)

        print("🧠 Step 1/5: Loading and preparing images...", flush=True)
        face_images_b64 = [input_data['face_image1'], input_data['face_image2']]
        pil_images = load_face_images(face_images_b64)
        face1_cropped = crop_face(pil_images[0]).resize((224, 224))
        face2_cropped = crop_face(pil_images[1]).resize((224, 224))

        composite_image = Image.new('RGB', (448, 224))
        composite_image.paste(face1_cropped, (0, 0))
        composite_image.paste(face2_cropped, (224, 0))

        prompt = "photo of a man and a woman kissing, faces of the people from the reference image, best quality, realistic, masterpiece, high resolution"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, two heads, multiple people, deformed"
        
        # --- Sliding Window Parameters ---
        window_size = 32
        stride = 16
        total_frames = len(POSE_SEQUENCE)
        all_frames = []

        print(f"🎨 Step 2/5: Starting sliding window generation for {total_frames} frames...", flush=True)
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

                print(f"  -> Generating chunk for frames {start_index} to {end_index-1}...", flush=True)
                output_chunk = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=chunk_poses,
                    controlnet_conditioning_scale=0.8,
                    ip_adapter_image=composite_image,
                    ip_adapter_scale=1.8,
                    num_frames=window_size,
                    guidance_scale=7.0,
                    num_inference_steps=50,
                ).frames[0]

                if i == 0:
                    all_frames.extend(output_chunk)
                else:
                    all_frames.extend(output_chunk[-stride:])
                
                if end_index >= total_frames:
                    break

        video_frames = all_frames[:total_frames]
        print(f"✅ Step 3/5: Finished generation. Total frames: {len(video_frames)}", flush=True)

        print("🚀 Step 4/5: Post-processing (export & smooth)...", flush=True)
        # --- Start timing the export process ---
        export_start_time = time.time()

        export_video_with_imageio(video_frames, raw_video_path, fps=8)
        
        # Upscaling is currently skipped for faster testing
        # upscale_video(raw_video_path, upscaled_video_path)
        
        smooth_video(raw_video_path, final_video_path, target_fps=48)
        
        # --- End timing and print the duration ---
        export_end_time = time.time()
        export_duration = export_end_time - export_start_time
        print(f"✅ Post-processing finished in {export_duration:.2f} seconds.")

        print("✅ Step 5/5: Done!", flush=True)
        return {"filename": final_filename}

    finally:
        print("🧹 Cleaning up...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if raw_video_path and os.path.exists(raw_video_path):
            os.remove(raw_video_path)

