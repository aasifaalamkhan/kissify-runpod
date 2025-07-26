import os
import torch
import uuid
import gc
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel

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

# --- WORKAROUND: Load only ONE IP-Adapter ---
print("[INFO] WORKAROUND: Loading only ONE IP-Adapter.", flush=True)
pipe.load_ip_adapter(ip_adapter_repo_id, subfolder="models", weight_name="ip-adapter_sd15.bin")

# --- Pre-process and cache the pose sequence for ControlNet ---
POSE_SEQUENCE = load_pose_sequence(CACHED_POSE_PATH)
if POSE_SEQUENCE is None:
    POSE_SEQUENCE = extract_pose_sequence(MOTION_TEMPLATE_PATH)
    save_pose_sequence(POSE_SEQUENCE, CACHED_POSE_PATH)
NUM_FRAMES = len(POSE_SEQUENCE)
print("✅ All models and pose data are ready.", flush=True)


# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video based on two input face images.
    Uses ControlNet for motion and a composite image for face identity.
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

        print("🧠 Step 1/5: Loading images...", flush=True)
        face_images_b64 = [input_data['face_image1'], input_data['face_image2']]
        pil_images = load_face_images(face_images_b64)

        print("👤 Step 2/5: Cropping faces...", flush=True)
        face1_cropped = crop_face(pil_images[0]).resize((224, 224))
        face2_cropped = crop_face(pil_images[1]).resize((224, 224))

        print("🔍 Step 3/5: Creating composite image...", flush=True)
        composite_image = Image.new('RGB', (448, 224))
        composite_image.paste(face1_cropped, (0, 0))
        composite_image.paste(face2_cropped, (224, 0))

        prompt = "photo of a man and a woman kissing, faces of the people from the reference image, best quality, realistic, masterpiece, high resolution"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, two heads, multiple people, deformed"

        print(f"🎨 Step 4/5: Generating animation ({NUM_FRAMES} frames)...", flush=True)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=POSE_SEQUENCE, # Using ControlNet with the pose sequence
                controlnet_conditioning_scale=0.8,
                ip_adapter_image=composite_image,
                ip_adapter_scale=1.8,
                num_frames=NUM_FRAMES,
                guidance_scale=7.0,
                num_inference_steps=50,
            ).frames[0]
        video_frames = output

        print("🚀 Step 5/5: Post-processing (upscale & smooth)...", flush=True)
        export_video_with_imageio(video_frames, raw_video_path, fps=8)
        upscale_video(raw_video_path, upscaled_video_path)
        smooth_video(upscaled_video_path, final_video_path, target_fps=48)

        return {"filename": final_filename}

    finally:
        print("🧹 Cleaning up...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        if raw_video_path and os.path.exists(raw_video_path):
            os.remove(raw_video_path)
        if upscaled_video_path and os.path.exists(upscaled_video_path):
            os.remove(upscaled_video_path)
