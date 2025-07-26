import os
import torch
import uuid # For generating unique filenames
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from utils import load_face_images, export_video_with_imageio

# Define the directory where videos will be stored
OUTPUT_DIR = "/workspace/outputs"


# ========= Load Models =========
print("[INFO] Initializing models and pipeline...", flush=True)

base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"
device = "cuda"

# We no longer need to load the image encoder separately, the pipeline handles it.
motion_adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16).to(device)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
pipe.enable_model_cpu_offload() # Re-enable CPU offload to save VRAM

# Load the IP-Adapter
pipe.load_ip_adapter(
    ip_adapter_repo_id,
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)

print("[INFO] Models and pipeline are initialized.", flush=True)

# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    # --- Simplified logic: Pass PIL images directly to the pipeline ---
    print("🧠 Loading face images...", flush=True)
    pil_images = load_face_images([
        input_data['face_image1'],
        input_data['face_image2']
    ])

    # --- New, more descriptive default prompt ---
    prompt = (input_data.get("prompt") or "").strip()
    if not prompt:
        prompt = "photo of a man and a woman in a passionate, romantic kiss, closeup, cinematic lighting, high detail, 4k"
    
    # --- New, more specific negative prompt ---
    negative_prompt = "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), deformed, ugly, disfigured, weird faces, open mouths, not kissing, two men, two women, blurry, duplicate"

    # --- Adjust IP-Adapter scale for stronger face influence ---
    pipe.set_ip_adapter_scale(1.2)

    print(f"🎨 Generating animation with prompt: '{prompt}'", flush=True)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=pil_images, # Pass the list of PIL images directly
            num_frames=16,
            guidance_scale=7.5, # Slightly increased for better prompt adherence
            num_inference_steps=30, # Increased for more detail
        ).frames[0]

    video_frames = result
    print("💾 Exporting video to local storage as MP4 using imageio...", flush=True)
    
    filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    export_video_with_imageio(video_frames, output_path, fps=8)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("MP4 export failed: The output file is missing or empty.")
    
    print(f"✅ MP4 exported successfully to {output_path}. Size: {os.path.getsize(output_path)} bytes.", flush=True)

    torch.cuda.empty_cache()

    print("✅ Done!", flush=True)
    return {"filename": filename}
