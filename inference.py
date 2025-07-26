import os
import torch
import uuid
import gc
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
# Import the new smoothing function
from utils import load_face_images, prepare_ip_adapter_inputs, export_video_with_imageio, upscale_video, smooth_video

# ... (OUTPUT_DIR and Model Loading sections are unchanged) ...
OUTPUT_DIR = "/workspace/outputs"
print("[INFO] Initializing models and pipeline...", flush=True)
base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"
device = "cuda"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    ip_adapter_repo_id, subfolder="models/image_encoder"
).to(device).eval()
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
motion_adapter = MotionAdapter.from_pretrained(motion_module_id).to(device)
pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
)
pipe.scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
pipe.to(device)
pipe.load_ip_adapter(
    ip_adapter_repo_id,
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)
print("[INFO] Models and pipeline are initialized.", flush=True)


# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    # ... (Variable initialization and image processing are unchanged) ...
    ip_embeds = None
    positive_embeds = None
    negative_embeds = None
    video_frames = None
    output = None
    
    # Define paths for temporary files
    temp_base_path = os.path.join(OUTPUT_DIR, f"temp_base_{uuid.uuid4()}.mp4")
    temp_upscaled_path = os.path.join(OUTPUT_DIR, f"temp_upscaled_{uuid.uuid4()}.mp4")
    final_filename = f"{uuid.uuid4()}.mp4"
    final_output_path = os.path.join(OUTPUT_DIR, final_filename)

    try:
        # ... (Image loading and embedding are unchanged) ...
        print("🧠 Loading and preparing face images...", flush=True)
        pil_images = load_face_images([input_data['face_image1'], input_data['face_image2']])
        prepared_images = prepare_ip_adapter_inputs(pil_images, device)
        print("🔍 Encoding faces with IP-Adapter...", flush=True)
        face_embeds = []
        for image in prepared_images:
            embeds = image_encoder(image).image_embeds
            face_embeds.append(embeds)
        positive_embeds = torch.cat(face_embeds, dim=0).mean(dim=0, keepdim=True)
        positive_embeds = positive_embeds.to(dtype=pipe.dtype)
        positive_embeds = positive_embeds.unsqueeze(0)
        negative_embeds = torch.zeros_like(positive_embeds)
        ip_embeds = torch.cat([negative_embeds, positive_embeds], dim=0)

        # ... (Prompts and parameters are unchanged) ...
        prompt = (input_data.get("prompt") or "").strip()
        if not prompt:
            prompt = "masterpiece, best quality, ultra-realistic 4k photo of a passionate kiss between a man and a woman, closed eyes, gentle smile, cinematic lighting, intimate atmosphere, soft-focus background, romantic"
        negative_prompt = "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), deformed, ugly, disfigured, weird faces, open mouths, awkward poses, stiff, blurry faces"
        pipe.set_ip_adapter_scale(1.8)

        print(f"🎨 Generating base animation with prompt: '{prompt}'", flush=True)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt, negative_prompt=negative_prompt, ip_adapter_image_embeds=[ip_embeds],
                num_frames=48, guidance_scale=5.0, num_inference_steps=50,
            ).frames[0]
        video_frames = output

        # --- NEW MULTI-STAGE POST-PROCESSING WORKFLOW ---
        
        # 1. Export a temporary, base video
        print("💾 Step 1/3: Exporting base video...", flush=True)
        export_video_with_imageio(video_frames, temp_base_path, fps=24)
        
        # 2. Upscale the base video
        print("🚀 Step 2/3: Upscaling video...", flush=True)
        upscale_video(temp_base_path, temp_upscaled_path)
        
        # 3. Smooth the upscaled video to create the final product
        print("🌊 Step 3/3: Smoothing video...", flush=True)
        smooth_video(temp_upscaled_path, final_output_path, target_fps=48)
        
        print("✅ Full generation and post-processing complete!", flush=True)
        return {"filename": final_filename}

    finally:
        # --- GUARANTEED MEMORY & FILE CLEANUP ---
        print("🧹 Cleaning up GPU memory and temporary files...", flush=True)
        
        # Cleanup python variables
        del ip_embeds, positive_embeds, negative_embeds, output, video_frames
        
        # Cleanup temporary files from disk
        if os.path.exists(temp_base_path):
            os.remove(temp_base_path)
        if os.path.exists(temp_upscaled_path):
            os.remove(temp_upscaled_path)

        # Cleanup GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Cleanup complete.", flush=True)
