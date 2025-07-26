import os
import torch
import uuid
import gc
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
# Import the new upscaling function
from utils import load_face_images, prepare_ip_adapter_inputs, export_video_with_imageio, upscale_video

# Define the directory where videos will be stored
OUTPUT_DIR = "/workspace/outputs"


# ========= Load Models =========
# ... (This section remains unchanged) ...
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
    # ... (Variable initialization remains unchanged) ...
    ip_embeds = None
    positive_embeds = None
    negative_embeds = None
    video_frames = None
    output = None

    try:
        # ... (Image loading and embedding remain unchanged) ...
        print("🧠 Loading and preparing face images...", flush=True)
        pil_images = load_face_images([
            input_data['face_image1'],
            input_data['face_image2']
        ])
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

        # ... (Prompts and parameters remain unchanged from the last update) ...
        prompt = (input_data.get("prompt") or "").strip()
        if not prompt:
            prompt = "masterpiece, best quality, ultra-realistic 4k photo of a passionate kiss between a man and a woman, closed eyes, gentle smile, cinematic lighting, intimate atmosphere, soft-focus background, romantic"
        negative_prompt = "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), deformed, ugly, disfigured, weird faces, open mouths, awkward poses, stiff, blurry faces"
        pipe.set_ip_adapter_scale(1.8)

        print(f"🎨 Generating base animation with prompt: '{prompt}'", flush=True)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=[ip_embeds],
                num_frames=48,
                guidance_scale=5.0,
                num_inference_steps=50,
            ).frames[0]
        video_frames = output

        # --- MODIFIED WORKFLOW: EXPORT, UPSCALE, and RETURN ---
        
        # 1. Export a temporary, low-resolution video
        temp_filename = f"temp_{uuid.uuid4()}.mp4"
        temp_output_path = os.path.join(OUTPUT_DIR, temp_filename)
        print("💾 Exporting temporary low-resolution video...", flush=True)
        export_video_with_imageio(video_frames, temp_output_path, fps=24)

        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            raise RuntimeError("Temporary MP4 export failed.")
        
        # 2. Upscale the temporary video to the final output file
        final_filename = f"{uuid.uuid4()}.mp4"
        final_output_path = os.path.join(OUTPUT_DIR, final_filename)
        upscale_video(temp_output_path, final_output_path)
        
        # 3. Clean up the temporary video file
        print(f"Cleaning up temporary file: {temp_output_path}", flush=True)
        os.remove(temp_output_path)
        
        print("✅ Full generation and upscaling process complete!", flush=True)
        return {"filename": final_filename}

    finally:
        # --- GUARANTEED MEMORY CLEANUP (Unchanged) ---
        print("🧹 Cleaning up GPU memory...", flush=True)
        del ip_embeds
        del positive_embeds
        del negative_embeds
        if output is not None:
            del output
        if video_frames is not None:
            del video_frames
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared.", flush=True)
