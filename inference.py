import os
import torch
import tempfile
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
from utils import load_face_images, prepare_ip_adapter_inputs, upload_to_catbox

# ========= Load Models =========
print("[INFO] Initializing models and pipeline...")

base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"
device = "cuda"

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float16
).to(device).eval()

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

motion_adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16).to(device)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_id,
    motion_adapter=motion_adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
pipe.load_ip_adapter(
    ip_adapter_repo_id,
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)
pipe.set_ip_adapter_scale(1.0)

print("[INFO] Models and pipeline are initialized.")

# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    # Model is already on the GPU from the server startup
    print("🧠 Loading and preparing face images...")
    face_images = load_face_images([
        input_data['face_image1'],
        input_data['face_image2']
    ])
    face_images = prepare_ip_adapter_inputs(face_images, device)

    print("🔍 Encoding faces with IP-Adapter...")
    face_embeds = []
    for face in face_images:
        inputs = image_processor(face, return_tensors="pt", do_rescale=False).to(device)
        embeds = image_encoder(**inputs).image_embeds
        face_embeds.append(embeds)

    # This is the positive embedding
    stacked_embeds = torch.cat(face_embeds, dim=0).mean(dim=0, keepdim=True)

    # --- FIX: Create negative embeddings and combine with positive embeddings ---
    # Create a tensor of zeros for the negative embeddings
    negative_embeds = torch.zeros_like(stacked_embeds)
    # Combine them into a single tensor for the pipeline
    ip_embeds = torch.cat([negative_embeds, stacked_embeds], dim=0)


    prompt = (input_data.get("prompt") or "").strip()
    if not prompt:
        prompt = "romantic kiss, closeup, cinematic, photorealistic, 4k, trending on artstation"
    
    negative_prompt = "bad quality, worse quality, low resolution, deformed, ugly"

    print(f"🎨 Generating animation with prompt: '{prompt}'")
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=16,
            guidance_scale=7.0,
            num_inference_steps=25,
            ip_adapter_image_embeds=[ip_embeds] # Use the combined embeddings
        ).frames[0]

    video_frames = result
    print("💾 Exporting video...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
      temp_path = f.name
    export_to_video(video_frames, temp_path, fps=8)

    print("☁️ Uploading to Catbox...")
    video_url = upload_to_catbox(temp_path)

    os.remove(temp_path)
    torch.cuda.empty_cache()

    print("✅ Done!")
    return {"video_url": video_url}
