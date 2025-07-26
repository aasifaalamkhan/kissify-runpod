import os
import torch
import uuid # For generating unique filenames
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
# --- We need the imageio exporter and the input preparer again ---
from utils import load_face_images, prepare_ip_adapter_inputs, export_video_with_imageio

# Define the directory where videos will be stored
OUTPUT_DIR = "/workspace/outputs"


# ========= Load Models =========
print("[INFO] Initializing models and pipeline...", flush=True)

base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-3"
ip_adapter_repo_id = "h94/IP-Adapter"
device = "cuda"

# --- FIX: We must load the image encoder and processor to manually create embeddings ---
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
pipe.enable_model_cpu_offload()

# Load the IP-Adapter
pipe.load_ip_adapter(
    ip_adapter_repo_id,
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)

print("[INFO] Models and pipeline are initialized.", flush=True)

# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    # --- FIX: Reverting to manual embedding generation to handle two faces correctly ---
    print("🧠 Loading and preparing face images...", flush=True)
    pil_images = load_face_images([
        input_data['face_image1'],
        input_data['face_image2']
    ])
    # Prepare images for the encoder
    prepared_images = prepare_ip_adapter_inputs(pil_images, device)

    print("🔍 Encoding faces with IP-Adapter...", flush=True)
    face_embeds = []
    for image in prepared_images:
        # Manually encode each image
        embeds = image_encoder(image).image_embeds
        face_embeds.append(embeds)

    # Average the embeddings of the two faces
    positive_embeds = torch.cat(face_embeds, dim=0).mean(dim=0, keepdim=True)
    
    # Create negative embeddings (unconditional)
    negative_embeds = torch.zeros_like(positive_embeds)
    
    # Combine for classifier-free guidance
    ip_embeds = torch.cat([negative_embeds, positive_embeds], dim=0)

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
            ip_adapter_image_embeds=[ip_embeds], # Pass the combined embeddings
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=30,
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
