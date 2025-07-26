import base64
from PIL import Image
import io
import os
import requests
import time
from torchvision import transforms
import numpy as np
import cv2
import imageio
import shutil
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def prepare_ip_adapter_inputs(images, device="cuda"):
    """
    Prepares a list of PIL images for IP-Adapter image encoder (CLIP format)
    """
    processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return [processor(img).unsqueeze(0).to(device) for img in images]


def load_and_encode_image(image_path):
    """
    Opens an image file and returns base64-encoded string
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_face_images(image_b64_list):
    """
    Converts base64 images to PIL Image objects
    """
    images = []
    for img_b64 in image_b64_list:
        try:
            img_bytes = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(image)
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
    return images


def export_video_with_imageio(video_frames, output_path, fps):
    """
    Exports video frames using imageio, which is often more robust at finding ffmpeg.
    """
    if not video_frames:
        raise ValueError("Input video_frames list is empty.")

    numpy_frames = [np.array(frame) for frame in video_frames]
    
    imageio.mimwrite(
        output_path,
        numpy_frames,
        fps=fps,
        quality=8,
        macro_block_size=None,
        output_params=['-pix_fmt', 'yuv420p']
    )
    print(f"Video saved to {output_path} using imageio.", flush=True)

# --- NEW FUNCTION FOR VIDEO UPSCALING ---
def upscale_video(input_path, output_path, device="cuda"):
    """
    Upscales a video using Real-ESRGAN.
    """
    print(f"🚀 Starting video upscaling for {input_path}...", flush=True)
    
    # --- Setup temporary directories ---
    temp_frame_dir = f"/tmp/frames_{os.path.basename(input_path)}"
    temp_upscaled_dir = f"/tmp/upscaled_{os.path.basename(input_path)}"
    os.makedirs(temp_frame_dir, exist_ok=True)
    os.makedirs(temp_upscaled_dir, exist_ok=True)

    try:
        # --- 1. Extract frames from video ---
        print("Extracting frames...", flush=True)
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data()['fps']
        for i, frame in enumerate(reader):
            imageio.imwrite(os.path.join(temp_frame_dir, f'frame_{i:04d}.png'), frame)
        reader.close()

        # --- 2. Initialize Real-ESRGAN model ---
        print("Initializing Real-ESRGAN upscaler...", flush=True)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            dni_weight=None,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True, # Use half precision for speed
            gpu_id=0 if "cuda" in device else None
        )

        # --- 3. Upscale each frame ---
        print("Upscaling frames...", flush=True)
        frame_files = sorted(os.listdir(temp_frame_dir))
        for frame_file in frame_files:
            img = cv2.imread(os.path.join(temp_frame_dir, frame_file))
            output, _ = upsampler.enhance(img, outscale=4)
            cv2.imwrite(os.path.join(temp_upscaled_dir, frame_file), output)

        # --- 4. Stitch upscaled frames back into a video ---
        print("Stitching upscaled frames into final video...", flush=True)
        writer = imageio.get_writer(output_path, fps=fps, quality=9, macro_block_size=None, output_params=['-pix_fmt', 'yuv420p'])
        upscaled_files = sorted(os.listdir(temp_upscaled_dir))
        for upscaled_file in upscaled_files:
            frame = imageio.imread(os.path.join(temp_upscaled_dir, upscaled_file))
            writer.append_data(frame)
        writer.close()

        print(f"✅ Upscaled video saved successfully to {output_path}", flush=True)

    finally:
        # --- 5. Cleanup temporary directories ---
        print("Cleaning up temporary frame directories...", flush=True)
        shutil.rmtree(temp_frame_dir)
        shutil.rmtree(temp_upscaled_dir)
