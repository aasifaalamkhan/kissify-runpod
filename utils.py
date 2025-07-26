import base64
from PIL import Image
import io
import os
import requests
import time
from torchvision import transforms
import numpy as np
import cv2 # Still needed for color conversion if we use it
import imageio # Import the new library

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

    print(f"Attempting to save video with imageio to {output_path}...", flush=True)
    
    # --- FIX: Explicitly set the pixel format to the most compatible option ---
    imageio.mimwrite(
        output_path, 
        numpy_frames, 
        fps=fps, 
        quality=8, 
        macro_block_size=None,
        pix_fmt='yuv420p' # Add this line for maximum browser compatibility
    )
    
    print(f"Video saved to {output_path} using imageio.", flush=True)

