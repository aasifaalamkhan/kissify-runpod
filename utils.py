import base64
from PIL import Image
import io
import os
import requests
import time
from torchvision import transforms
import numpy as np
import cv2 # Import OpenCV

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


def export_video_with_opencv(video_frames, output_path, fps):
    """
    Manually exports video frames using OpenCV to ensure codec compatibility.
    """
    if not video_frames:
        raise ValueError("Input video_frames list is empty.")
    
    first_frame = video_frames[0]
    height, width, layers = np.array(first_frame).shape
    size = (width, height)
    
    # --- Trying the 'avc1' (H.264) codec as a more compatible alternative ---
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for frame_pil in video_frames:
        frame_np = np.array(frame_pil)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    # Add flush=True to ensure this message appears immediately
    print(f"Video saved to {output_path} using OpenCV with avc1 codec.", flush=True)
