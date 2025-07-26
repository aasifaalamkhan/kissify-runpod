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


def upload_to_catbox(file_path, max_retries=3, delay=3):
    url = "https://catbox.moe/user/api.php"

    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    url,
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": f},
                    timeout=30
                )
            if response.status_code == 200 and response.text.startswith("https://"):
                return response.text.strip()
            else:
                raise RuntimeError(f"Catbox upload failed with status {response.status_code}: {response.text}")
        except Exception as e:
            if attempt < max_retries:
                print(f"[WARN] Upload attempt {attempt} failed: {e}")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Catbox upload failed after {max_retries} attempts: {e}")


def export_video_with_opencv(video_frames, output_path, fps):
    """
    Manually exports video frames using OpenCV to ensure codec compatibility.
    """
    if not video_frames:
        raise ValueError("Input video_frames list is empty.")
    
    # Get frame size from the first image
    first_frame = video_frames[0]
    height, width, layers = np.array(first_frame).shape
    size = (width, height)
    
    # --- FIX: Switched to a more compatible H.264 codec ('avc1') ---
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for frame_pil in video_frames:
        # Convert PIL Image (RGB) to OpenCV format (BGR)
        frame_np = np.array(frame_pil)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release() # Release the VideoWriter
    print(f"Video saved to {output_path} using OpenCV with avc1 codec.")
