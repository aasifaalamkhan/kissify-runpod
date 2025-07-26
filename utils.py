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
import ffmpeg
from facenet_pytorch import MTCNN
from controlnet_aux import OpenposeDetector # New import for pose detection

# ... (Face detector initialization is unchanged) ...
print("Initializing MTCNN face detector...", flush=True)
face_detector = MTCNN(
    keep_all=False, post_process=False, min_face_size=40,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("✅ Face detector initialized.", flush=True)

# --- NEW: Initialize the OpenPose detector ---
print("Initializing OpenPose detector...", flush=True)
pose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
print("✅ OpenPose detector initialized.", flush=True)


# --- NEW FUNCTION TO EXTRACT POSE SEQUENCE FROM TEMPLATE ---
def extract_pose_sequence(template_video_path):
    """
    Processes a template video to extract a sequence of OpenPose skeleton images.
    """
    if not os.path.exists(template_video_path):
        raise FileNotFoundError(f"Motion template not found at {template_video_path}. Please add the file to your workspace.")

    print(f"Extracting pose sequence from {template_video_path}...", flush=True)
    reader = imageio.get_reader(template_video_path)
    pose_sequence = []
    for frame in reader:
        pil_frame = Image.fromarray(frame).convert("RGB")
        pose_image = pose_detector(pil_frame)
        pose_sequence.append(pose_image)
    reader.close()
    
    if not pose_sequence:
        raise ValueError("Could not extract any frames or poses from the motion template.")
        
    print(f"✅ Extracted {len(pose_sequence)} poses from motion template.", flush=True)
    return pose_sequence

# ... (The rest of the functions are unchanged) ...
def crop_face(pil_image):
    # ...
    return cropped_image
def prepare_ip_adapter_inputs(images, device="cuda"):
    # ...
    return [processor(img).unsqueeze(0).to(device) for img in images]
def load_and_encode_image(image_path):
    # ...
    return base64.b64encode(f.read()).decode("utf-8")
def load_face_images(image_b64_list):
    # ...
    return images
def export_video_with_imageio(video_frames, output_path, fps):
    # ...
    print(f"Video saved to {output_path} using imageio.", flush=True)
def upscale_video(input_path, output_path, device="cuda"):
    # ...
    print(f"✅ Upscaled video saved successfully to {output_path}", flush=True)
def smooth_video(input_path, output_path, target_fps=48):
    # ...
    print(f"✅ Smoothed video saved successfully to {output_path}", flush=True)
