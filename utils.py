import base64
from PIL import Image
import io
import os
import requests
import time
import torch  # <-- ADD THIS LINE
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
    boxes, _ = face_detector.detect(pil_image)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No face detected in one of the images.")
    
    # Get the bounding box of the first detected face
    box = boxes[0]
    
    # Crop the image
    # The box is [x1, y1, x2, y2]
    cropped_image = pil_image.crop(box)
    return cropped_image

def prepare_ip_adapter_inputs(images, device="cuda"):
    processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return [processor(img).unsqueeze(0).to(device) for img in images]

def load_and_encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_face_images(image_b64_list):
    images = []
    for b64_str in image_b64_list:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
    return images

def export_video_with_imageio(video_frames, output_path, fps):
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in video_frames:
        writer.append_data(np.array(frame))
    writer.close()
    print(f"Video saved to {output_path} using imageio.", flush=True)

def upscale_video(input_path, output_path, device="cuda"):
    print(f"Upscaling video: {input_path}", flush=True)
    model_name = 'RealESRGAN_x4plus_anime_6B'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/{model_name}.pth',
        model=model,
        dni_weight=None,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True if 'cuda' in device else False,
        gpu_id=0 if 'cuda' in device else None
    )

    # Use ffmpeg-python to handle video frames
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 4, height * 4))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Upscale frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = upsampler.enhance(img, outscale=4)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        out.write(output)

    cap.release()
    out.release()
    print(f"✅ Upscaled video saved successfully to {output_path}", flush=True)

def smooth_video(input_path, output_path, target_fps=48):
    print(f"Smoothing video to {target_fps} FPS: {input_path}", flush=True)
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('minterpolate', fps=target_fps, mi_mode='mci', mc_mode='aobmc', me_mode='bidir', vsbmc=1)
            .output(output_path)
            .run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as e:
        print('ffmpeg stdout:', e.stdout.decode('utf8'))
        print('ffmpeg stderr:', e.stderr.decode('utf8'))
        raise e
    print(f"✅ Smoothed video saved successfully to {output_path}", flush=True)
