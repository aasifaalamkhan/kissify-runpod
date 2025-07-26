import base64
from PIL import Image
import io
import os
import torch
import numpy as np
import cv2
import imageio
import ffmpeg
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from facenet_pytorch import MTCNN
from controlnet_aux import OpenposeDetector

# --- Initialize Detectors ---
print("[utils.py] Initializing MTCNN face detector...", flush=True)
face_detector = MTCNN(
    keep_all=False, post_process=False, min_face_size=40,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("[utils.py] Initializing OpenPose detector...", flush=True)
pose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
print("✅ Detectors initialized.", flush=True)


def load_face_images(image_b64_list):
    """Decodes a list of base64 strings into a list of PIL images."""
    images = []
    for b64_str in image_b64_list:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
    return images

def crop_face(pil_image):
    """Detects a face and returns a cropped PIL Image."""
    boxes, _ = face_detector.detect(pil_image)
    if boxes is None or len(boxes) == 0:
        print("⚠️ Warning: No face detected. Using resized original image as fallback.", flush=True)
        return pil_image.resize((224, 224))
    
    # Use the first detected face
    box = boxes[0]
    # The box is [x1, y1, x2, y2], safe to pass directly to crop
    return pil_image.crop(box)

def extract_pose_sequence(template_video_path):
    """Processes a template video to extract a sequence of OpenPose skeleton images."""
    if not os.path.exists(template_video_path):
        raise FileNotFoundError(f"Motion template not found at {template_video_path}.")

    print(f"Extracting pose sequence from {template_video_path}...", flush=True)
    reader = imageio.get_reader(template_video_path)
    pose_sequence = [pose_detector(Image.fromarray(frame).convert("RGB")) for frame in reader]
    reader.close()
    
    if not pose_sequence:
        raise ValueError("Could not extract any poses from the motion template.")
    
    print(f"✅ Extracted {len(pose_sequence)} poses.", flush=True)
    return pose_sequence

def save_pose_sequence(pose_sequence, filepath):
    """Saves a list of pose images to a .npy file for caching."""
    numpy_poses = [np.array(p) for p in pose_sequence]
    np.save(filepath, numpy_poses)
    print(f"✅ Saved pose sequence to cache: {filepath}", flush=True)

def load_pose_sequence(filepath):
    """Loads a cached pose sequence from a .npy file."""
    if not os.path.exists(filepath):
        return None
    print(f"Loading cached pose sequence from {filepath}...", flush=True)
    numpy_poses = np.load(filepath, allow_pickle=True)
    pose_sequence = [Image.fromarray(p) for p in numpy_poses]
    print(f"✅ Loaded {len(pose_sequence)} poses from cache.", flush=True)
    return pose_sequence

def export_video_with_imageio(video_frames, output_path, fps):
    """Saves a list of frames to a video file."""
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in video_frames:
        writer.append_data(np.array(frame))
    writer.close()

def upscale_video(input_path, output_path, device="cuda"):
    """Upscales a video file using RealESRGAN."""
    model_name = 'RealESRGAN_x4plus'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth',
        model=model, tile=400, tile_pad=10, pre_pad=0, half=True, gpu_id=0 if 'cuda' in device else None
    )
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 4, height * 4))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = upsampler.enhance(img, outscale=4)
        output = cv2.cvtColor(output, cv2.COLOR_RGB_BGR)
        out.write(output)
    cap.release()
    out.release()

def smooth_video(input_path, output_path, target_fps=48):
    """Increases video frame rate for smoother motion using ffmpeg."""
    try:
        (
            ffmpeg.input(input_path).filter('minterpolate', fps=target_fps, mi_mode='mci')
            .output(output_path).run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as e:
        print('ffmpeg stdout:', e.stdout.decode('utf8'))
        print('ffmpeg stderr:', e.stderr.decode('utf8'))
        raise e
