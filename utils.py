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
import ffmpeg # New import

# ... (prepare_ip_adapter_inputs, load_and_encode_image, load_face_images functions are unchanged) ...

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
    # ... (This function is unchanged) ...
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


def upscale_video(input_path, output_path, device="cuda"):
    # ... (This function is unchanged) ...
    print(f"🚀 Starting video upscaling for {input_path}...", flush=True)
    temp_frame_dir = f"/tmp/frames_{os.path.basename(input_path)}"
    temp_upscaled_dir = f"/tmp/upscaled_{os.path.basename(input_path)}"
    os.makedirs(temp_frame_dir, exist_ok=True)
    os.makedirs(temp_upscaled_dir, exist_ok=True)
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data()['fps']
        for i, frame in enumerate(reader):
            imageio.imwrite(os.path.join(temp_frame_dir, f'frame_{i:04d}.png'), frame)
        reader.close()
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model, dni_weight=None, tile=0, tile_pad=10, pre_pad=0, half=True,
            gpu_id=0 if "cuda" in device else None
        )
        frame_files = sorted(os.listdir(temp_frame_dir))
        for frame_file in frame_files:
            img = cv2.imread(os.path.join(temp_frame_dir, frame_file))
            output, _ = upsampler.enhance(img, outscale=4)
            cv2.imwrite(os.path.join(temp_upscaled_dir, frame_file), output)
        writer = imageio.get_writer(output_path, fps=fps, quality=9, macro_block_size=None, output_params=['-pix_fmt', 'yuv420p'])
        upscaled_files = sorted(os.listdir(temp_upscaled_dir))
        for upscaled_file in upscaled_files:
            frame = imageio.imread(os.path.join(temp_upscaled_dir, upscaled_file))
            writer.append_data(frame)
        writer.close()
        print(f"✅ Upscaled video saved successfully to {output_path}", flush=True)
    finally:
        shutil.rmtree(temp_frame_dir)
        shutil.rmtree(temp_upscaled_dir)


# --- NEW FUNCTION FOR VIDEO SMOOTHING ---
def smooth_video(input_path, output_path, target_fps=48):
    """
    Smooths a video by interpolating frames to a higher FPS using FFmpeg.
    """
    print(f"🌊 Smoothing video to {target_fps}fps for {input_path}...", flush=True)
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('minterpolate', fps=target_fps, mi_mode='mci') # mci = motion compensated interpolation
            .output(output_path, pix_fmt='yuv420p', q=8) # Set quality
            .run(overwrite_output=True, quiet=True)
        )
        print(f"✅ Smoothed video saved successfully to {output_path}", flush=True)
    except ffmpeg.Error as e:
        print("FFmpeg Error:", e.stderr.decode(), flush=True)
        raise
