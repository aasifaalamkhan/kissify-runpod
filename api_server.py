import os
from flask import Flask, request, jsonify, send_from_directory, url_for
from inference import generate_kissing_video, pipe
import torch

app = Flask(__name__)

# Define the directory where videos will be stored
OUTPUT_DIR = "/workspace/outputs"
# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Pre-load and Warm-up Model ---
print("[INFO] Loading model to GPU for the first time...")
# The pipe is already moved to the GPU in inference.py

print("[INFO] Performing warm-up inference step...")
with torch.inference_mode():
    # --- FIX: Create dummy embeddings with the same dtype as the pipeline (float32) ---
    positive_embeds = torch.randn(1, 1, 1, 1024, dtype=pipe.dtype, device="cuda")
    negative_embeds = torch.zeros_like(positive_embeds)
    dummy_ip_adapter_embeds = torch.cat([negative_embeds, positive_embeds], dim=0)
    
    pipe(
        prompt="warmup",
        num_inference_steps=1,
        num_frames=1,
        ip_adapter_image_embeds=[dummy_ip_adapter_embeds]
    )
torch.cuda.empty_cache()
print("✅ [INFO] Model is warmed up and ready.")


# --- Route to Serve Video Files ---
@app.route('/outputs/<path:filename>')
def serve_video(filename):
    """
    Serves a video file from the output directory.
    """
    print(f"Serving file: {filename} from {OUTPUT_DIR}", flush=True)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


# --- Main API Route ---
@app.route('/generate', methods=['POST'])
def handle_generation():
    """
    The main API endpoint for video generation.
    """
    data = request.get_json()
    if not data or 'face_image1' not in data or 'face_image2' not in data:
        return jsonify({"error": "Request must include 'face_image1' and 'face_image2'"}), 400

    try:
        result = generate_kissing_video(data)
        filename = result.get("filename")

        if not filename:
            raise RuntimeError("Generation succeeded but did not return a filename.")

        proto = request.headers.get("X-Forwarded-Proto", "http")
        host = request.headers.get("X-Forwarded-Host", request.host)
        base_url = f"{proto}://{host}"
        video_path = url_for('serve_video', filename=filename)
        video_url = f"{base_url}{video_path}"
        
        print(f"Generated public video URL: {video_url}", flush=True)
        return jsonify({"video_url": video_url})

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        return jsonify({"error": f"An error occurred during generation: {str(e)}"}), 500


# --- Start Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
