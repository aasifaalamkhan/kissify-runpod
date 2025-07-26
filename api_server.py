from flask import Flask, request, jsonify
from inference import generate_kissing_video, pipe
import torch

app = Flask(__name__)

# --- Pre-load and Warm-up Model ---
print("[INFO] Loading model to GPU for the first time...")
pipe.to("cuda")

# --- FIX: Create dummy embeddings for the warmup call ---
# The IP-Adapter expects both negative and positive embeddings for classifier-free guidance.
print("[INFO] Performing warm-up inference step...")
with torch.inference_mode():
    # Create a dummy positive embedding
    positive_embeds = torch.randn(1, 1, 1024, dtype=torch.float16, device="cuda")
    # Create a dummy negative embedding (a tensor of zeros)
    negative_embeds = torch.zeros_like(positive_embeds)
    # Concatenate them. The pipeline expects the format [negative, positive].
    dummy_ip_adapter_embeds = torch.cat([negative_embeds, positive_embeds], dim=0)

    pipe(
        prompt="warmup",
        num_inference_steps=1,
        num_frames=1,
        ip_adapter_image_embeds=[dummy_ip_adapter_embeds] # Pass the combined tensor
    )
torch.cuda.empty_cache()
print("✅ [INFO] Model is warmed up and ready.")


# --- API Route ---
@app.route('/generate', methods=['POST'])
def handle_generation():
    """
    The main API endpoint for video generation.
    """
    data = request.get_json()
    if not data or 'face_image1' not in data or 'face_image2' not in data:
        return jsonify({"error": "Request must include 'face_image1' and 'face_image2'"}), 400

    try:
        # Pass the validated input directly to your generation function
        result = generate_kissing_video(data)
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred during generation: {str(e)}"}), 500


# --- Start Server ---
if __name__ == '__main__':
    # Running on 0.0.0.0 makes the server accessible from outside the container
    app.run(host='0.0.0.0', port=5000)
