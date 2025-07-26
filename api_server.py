import os
from flask import Flask, request, jsonify, send_from_directory, url_for
from inference import generate_kissing_video, pipe  # This line loads and prepares the model

app = Flask(__name__)

# Define the directory where videos will be stored
OUTPUT_DIR = "/workspace/outputs"
# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ [INFO] Model loaded and ready via inference.py import.")

# --- Route to Serve Video Files ---
@app.route('/outputs/<path:filename>')
def serve_video(filename):
    """
    Serves a video file from the output directory.
    """
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

        # This logic correctly constructs the public URL for RunPod
        proto = request.headers.get("X-Forwarded-Proto", "http")
        host = request.headers.get("X-Forwarded-Host", request.host)
        base_url = f"{proto}://{host}"
        video_path = url_for('serve_video', filename=filename)
        video_url = f"{base_url.rstrip('/')}{video_path}"

        print(f"Generated public video URL: {video_url}", flush=True)
        return jsonify({"video_url": video_url})

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        # It's helpful to see the full traceback in the server logs for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during generation: {str(e)}"}), 500


# --- Start Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
