import os
from flask import Flask, request, jsonify, send_from_directory, url_for, Response, stream_with_context
from inference import generate_kissing_video, pipe 
import json

app = Flask(__name__)

# Define the directory where videos will be stored
OUTPUT_DIR = "/workspace/outputs"
# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ [INFO] Model loaded and ready via inference.py import.")

# --- Route to Serve Video and Image Files ---
@app.route('/outputs/<path:filename>')
def serve_video(filename):
    """
    Serves a file (video or image) from the output directory.
    """
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


# --- Main API Route (MODIFIED FOR STREAMING) ---
@app.route('/generate', methods=['POST'])
def handle_generation():
    """
    The main API endpoint for video generation.
    NOW STREAMS progress back to the client.
    """
    data = request.get_json()
    if not data or 'face_image1' not in data or 'face_image2' not in data:
        return jsonify({"error": "Request must include 'face_image1' and 'face_image2'"}), 400

    def generate_stream():
        final_filename = None
        try:
            # The inference function now yields its progress
            for log_message in generate_kissing_video(data):
                # Check if the message is the final result, composite filename, or a log
                if isinstance(log_message, dict) and 'filename' in log_message:
                    final_filename = log_message['filename']
                    yield f"data: {json.dumps({'status': 'Done'})}\n\n"
                # --- NEW: Handle the composite image filename ---
                elif isinstance(log_message, dict) and 'composite_filename' in log_message:
                    comp_filename = log_message['composite_filename']
                    proto = request.headers.get("X-Forwarded-Proto", "http")
                    host = request.headers.get("X-Forwarded-Host", request.host)
                    base_url = f"{proto}://{host}"
                    comp_path = url_for('serve_video', filename=comp_filename)
                    comp_url = f"{base_url.rstrip('/')}{comp_path}"
                    # Yield the composite image URL to the client
                    yield f"data: {json.dumps({'composite_image_url': comp_url})}\n\n"
                else:
                    # Send progress updates as Server-Sent Events (SSE)
                    yield f"data: {json.dumps({'status': log_message})}\n\n"

            if not final_filename:
                 raise RuntimeError("Generation finished but did not return a filename.")

            # Construct the final URL once generation is complete
            proto = request.headers.get("X-Forwarded-Proto", "http")
            host = request.headers.get("X-Forwarded-Host", request.host)
            base_url = f"{proto}://{host}"
            video_path = url_for('serve_video', filename=final_filename)
            video_url = f"{base_url.rstrip('/')}{video_path}"

            # Yield the final URL as the last message
            final_payload = {"video_url": video_url}
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"An error occurred during generation: {str(e)}"
            # Yield the error message
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    # Return a streaming response
    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')


# --- Start Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)