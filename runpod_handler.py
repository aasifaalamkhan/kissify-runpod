import runpod
from inference import generate_kissing_video, pipe  # Import the pipe
import torch

# Ensure the model is loaded before starting the handler
print("[INFO] Pre-loading and warming up the model for RunPod...")
pipe.to("cuda")
# Perform a single dummy inference step to warm up the model
pipe(prompt="warmup", num_inference_steps=1, num_frames=1)
torch.cuda.empty_cache()
print("[INFO] Model is ready.")


def handler(job):
    """
    This is the handler function that RunPod will call for each job.
    """
    job_input = job['input']

    # Basic input validation
    if 'face_image1' not in job_input or 'face_image2' not in job_input:
        return {"error": "Input must include 'face_image1' and 'face_image2'."}

    try:
        # Pass the validated input directly to your generation function
        result = generate_kissing_video(job_input)
        return result
    except Exception as e:
        # It's good practice to return any errors
        return {"error": f"An error occurred during generation: {str(e)}"}


# Start the serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})