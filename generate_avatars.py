cat > generate_avatars.py << 'EOF'
import os
import logging
import random
import boto3
import cv2
import numpy as np
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Import GFPGAN restoration tool
from gfpgan import GFPGANer

# Load environment variables from .env file or system environment
load_dotenv()

# AWS configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")

# Optional: Fine-tuned model path (if provided, this model will be used instead of the base SDXL)
FINE_TUNED_MODEL_PATH = os.getenv("FINE_TUNED_MODEL_PATH")  # e.g., "./my_finetuned_sdxl_model"

# GFPGAN configuration: set path to GFPGAN model file (download from GFPGAN repository)
GFPGAN_MODEL_PATH = os.getenv("GFPGAN_MODEL_PATH", "GFPGANv1.3.pth")
GFPGAN_UPSCALE = 2  # Upscale factor for GFPGAN output

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
feedback_log = open("generation_feedback.log", "a")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_S3_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load SDXL pipeline
if FINE_TUNED_MODEL_PATH and os.path.exists(FINE_TUNED_MODEL_PATH):
    model_path = FINE_TUNED_MODEL_PATH
    logging.info(f"Loading fine-tuned model from {model_path}")
else:
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    logging.info(f"Loading base model: {model_path}")

pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipeline = pipeline.to(device)

# Initialize GFPGANer for post-processing (ensure the GFPGAN model file is available)
gfpganer = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=GFPGAN_UPSCALE,
    arch='clean',  # or use 'original' if preferred
    channel_multiplier=2,
    device=device
)

# Generation settings
NUM_AVATARS = 1000  # total number of avatars to generate
VIEWS = ["front", "profile", "back"]
ensemble_size = 3  # number of samples to generate per view for selection
# List of ethnicities for variation
ethnicities = ["Caucasian", "African", "Asian", "Latina", "Middle Eastern"]

# Output directory for generated avatars
output_dir = "generated_avatars"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_prompt(view: str, ethnicity: str) -> str:
    """
    Generate an advanced composite prompt for a given view and ethnicity.
    """
    base_prompt = (
        f"Ultra-detailed, hyper-realistic portrait of a young {ethnicity} woman between 18 and 30 years old, "
        f"professional studio lighting, intricate facial features, and high resolution."
    )
    # Append view-specific details
    if view == "front":
        view_details = "Full frontal face with direct gaze."
    elif view == "profile":
        view_details = "Side profile capturing detailed contours."
    elif view == "back":
        view_details = "Slightly turned back view showing hair and neck details."
    else:
        view_details = ""
    # Combine into final prompt
    prompt = f"{base_prompt} {view_details}"
    return prompt

def calculate_quality_metric(image: Image.Image) -> float:
    """
    Use variance of Laplacian on a grayscale version as a simple sharpness metric.
    Higher value indicates a sharper (potentially higher quality) image.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(image_cv, cv2.CV_64F).var()

def post_process_image(image_path: str) -> Image.Image:
    """
    Enhance the image using GFPGAN for face restoration.
    """
    # Read image using OpenCV
    image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_cv is None:
        logging.error(f"Failed to load image for post-processing: {image_path}")
        return Image.open(image_path)
    
    # GFPGANer expects BGR image, process it
    _, _, restored_image = gfpganer.enhance(image_cv, has_aligned=False, only_center_face=False, paste_back=True)
    # Convert back to PIL Image (from BGR to RGB)
    restored_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(restored_image)

def generate_avatar(avatar_index: int):
    avatar_folder = os.path.join(output_dir, f"avatar_{avatar_index}")
    os.makedirs(avatar_folder, exist_ok=True)
    
    for view in VIEWS:
        ethnicity = random.choice(ethnicities)
        logging.info(f"Generating {view} view for avatar {avatar_index} ({ethnicity})")
        best_quality = -1
        best_image = None

        # Ensemble generation: create several samples and select the best one
        for i in range(ensemble_size):
            prompt = generate_prompt(view, ethnicity)
            logging.info(f"Sample {i+1}/{ensemble_size} with prompt: {prompt}")
            try:
                # Use enhanced inference parameters for better quality
                result = pipeline(prompt, num_inference_steps=75, guidance_scale=8.0)
                image = result.images[0]
                quality = calculate_quality_metric(image)
                logging.info(f"Sample {i+1} quality (sharpness metric): {quality:.2f}")
                # Log quality metric for continuous feedback
                feedback_log.write(f"Avatar {avatar_index} view {view} sample {i+1}: {quality:.2f}\n")
                if quality > best_quality:
                    best_quality = quality
                    best_image = image
            except Exception as e:
                logging.error(f"Error during sample {i+1} generation for avatar {avatar_index} {view}: {e}")

        if best_image is None:
            logging.error(f"No valid image generated for avatar {avatar_index} view {view}. Skipping.")
            continue

        # Save the best image temporarily
        temp_path = os.path.join(avatar_folder, f"{view}_temp.png")
        best_image.save(temp_path)
        logging.info(f"Best sample saved temporarily to {temp_path}")

        # Post-process the image using GFPGAN
        try:
            enhanced_image = post_process_image(temp_path)
            final_path = os.path.join(avatar_folder, f"{view}.png")
            enhanced_image.save(final_path)
            logging.info(f"Post-processed image saved as {final_path}")
        except Exception as e:
            logging.error(f"Post-processing failed for avatar {avatar_index} view {view}: {e}")
            final_path = temp_path  # fallback

        # Upload final image to S3
        try:
            s3_key = f"avatars/avatar_{avatar_index}/{view}.png"
            s3_client.upload_file(final_path, AWS_STORAGE_BUCKET_NAME, s3_key)
            logging.info(f"Uploaded {final_path} to s3://{AWS_STORAGE_BUCKET_NAME}/{s3_key}")
        except Exception as e:
            logging.error(f"Failed to upload {final_path} to S3: {e}")

if __name__ == "__main__":
    try:
        for i in range(1, NUM_AVATARS + 1):
            logging.info(f"Starting generation for avatar {i}")
            generate_avatar(i)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    finally:
        feedback_log.close()
EOF
