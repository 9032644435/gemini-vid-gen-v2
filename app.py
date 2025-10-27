from flask import Flask, render_template, request, jsonify
import os
import uuid
from google.cloud import firestore
from google.cloud import tasks_v2
import logging
import json
import base64
import time # For placeholder delay

# Corrected imports for OIDC and AI models
from google.oauth2 import id_token
from google.auth.transport import requests as auth_requests
from vertexai.vision_models import VideoGenerationModel, GenerateVideosConfig # Correct Video Classes
from google.cloud import aiplatform, storage
import requests

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "gemini-vid-gen-v2")
REGION = os.getenv("GOOGLE_REGION", "us-central1")
FIRESTORE_COLLECTION = "video-generations"
TASK_QUEUE = "video-gen-queue"
TASK_SPN = os.getenv("TASK_WORKER_SA_EMAIL", "video-gen-worker-sa@gemini-vid-gen-v2.iam.gserviceaccount.com")
BUCKET_NAME = os.getenv("GOOGLE_BUCKET_NAME", f"{PROJECT_ID}-video-outputs")

VIDEO_MODEL_ID = "veo-3.1-generate-preview" # Correct Model ID

# --- Client Initialization ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
db = None
tasks_client = None
storage_client = None
try:
    db = firestore.Client(project=PROJECT_ID)
    tasks_client = tasks_v2.CloudTasksClient()
    aiplatform.init(project=PROJECT_ID, location=REGION) # Keep aiplatform init
    storage_client = storage.Client()
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.critical(f"FATAL: Error initializing Google Cloud clients: {e}") # Use critical for startup errors
    # Prevent app from starting cleanly if clients fail
    # Depending on deployment, might cause restarts, which is desired


# --- Helper Function for Dynamic URL ---
def get_cloud_run_url():
    """
    Gets the public URL of the currently running Cloud Run service.
    Prioritizes SERVICE_URL env var, then falls back to metadata server.
    """
    service_url = os.getenv('SERVICE_URL')
    if service_url:
        logging.info(f"Using service URL from environment variable: {service_url}")
        # Ensure URL always ends with a slash for easy endpoint appending elsewhere
        return service_url if service_url.endswith('/') else service_url + '/'

    logging.warning("SERVICE_URL env var not found, attempting to fetch from metadata server.")
    try:
        metadata_server_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_url"
        # Add a timeout to prevent hanging
        metadata_response = requests.get(metadata_server_url, headers={"Metadata-Flavor": "Google"}, timeout=5)
        metadata_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        service_url = metadata_response.text
        logging.info(f"Detected Cloud Run service URL from metadata: {service_url}")
        return service_url if service_url.endswith('/') else service_url + '/'
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not fetch Cloud Run URL from metadata server: {e}")
        return None # Return None to indicate failure

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    """
    Handles video generation requests.
    - Creates a job ID and Firestore document.
    - Creates a Cloud Task with OIDC auth to trigger the video generation process.
    - Returns the job ID to the client.
    """
    if not db or not tasks_client:
         logging.error("Clients not initialized during generate_video request.")
         return jsonify({'error': 'Server configuration error: clients not available'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')
        num_videos = data.get('num_videos', 1)

        # --- Validation ---
        if not prompt: return jsonify({'error': 'Missing prompt'}), 400
        if aspect_ratio not in ['16:9', '9:16']: return jsonify({'error': 'Invalid aspect_ratio'}), 400
        try:
            duration = int(duration)
            if not (1 <= duration <= 10): raise ValueError("Duration must be between 1 and 10") # Be more specific if API has limits
        except (ValueError, TypeError, TypeError): return jsonify({'error': 'Invalid duration.'}), 400
        try:
             num_videos = int(num_videos)
             if not (1 <= num_videos <= 4): raise ValueError("Number of videos must be between 1 and 4")
        except (ValueError, TypeError): return jsonify({'error': 'Invalid num_videos.'}), 400
        # --- End Validation ---

        job_id = str(uuid.uuid4())
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc_ref.set({
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'duration': duration,
            'num_videos': num_videos,
            'status': 'pending',
            'job_id': job_id,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} created in Firestore.")

        target_url = get_cloud_run_url()
        if not target_url:
             logging.error("Failed to get Cloud Run service URL for task creation.")
             doc_ref.update({'status': 'failed', 'error': 'Server configuration error: Could not determine service URL'})
             return jsonify({'error': 'Server configuration error: Could not determine service URL'}), 500

        worker_endpoint = target_url + 'api/run-task' # Append endpoint path

        task_payload = {
            'job_id': job_id,
            'prompt': prompt, # Pass data needed by worker
            'aspect_ratio': aspect_ratio,
            'duration': duration,
            'num_videos': num_videos
        }

        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': worker_endpoint,
                'oidc_token': {
                    'service_account_email': TASK_SPN,
                    'audience': target_url # Use root URL for audience
                },
                'headers': {'Content-type': 'application/json'},
                'body': json.dumps(task_payload).encode('utf-8')
            }
            # Optional: Add dispatch_deadline or other task settings
        }

        logging.info(f"Creating task for job {job_id} targeting {worker_endpoint}")
        task_parent = tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE)
        task_response = tasks_client.create_task(parent=task_parent, task=task)
        logging.info(f"Task {task_response.name} created successfully for job {job_id}.")

        return jsonify({'job_id': job_id, 'status': 'pending'}), 202 # 202 Accepted

    except Exception as e:
        job_id_local = locals().get('job_id', 'unknown')
        logging.exception(f"Error in /api/generate-video for job {job_id_local}: {e}") # Log full traceback
        if 'doc_ref' in locals():
            try:
                error_message = f'Failed to create task: {type(e).__name__}'
                doc_ref.update({'status': 'failed', 'error': error_message})
            except Exception as db_e:
                logging.error(f"Also failed to update Firestore status to failed for job {job_id_local}: {db_e}")
        return jsonify({'error': f'Failed to create generation task: {str(e)}'}), 500


@app.route('/api/check-status/<job_id>')
def check_status(job_id):
    """ Checks the status of a video generation job. """
    if not db:
        logging.error("Firestore client not available during check_status.")
        return jsonify({'error': 'Server configuration error'}), 500
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return jsonify(doc.to_dict())
        else:
            return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        logging.exception(f"Error checking status for job {job_id}: {e}")
        return jsonify({'error': 'Failed to check job status'}), 500


@app.route('/api/run-task', methods=['POST'])
def run_task():
    """ Worker endpoint triggered by Cloud Tasks to generate a video. """
    if not db or not storage_client: # Check required clients
        logging.error("Clients not initialized during run_task request.")
        return "Server configuration error: clients not available", 500

    # Verify OIDC token first
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.error("Missing or invalid Authorization header in task request")
            return "Unauthorized", 401

        token = auth_header.split(' ')[1]
        request_session = auth_requests.Request()
        audience = os.getenv('SERVICE_URL')
        if not audience:
             audience = get_cloud_run_url() # Fallback, but should be set in Cloud Run
        if not audience:
             logging.error("Could not determine audience for OIDC validation")
             return "Configuration error: Cannot determine audience", 500

        audience = audience.rstrip('/') # Apply the fix

        logging.info(f"Verifying OIDC token for audience: {audience}")
        decoded_token = id_token.verify_oauth2_token(token, request_session, audience=audience)
        logging.info(f"Token verified for email: {decoded_token.get('email')}")

        if decoded_token.get('email') != TASK_SPN:
            logging.error(f"Token email {decoded_token.get('email')} does not match expected SA {TASK_SPN}")
            return "Forbidden", 403

    except Exception as e:
        logging.exception(f"OIDC token verification failed: {e}")
        return f"Unauthorized: {e}", 401

    # If OIDC passes, proceed with task
    job_id = None # Initialize job_id
    try:
        data = request.get_json()
        if not data:
             return "Bad Request: Invalid JSON", 400

        job_id = data.get('job_id')
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')
        num_videos = data.get('num_videos')

        if not all([job_id, prompt, aspect_ratio, duration, num_videos]):
            logging.error(f"Missing required data in task payload for job {job_id or 'UNKNOWN'}: {data}")
            return "Bad Request: Missing data", 400

        # Ensure types are correct after getting from JSON
        duration = int(duration)
        num_videos = int(num_videos)

    except Exception as e:
        logging.exception(f"Failed to parse request JSON for job {job_id or 'UNKNOWN'}")
        return "Bad Request: Invalid JSON", 400

    # --- Main Processing Logic ---
    doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
    try:
        doc_ref.update({'status': 'processing', 'updated_at': firestore.SERVER_TIMESTAMP})
        logging.info(f"Job {job_id} status updated to processing.")

        # --- CORRECTED AI Generation Call ---
        logging.info(f"Generating video for job {job_id} with prompt: '{prompt}'")

        # Use the specific VideoGenerationModel
        model = VideoGenerationModel.from_pretrained(VIDEO_MODEL_ID)

        # Use the specific GenerateVideosConfig
        config = GenerateVideosConfig(
            generation_length_secs=duration, # Use correct parameter name
            aspect_ratio=aspect_ratio
        )
        number_of_videos_to_gen = num_videos

        logging.info(f"Calling Veo model with config: aspect_ratio={config.aspect_ratio}, duration={config.generation_length_secs}s, count={number_of_videos_to_gen}")

        # Use the generate_videos method
        # TODO: Verify exact response structure and error handling for this SDK call
        video_response = model.generate_videos(
            prompt=prompt,
            config=config,
            number_of_videos=number_of_videos_to_gen
        )
        logging.info(f"Veo model call completed for job {job_id}.")
        # --- End AI Generation Call ---

        # --- Process and Upload ---
        video_urls = []
        bucket = storage_client.bucket(BUCKET_NAME)

        # TODO: Replace placeholder logic with actual response handling
        # Example assuming response structure needs verification
        try:
             # GUESSING response structure - adjust based on actual SDK output!
             # Check if response has video data (could be in various places)
             # This part is HIGHLY LIKELY TO NEED ADJUSTMENT after first successful call

             # Simple check if the response object itself has a way to get bytes
             if hasattr(video_response, 'media') and video_response.media: # Example structure
                 outputs_to_process = video_response.media
             elif hasattr(video_response, '_raw_response') and video_response._raw_response.video_outputs:
                 outputs_to_process = video_response._raw_response.video_outputs
             else:
                  logging.warning(f"Response structure not recognized for job {job_id}. Using MOCK data.")
                  # Use mock data if extraction fails
                  outputs_to_process = [{'bytes_base64_encoded': base64.b64encode(b"fake data").decode('utf-8')}] * number_of_videos_to_gen # Mock based on count


             for i, output in enumerate(outputs_to_process):
                 # Try different ways to get base64 or bytes depending on actual structure
                 if hasattr(output, 'bytes_base64_encoded'):
                     video_base64 = output.bytes_base64_encoded
                 elif hasattr(output, 'data'): # Alternative structure
                      video_base64 = base64.b64encode(output.data).decode('utf-8')
                 else:
                      logging.warning(f"Could not find video data in output {i} for job {job_id}. Skipping.")
                      continue # Skip this output if data not found

                 if not video_base64:
                      logging.warning(f"Video data in output {i} for job {job_id} is empty. Skipping.")
                      continue

                 video_bytes = base64.b64decode(video_base64)
                 blob_name = f"{job_id}-{i}.mp4" # Suffix if multiple videos
                 blob = bucket.blob(blob_name)
                 blob.upload_from_string(video_bytes, content_type='video/mp4')
                 blob.make_public() # Simple access control
                 video_urls.append(blob.public_url)
                 logging.info(f"Video {i} for job {job_id} uploaded to {blob.public_url}")

             if not video_urls:
                  raise Exception("No videos were successfully processed or uploaded.")

        except (AttributeError, IndexError, TypeError, ValueError, Exception) as proc_err:
             logging.error(f"Failed processing/uploading Veo response for job {job_id}: {proc_err}")
             # Log raw response if possible and not too large
             # logging.error(f"Raw Veo response: {video_response}") # Be cautious with size
             raise Exception(f"Failed to process/upload Veo response: {proc_err}") from proc_err
        # --- End Process and Upload ---


        # Update Firestore status to 'complete'
        final_status = {
            'status': 'complete',
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        if len(video_urls) == 1:
            final_status['video_url'] = video_urls[0]
        else:
            final_status['video_urls'] = video_urls # Store list if multiple

        doc_ref.update(final_status)
        logging.info(f"Job {job_id} completed successfully.")

        return jsonify({"status": "success"}), 200 # OK to Cloud Tasks

    except Exception as e:
        # Log the full traceback for the exception during processing
        logging.exception(f"Core error in run_task for job {job_id}: {e}")
        try:
            error_details = f"{type(e).__name__}: {str(e)}"
            doc_ref.update({
                'status': 'failed',
                'error': error_details,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
        except Exception as db_e:
            logging.error(f"Also failed to update Firestore status to failed after core error for job {job_id}: {db_e}")
        # Return 500 so Cloud Tasks might retry
        return f"Internal Server Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
