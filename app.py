from flask import Flask, render_template, request, jsonify
import os
import uuid
from google.cloud import firestore
from google.cloud import tasks_v2
import logging
import json
import base64
import time # For placeholder delay/simulated work

# Corrected imports for OIDC and AI models
from google.oauth2 import id_token
from google.auth.transport import requests as auth_requests
# Use the specific video classes
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
    # Initialize aiplatform without specifying project/location here if ADC works
    aiplatform.init(project=PROJECT_ID, location=REGION)
    storage_client = storage.Client()
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.critical(f"FATAL: Error initializing Google Cloud clients: {e}")
    # Consider how to handle failure - maybe exit or disable endpoints


# --- Helper Function for Dynamic URL ---
def get_cloud_run_url():
    """
    Gets the public URL of the currently running Cloud Run service.
    Prioritizes SERVICE_URL env var, then falls back to metadata server.
    """
    service_url = os.getenv('SERVICE_URL')
    if service_url:
        logging.info(f"Using service URL from environment variable: {service_url}")
        # Ensure URL always ends with a slash
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
    # Make sure 'templates/index.html' exists
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

    job_id = None # Initialize job_id outside try block for error logging
    doc_ref = None
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'Invalid JSON payload'}), 400

        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')
        num_videos = data.get('num_videos', 1)

        # Validation
        if not prompt: return jsonify({'error': 'Missing prompt'}), 400
        if aspect_ratio not in ['16:9', '9:16']: return jsonify({'error': 'Invalid aspect_ratio'}), 400
        try:
            duration = int(duration)
            # Add specific valid durations if known for the model (e.g., 4, 6, 8, 10 for Veo 3.1)
            if duration not in [4, 6, 8, 10]: raise ValueError("Duration invalid for Veo 3.1")
        except: return jsonify({'error': 'Invalid duration.'}), 400
        try:
             num_videos = int(num_videos)
             if not (1 <= num_videos <= 4): raise ValueError("Num videos invalid")
        except: return jsonify({'error': 'Invalid num_videos.'}), 400

        job_id = str(uuid.uuid4())
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc_ref.set({
            'prompt': prompt, 'aspect_ratio': aspect_ratio, 'duration': duration,
            'num_videos': num_videos, 'status': 'pending', 'job_id': job_id,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} created in Firestore.")

        target_url = get_cloud_run_url()
        if not target_url:
             logging.error("Failed to get Cloud Run service URL for task creation.")
             doc_ref.update({'status': 'failed', 'error': 'Server config error: No service URL'})
             return jsonify({'error': 'Server config error: No service URL'}), 500

        worker_endpoint = target_url + 'api/run-task' # Append endpoint path
        task_payload = {
            'job_id': job_id, 'prompt': prompt, 'aspect_ratio': aspect_ratio,
            'duration': duration, 'num_videos': num_videos
        }
        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': worker_endpoint,
                'oidc_token': {
                    'service_account_email': TASK_SPN,
                    'audience': target_url # Root URL is the audience
                },
                'headers': {'Content-type': 'application/json'},
                'body': json.dumps(task_payload).encode('utf-8')
            }
        }

        logging.info(f"Creating task for job {job_id} targeting {worker_endpoint}")
        task_parent = tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE)
        task_response = tasks_client.create_task(parent=task_parent, task=task)
        logging.info(f"Task {task_response.name} created successfully for job {job_id}.")

        return jsonify({'job_id': job_id, 'status': 'pending'}), 202 # Accepted

    except Exception as e:
        job_id_local = job_id if job_id else 'unknown'
        logging.exception(f"Error in /api/generate-video for job {job_id_local}: {e}") # Log full traceback
        if doc_ref: # Check if doc_ref was assigned
            try:
                error_message = f'Failed to create task: {type(e).__name__}'
                doc_ref.update({'status': 'failed', 'error': error_message})
            except Exception as db_e:
                logging.error(f"Also failed to update Firestore for job {job_id_local}: {db_e}")
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
        return "Server configuration error", 500

    # Verify OIDC token first
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.error("Missing or invalid Authorization header in task request")
            return "Unauthorized", 401

        token = auth_header.split(' ')[1]
        request_session = auth_requests.Request()
        audience = os.getenv('SERVICE_URL')
        if not audience: audience = get_cloud_run_url() # Fallback
        if not audience:
             logging.error("Could not determine audience for OIDC validation")
             return "Configuration error", 500

        audience = audience.rstrip('/') # Apply the fix

        logging.info(f"Verifying OIDC token for audience: {audience}")
        decoded_token = id_token.verify_oauth2_token(token, request_session, audience=audience)
        logging.info(f"Token verified for email: {decoded_token.get('email')}")

        if decoded_token.get('email') != TASK_SPN:
            logging.error(f"Token email mismatch: {decoded_token.get('email')} vs {TASK_SPN}")
            return "Forbidden", 403

    except Exception as e:
        logging.exception(f"OIDC token verification failed: {e}")
        return f"Unauthorized: {e}", 401

    # If OIDC passes, proceed
    job_id = None
    doc_ref = None
    try:
        data = request.get_json()
        if not data: return "Bad Request: Invalid JSON", 400

        job_id = data.get('job_id')
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')
        num_videos = data.get('num_videos')

        if not all([job_id, prompt, aspect_ratio, duration, num_videos]):
            logging.error(f"Missing data in task payload for job {job_id or 'UNKNOWN'}: {data}")
            return "Bad Request: Missing data", 400

        # Ensure types are correct
        duration = int(duration)
        num_videos = int(num_videos)

        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id) # Get doc_ref early

    except Exception as e:
        logging.exception(f"Failed to parse request JSON for job {job_id or 'UNKNOWN'}")
        return "Bad Request: Invalid JSON", 400

    # --- Main Processing Logic ---
    try:
        doc_ref.update({'status': 'processing', 'updated_at': firestore.SERVER_TIMESTAMP})
        logging.info(f"Job {job_id} status updated to processing.")

        # --- CORRECTED AI Generation Call ---
        logging.info(f"Generating video for job {job_id} with prompt: '{prompt}'")

        # Use the specific VideoGenerationModel
        model = VideoGenerationModel.from_pretrained(VIDEO_MODEL_ID)

        # Use the specific GenerateVideosConfig
        config = GenerateVideosConfig(
            generation_length_secs=duration,
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
        try:
             # GUESSING response structure - adjust based on actual SDK output!
             outputs_to_process = []
             # Try accessing the response data; this structure might change!
             if hasattr(video_response, '_raw_response') and video_response._raw_response and hasattr(video_response._raw_response, 'video_outputs'):
                 outputs_to_process = video_response._raw_response.video_outputs
                 logging.info(f"Found {len(outputs_to_process)} video outputs in raw response.")
             elif isinstance(video_response, list): # Check if it's already a list of outputs
                 outputs_to_process = video_response
                 logging.info(f"Response is a list with {len(outputs_to_process)} items.")
             else:
                  logging.warning(f"Response structure not recognized for job {job_id}. Cannot process result.")
                  logging.info(f"Raw response type: {type(video_response)}")
                  # logging.info(f"Raw response dir: {dir(video_response)}") # Be careful logging potentially large objects
                  raise Exception("Video generation response format not recognized or empty.")


             for i, output in enumerate(outputs_to_process):
                 video_base64 = None
                 # Try common attribute names for base64 data
                 if hasattr(output, 'bytes_base64_encoded'):
                     video_base64 = output.bytes_base64_encoded
                 elif hasattr(output, 'base64_content'): # Another possible name
                     video_base64 = output.base64_content
                 elif isinstance(output, str): # Maybe it's just a list of base64 strings?
                     video_base64 = output
                 # Add more checks if needed based on actual response

                 if not video_base64:
                      logging.warning(f"Video data in output {i} for job {job_id} is empty or not found. Skipping.")
                      continue

                 video_bytes = base64.b64decode(video_base64)
                 if not video_bytes:
                      logging.warning(f"Decoded video bytes for output {i} job {job_id} are empty. Skipping.")
                      continue

                 blob_name = f"{job_id}-{i}.mp4" # Suffix if multiple videos
                 blob = bucket.blob(blob_name)
                 blob.upload_from_string(video_bytes, content_type='video/mp4')
                 blob.make_public() # Simple access control
                 video_urls.append(blob.public_url)
                 logging.info(f"Video {i} for job {job_id} uploaded to {blob.public_url}")

             if not video_urls:
                  raise Exception("No videos were successfully processed or uploaded from response.")

        except Exception as proc_err:
             logging.error(f"Failed processing/uploading Veo response for job {job_id}: {proc_err}")
             raise Exception(f"Failed to process/upload Veo response: {proc_err}") from proc_err
        # --- End Process and Upload ---


        # Update Firestore status to 'complete'
        final_status = {
            'status': 'complete',
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        if len(video_urls) == 1:
            final_status['video_url'] = video_urls[0] # Use 'video_url' for single video
        else:
            final_status['video_urls'] = video_urls # Use 'video_urls' for multiple

        doc_ref.update(final_status)
        logging.info(f"Job {job_id} completed successfully.")

        return jsonify({"status": "success"}), 200 # OK to Cloud Tasks

    except Exception as e:
        # Log the full traceback for the exception during processing
        logging.exception(f"Core error in run_task for job {job_id}: {e}")
        if doc_ref: # Check if doc_ref was assigned before error
             try:
                 error_details = f"{type(e).__name__}: {str(e)}"
                 doc_ref.update({
                     'status': 'failed',
                     'error': error_details,
                     'updated_at': firestore.SERVER_TIMESTAMP
                 })
             except Exception as db_e:
                 logging.error(f"Also failed to update Firestore status after core error for job {job_id}: {db_e}")
        # Return 500 so Cloud Tasks might retry
        return f"Internal Server Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    # Use environment variable for debug flag, default to False
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
