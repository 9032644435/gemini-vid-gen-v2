from flask import Flask, render_template, request, jsonify
import os
import uuid
from google.cloud import firestore
from google.cloud import tasks_v2
import logging
import json # Added as per instructions for task payload
import base64
from google.oauth2 import id_token
from google.auth.transport import requests as auth_requests
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform, storage
import requests

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "gemini-vid-gen-v2")
REGION = os.getenv("GOOGLE_REGION", "us-central1")
FIRESTORE_COLLECTION = "video-generations"
TASK_QUEUE = "video-gen-queue"
TASK_SPN = os.getenv("TASK_WORKER_SA_EMAIL", "video-gen-worker-sa@gemini-vid-gen-v2.iam.gserviceaccount.com")
BUCKET_NAME = os.getenv("GOOGLE_BUCKET_NAME", f"{PROJECT_ID}-video-outputs")

VIDEO_MODEL_ID = "veo-3.1-generate-preview"

# --- Client Initialization ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
try:
    db = firestore.Client(project=PROJECT_ID)
    tasks_client = tasks_v2.CloudTasksClient()
    aiplatform.init(project=PROJECT_ID, location=REGION)
    storage_client = storage.Client()
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.error(f"Error initializing clients: {e}")
    db = None
    tasks_client = None
    aiplatform = None
    storage_client = None


# --- Helper Function for Dynamic URL ---
def get_cloud_run_url():
    """
    Gets the public URL of the currently running Cloud Run service.
    Prioritizes SERVICE_URL env var, then falls back to metadata server.
    """
    service_url = os.getenv('SERVICE_URL')
    if service_url:
        logging.info(f"Using service URL from environment variable: {service_url}")
        return service_url if service_url.endswith('/') else service_url + '/'

    logging.info("SERVICE_URL not set, attempting to fetch from metadata server.")
    try:
        metadata_server_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_url"
        metadata_response = requests.get(metadata_server_url, headers={"Metadata-Flavor": "Google"})
        metadata_response.raise_for_status()
        service_url = metadata_response.text
        logging.info(f"Detected Cloud Run service URL from metadata: {service_url}")
        return service_url if service_url.endswith('/') else service_url + '/'
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not fetch Cloud Run URL from metadata server: {e}")
        return None

@app.route('/')
def index():
    # Assuming index.html is in templates/ - Jules needs to confirm/create this
    return render_template('index.html') # Added by previous instruction

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
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        # Get new parameters
        duration = data.get('duration')
        num_videos = data.get('num_videos', 1) # Default to 1 if not provided

        # --- Validation ---
        if not prompt:
            return jsonify({'error': 'Missing prompt'}), 400
        if aspect_ratio not in ['16:9', '9:16']:
             return jsonify({'error': 'Invalid aspect_ratio'}), 400
        try:
            duration = int(duration)
            if duration <= 0:
                 raise ValueError("Duration must be positive")
        except (ValueError, TypeError):
             return jsonify({'error': 'Invalid duration. Must be a positive integer.'}), 400
        try:
             num_videos = int(num_videos)
             if not 1 <= num_videos <= 4:
                 raise ValueError("Number of videos must be between 1 and 4")
        except (ValueError, TypeError):
              return jsonify({'error': 'Invalid num_videos. Must be an integer between 1 and 4.'}), 400
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

        worker_endpoint = target_url + 'api/run-task'

        task_payload = {
            'job_id': job_id,
            'prompt': prompt,
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
                    'audience': target_url
                },
                'headers': {'Content-type': 'application/json'},
                'body': json.dumps(task_payload).encode('utf-8')
            }
        }

        logging.info(f"Creating task for job {job_id} targeting {worker_endpoint} with payload: {task_payload}")
        task_parent = tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE) # Corrected parent usage
        task_response = tasks_client.create_task(parent=task_parent, task=task)
        logging.info(f"Task {task_response.name} created successfully for job {job_id}.")

        return jsonify({'job_id': job_id, 'status': 'pending'}), 202

    except Exception as e:
        job_id_local = locals().get('job_id', 'unknown') # Safer way to get job_id if defined
        logging.exception(f"Error during task creation for job {job_id_local}: {e}")
        if 'doc_ref' in locals(): # Check if doc_ref was defined before error
            try:
                error_message = f'Failed to create generation task: {type(e).__name__} - {str(e)}'
                doc_ref.update({'status': 'failed', 'error': error_message})
            except Exception as db_e:
                logging.error(f"Failed to update Firestore status to failed for job {job_id_local}: {db_e}")
        return jsonify({'error': f'Failed to create generation task: {str(e)}'}), 500


@app.route('/api/check-status/<job_id>')
def check_status(job_id):
    """
    Checks the status of a video generation job.
    - Queries Firestore for the job document.
    - Returns the job status and video URL if complete.
    """
    if not db:
         logging.error("Firestore client not initialized during check_status request.")
         return jsonify({'error': 'Server configuration error: database not available'}), 500
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return jsonify(doc.to_dict())
        else:
            return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        logging.exception(f"Error checking status for job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-task', methods=['POST'])
def run_task():
    """
    Worker endpoint triggered by Cloud Tasks to generate a video.
    """
    # Verify OIDC token
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.error("Missing or invalid Authorization header")
            return "Unauthorized", 401

        token = auth_header.split(' ')[1]
        # Use google.auth.transport.requests for the request object
        request_session = auth_requests.Request()
        # It's better to get audience dynamically if possible, but env var is fallback
        audience = os.getenv('SERVICE_URL')
        if not audience: # Fallback if env var not set (should not happen in Cloud Run)
             audience = get_cloud_run_url() # This might fail again if metadata issue persists
        if not audience:
             logging.error("Could not determine audience for OIDC validation")
             return "Configuration error: Cannot determine audience", 500

        # Remove trailing slash for audience validation
        audience = audience.rstrip('/')

        logging.info(f"Verifying OIDC token for audience: {audience}")
        decoded_token = id_token.verify_oauth2_token(token, request_session, audience=audience)
        logging.info(f"Token verified for email: {decoded_token.get('email')}")

        if decoded_token.get('email') != TASK_SPN:
            logging.error(f"Token email {decoded_token.get('email')} does not match expected SA {TASK_SPN}")
            return "Forbidden", 403

    except Exception as e:
        logging.exception(f"OIDC token verification failed: {e}")
        return f"Unauthorized: {e}", 401


    # Get payload
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')
        num_videos = data.get('num_videos')

        if not all([job_id, prompt, aspect_ratio, duration, num_videos]):
            logging.error(f"Missing required data in task payload for job {job_id}")
            return "Bad Request: Missing data", 400

    except Exception as e:
        logging.exception("Failed to parse request JSON")
        return "Bad Request: Invalid JSON", 400

    doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
    try:
        # Update Firestore status to 'processing'
        doc_ref.update({'status': 'processing', 'updated_at': firestore.SERVER_TIMESTAMP})
        logging.info(f"Job {job_id} status updated to processing.")

        # --- Start AI Generation ---
        logging.info(f"Generating video for job {job_id} with prompt: '{prompt}'")

        model = GenerativeModel(VIDEO_MODEL_ID)
        generation_params = {
            "durationSeconds": duration,
            "aspectRatio": aspect_ratio,
            "sampleCount": num_videos
        }

        logging.info(f"Calling Veo model for job {job_id} with params: {generation_params}")
        video_response = model.generate_content(
            [prompt],
            generation_config=generation_params
        )
        logging.info(f"Veo model returned for job {job_id}.")

        # --- End AI Generation ---
        # Process and upload each generated video
        video_urls = []
        if not storage_client:
            raise Exception("Storage client not initialized")
        bucket = storage_client.bucket(BUCKET_NAME)

        # --- Placeholder result handling - NEEDS VERIFICATION ---
        logging.warning(f"Using MOCK video data for job {job_id}. Needs real result handling from Veo SDK response.")
        import time
        time.sleep(10) # Simulate AI processing time
        dummy_video_bytes = b"fake video data from mock"
        video_base64 = base64.b64encode(dummy_video_bytes).decode('utf-8')

        for i in range(num_videos):
            video_bytes = base64.b64decode(video_base64)
            blob_name = f"{job_id}-{i}.mp4"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(video_bytes, content_type='video/mp4')
            blob.make_public()
            video_urls.append(blob.public_url)
            logging.info(f"Video {i} for job {job_id} uploaded to {blob.public_url}")
        # --- End Placeholder ---

        # Update Firestore status to 'complete'
        final_status = {
            'status': 'complete',
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        # Store a single URL or a list of URLs
        if len(video_urls) == 1:
            final_status['video_url'] = video_urls[0]
        else:
            final_status['video_urls'] = video_urls

        doc_ref.update(final_status)
        logging.info(f"Job {job_id} completed successfully.")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        logging.exception(f"Error during video generation/upload for job {job_id}: {e}")
        try:
            # Add error details from the exception
            error_details = f"{type(e).__name__}: {str(e)}"
            doc_ref.update({
                'status': 'failed',
                'error': error_details,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
        except Exception as db_e:
            logging.error(f"Failed to update Firestore status to failed for job {job_id}: {db_e}")
        # Return error details in response for easier debugging in Cloud Tasks logs
        return f"Internal Server Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
