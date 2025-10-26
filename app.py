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
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')

        if not prompt or not aspect_ratio:
            return jsonify({'error': 'Missing prompt or aspect_ratio'}), 400

        job_id = str(uuid.uuid4())
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc_ref.set({
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
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
            'duration': 10
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

        logging.info(f"Creating task for job {job_id} targeting {worker_endpoint}")
        task_parent = tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE)
        task_response = tasks_client.create_task(parent=task_parent, task=task)
        logging.info(f"Task {task_response.name} created successfully for job {job_id}.")

        return jsonify({'job_id': job_id, 'status': 'pending'}), 202

    except Exception as e:
        logging.exception(f"Error during task creation for job {job_id if 'job_id' in locals() else 'unknown'}: {e}")
        if 'job_id' in locals() and 'doc_ref' in locals():
            try:
                doc_ref.update({'status': 'failed', 'error': f'Task creation failed: {str(e)}'})
            except Exception as db_e:
                logging.error(f"Failed to update Firestore status to failed for job {job_id}: {db_e}")
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
        audience = get_cloud_run_url()
        if not audience:
             logging.error("Could not determine audience for OIDC validation")
             return "Configuration error", 500

        decoded_token = id_token.verify_oauth2_token(token, auth_requests.Request(), audience=audience)

        if decoded_token['email'] != TASK_SPN:
            logging.error(f"Token email {decoded_token['email']} does not match expected SA {TASK_SPN}")
            return "Forbidden", 403

    except Exception as e:
        logging.exception(f"OIDC token verification failed: {e}")
        return "Unauthorized", 401

    # Get payload
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')

        if not all([job_id, prompt, aspect_ratio, duration]):
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

        # Call Vertex AI
        model = GenerativeModel(VIDEO_MODEL_ID)
        generation_params = {
            "durationSeconds": duration, # Use camelCase matching REST API
            "aspectRatio": aspect_ratio  # Use camelCase matching REST API
        }
        logging.info(f"Calling Veo model with params: {generation_params}") # Log parameters
        video_response = model.generate_content(
            [prompt],
            generation_config=generation_params
        )

        # This assumes the first result is the one we want and it is base64 encoded
        video_bytes = base64.b64decode(video_response.candidates[0].content.parts[0].video)

        # Upload to GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{job_id}.mp4")
        blob.upload_from_string(video_bytes, content_type='video/mp4')
        blob.make_public()
        video_url = blob.public_url
        logging.info(f"Video for job {job_id} uploaded to {video_url}")

        # Update Firestore status to 'complete'
        doc_ref.update({
            'status': 'complete',
            'video_url': video_url,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} completed successfully.")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        logging.exception(f"Error during video generation for job {job_id}: {e}")
        try:
            doc_ref.update({
                'status': 'failed',
                'error': str(e),
                'updated_at': firestore.SERVER_TIMESTAMP
            })
        except Exception as db_e:
            logging.error(f"Failed to update Firestore status to failed for job {job_id}: {db_e}")
        return "Internal Server Error", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
