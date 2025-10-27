from flask import Flask, render_template, request, jsonify
import os
import uuid
from google.cloud import firestore
from google.cloud import tasks_v2
import logging
import json
import base64
import time # For polling delay

# Standard imports needed
from google.oauth2 import id_token
from google.auth.transport import requests as auth_requests
from google.cloud import storage
import requests
import google.auth # For getting access token for REST API

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "gemini-vid-gen-v2")
REGION = os.getenv("GOOGLE_REGION", "us-central1") # Ensure this is where model is available
FIRESTORE_COLLECTION = "video-generations"
TASK_QUEUE = "video-gen-queue"
TASK_SPN = os.getenv("TASK_WORKER_SA_EMAIL", "video-gen-worker-sa@gemini-vid-gen-v2.iam.gserviceaccount.com")
BUCKET_NAME = os.getenv("GOOGLE_BUCKET_NAME", f"{PROJECT_ID}-video-outputs")
VIDEO_MODEL_ID = "veo-3.1-generate-preview" # Check if REST uses full ID or just name

# API Endpoint base
# Format: https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{VIDEO_MODEL_ID}
API_ENDPOINT_BASE = f"https://{REGION}-aiplatform.googleapis.com/v1"
PREDICT_URL = f"{API_ENDPOINT_BASE}/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{VIDEO_MODEL_ID}:predictLongRunning"
FETCH_OP_URL = f"{API_ENDPOINT_BASE}/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{VIDEO_MODEL_ID}:fetchPredictOperation"

# --- Client Initialization ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
db = None
tasks_client = None
storage_client = None
try:
    db = firestore.Client(project=PROJECT_ID)
    tasks_client = tasks_v2.CloudTasksClient()
    storage_client = storage.Client()
    # aiplatform.init is not strictly needed if only using REST
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.critical(f"FATAL: Error initializing Google Cloud clients: {e}")

# --- Helper Function for Dynamic URL ---
def get_cloud_run_url():
    service_url = os.getenv('SERVICE_URL')
    if service_url:
        logging.info(f"Using service URL from environment variable: {service_url}")
        return service_url if service_url.endswith('/') else service_url + '/'
    logging.warning("SERVICE_URL env var not found, falling back to metadata server.")
    try:
        metadata_server_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_url"
        metadata_response = requests.get(metadata_server_url, headers={"Metadata-Flavor": "Google"}, timeout=5)
        metadata_response.raise_for_status()
        service_url = metadata_response.text
        logging.info(f"Detected Cloud Run service URL from metadata: {service_url}")
        return service_url if service_url.endswith('/') else service_url + '/'
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not fetch Cloud Run URL from metadata server: {e}")
        return None

# --- Helper for getting Auth Token ---
def get_auth_token():
    try:
        credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as e:
        logging.error(f"Failed to get auth token: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    if not db or not tasks_client:
         logging.error("Clients not initialized.")
         return jsonify({'error': 'Server config error'}), 500

    job_id = None
    doc_ref = None
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'Invalid JSON'}), 400

        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = data.get('duration')
        num_videos = data.get('num_videos', 1)

        # Validation (keep as before)
        if not prompt: return jsonify({'error': 'Missing prompt'}), 400
        if aspect_ratio not in ['16:9', '9:16']: return jsonify({'error': 'Invalid aspect_ratio'}), 400
        try:
            duration = int(duration)
            if not (1 <= duration <= 10): raise ValueError("Duration invalid")
        except: return jsonify({'error': 'Invalid duration.'}), 400
        try:
             num_videos = int(num_videos)
             if not (1 <= num_videos <= 4): raise ValueError("Num videos invalid")
        except: return jsonify({'error': 'Invalid num_videos.'}), 400

        job_id = str(uuid.uuid4())
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        # Define the GCS output path *before* creating the task
        storage_uri = f"gs://{BUCKET_NAME}/{job_id}/" # Needs trailing slash

        doc_ref.set({
            'prompt': prompt, 'aspect_ratio': aspect_ratio, 'duration': duration,
            'num_videos': num_videos, 'status': 'pending', 'job_id': job_id,
            'storage_uri_requested': storage_uri, # Store for reference
            'created_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} created in Firestore.")

        target_url = get_cloud_run_url()
        if not target_url:
             logging.error("Failed to get Cloud Run service URL.")
             doc_ref.update({'status': 'failed', 'error': 'Server config error: No service URL'})
             return jsonify({'error': 'Server config error: No service URL'}), 500

        worker_endpoint = target_url + 'api/run-task'
        task_payload = { # Pass only what the worker needs to *start* the AI job
            'job_id': job_id,
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'duration': duration,
            'num_videos': num_videos,
            'storageUri': storage_uri # Pass GCS URI to worker
        }
        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': worker_endpoint,
                'oidc_token': { 'service_account_email': TASK_SPN, 'audience': target_url },
                'headers': {'Content-type': 'application/json'},
                'body': json.dumps(task_payload).encode('utf-8')
            }
        }

        logging.info(f"Creating task for job {job_id} targeting {worker_endpoint}")
        task_parent = tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE)
        task_response = tasks_client.create_task(parent=task_parent, task=task)
        logging.info(f"Task {task_response.name} created for job {job_id}.")

        return jsonify({'job_id': job_id, 'status': 'pending'}), 202

    except Exception as e:
        job_id_local = job_id if job_id else 'unknown'
        logging.exception(f"Error in /api/generate-video for job {job_id_local}: {e}")
        if doc_ref:
            try:
                error_message = f'Failed to create task: {type(e).__name__}'
                doc_ref.update({'status': 'failed', 'error': error_message})
            except Exception as db_e:
                logging.error(f"Also failed to update Firestore for job {job_id_local}: {db_e}")
        return jsonify({'error': f'Failed to create task: {str(e)}'}), 500


@app.route('/api/check-status/<job_id>')
def check_status(job_id):
    """
    Checks the status. If 'processing_ai', polls the AI operation.
    """
    if not db:
        logging.error("DB client not available.")
        return jsonify({'error': 'Server configuration error'}), 500

    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Job not found'}), 404

        job_data = doc.to_dict()
        current_status = job_data.get('status')
        operation_name = job_data.get('operation_name')

        # If status indicates AI processing, poll the AI operation status
        if current_status == 'processing_ai' and operation_name:
            logging.info(f"Polling AI operation for job {job_id}: {operation_name}")
            token = get_auth_token()
            if not token:
                # Keep status as processing_ai, error logged in get_auth_token
                logging.error(f"Could not get auth token to poll operation for job {job_id}")
                return jsonify(job_data), 200 # Return current data, don't change status

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            poll_payload = {"operationName": operation_name}

            try:
                poll_response = requests.post(FETCH_OP_URL, headers=headers, json=poll_payload, timeout=10)
                poll_response.raise_for_status() # Raise exception for bad status codes
                op_data = poll_response.json()

                if op_data.get('done'):
                    logging.info(f"AI operation DONE for job {job_id}.")
                    # Process the result from op_data['response']
                    ai_response_data = op_data.get('response', {})
                    video_outputs = ai_response_data.get('videos', [])
                    video_urls = []

                    if not video_outputs:
                         logging.error(f"AI operation done but no 'videos' field in response for job {job_id}. Response: {ai_response_data}")
                         raise Exception("AI processing finished but returned no video outputs.")

                    for i, output in enumerate(video_outputs):
                         gcs_uri = output.get('gcsUri')
                         if not gcs_uri:
                             logging.warning(f"Output {i} for job {job_id} missing gcsUri. Skipping.")
                             continue
                         # Convert gs:// URI to public https:// URL
                         if gcs_uri.startswith("gs://"):
                             try:
                                 bucket_name_from_uri = gcs_uri.split('/')[2]
                                 object_name = '/'.join(gcs_uri.split('/')[3:])
                                 # We need storage client here
                                 if not storage_client: raise Exception("Storage client not available")
                                 bucket = storage_client.bucket(bucket_name_from_uri)
                                 blob = bucket.blob(object_name)
                                 if blob.exists():
                                     # Assuming objects in the target bucket are publicly readable
                                     public_url = blob.public_url
                                     video_urls.append(public_url)
                                     logging.info(f"Processed GCS URI for job {job_id}: {gcs_uri} -> {public_url}")
                                 else:
                                      logging.warning(f"Blob {object_name} not found for job {job_id} despite AI success.")
                             except Exception as url_err:
                                  logging.error(f"Error converting GCS URI {gcs_uri} to URL for job {job_id}: {url_err}")
                         else:
                              logging.warning(f"Output {i} for job {job_id} has unexpected URI format: {gcs_uri}")

                    if not video_urls:
                         raise Exception("AI processing finished but no valid video URLs could be constructed.")

                    # Update Firestore to 'complete'
                    final_status_update = {
                        'status': 'complete',
                        'updated_at': firestore.SERVER_TIMESTAMP
                    }
                    if len(video_urls) == 1:
                        final_status_update['video_url'] = video_urls[0]
                    else:
                        final_status_update['video_urls'] = video_urls
                    doc_ref.update(final_status_update)
                    logging.info(f"Job {job_id} successfully completed.")
                    # Return the updated data immediately
                    job_data.update(final_status_update)
                    return jsonify(job_data), 200
                else:
                    # Operation still running, just return current status
                    logging.info(f"AI operation still running for job {job_id}.")
                    return jsonify(job_data), 200

            except requests.exceptions.RequestException as poll_err:
                 logging.error(f"Failed to poll AI operation status for job {job_id}: {poll_err}")
                 # Optionally update status to failed here, or let it retry on next poll
                 # doc_ref.update({'status': 'failed', 'error': 'Failed to poll AI status'})
                 # return jsonify({'status': 'failed', 'error': 'Failed to poll AI status'}), 500
                 return jsonify(job_data), 200 # Return current data, error logged

        else:
            # Status is not 'processing_ai', just return current data
            return jsonify(job_data), 200

    except Exception as e:
        logging.exception(f"Error in check_status for job {job_id}: {e}")
        # Attempt to update status to failed if possible
        try:
             if doc_ref: doc_ref.update({'status': 'failed', 'error': f'Check status failed: {str(e)}'})
        except: pass # Ignore errors updating status during another error
        return jsonify({'error': 'Failed to check job status'}), 500


@app.route('/api/run-task', methods=['POST'])
def run_task():
    """ Worker endpoint: Calls the AI model via REST and stores the operation name. """
    if not db:
        logging.error("DB client not available.")
        return "Server configuration error", 500

    # OIDC Verification (Keep as before)
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '): return "Unauthorized", 401
        token = auth_header.split(' ')[1]
        request_session = auth_requests.Request()
        audience = os.getenv('SERVICE_URL')
        if not audience: audience = get_cloud_run_url()
        if not audience: return "Config error: No audience", 500
        audience = audience.rstrip('/')
        logging.info(f"Verifying OIDC token for audience: {audience}")
        decoded_token = id_token.verify_oauth2_token(token, request_session, audience=audience)
        if decoded_token.get('email') != TASK_SPN: return "Forbidden", 403
        logging.info(f"Token verified for email: {decoded_token.get('email')}")
    except Exception as e:
        logging.exception(f"OIDC verification failed: {e}")
        return f"Unauthorized: {e}", 401

    # Get payload
    job_id = None
    doc_ref = None
    try:
        data = request.get_json()
        if not data: return "Bad Request: Invalid JSON", 400
        job_id = data.get('job_id')
        prompt = data.get('prompt')
        aspect_ratio = data.get('aspect_ratio')
        duration = int(data.get('duration'))
        num_videos = int(data.get('num_videos'))
        storage_uri = data.get('storageUri') # Get GCS path from payload
        if not all([job_id, prompt, aspect_ratio, duration, num_videos, storage_uri]):
            return "Bad Request: Missing data", 400
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
    except Exception as e:
        logging.exception(f"Payload parsing failed for job {job_id or 'UNKNOWN'}")
        return "Bad Request", 400

    # --- Call AI via REST API ---
    try:
        doc_ref.update({'status': 'calling_ai', 'updated_at': firestore.SERVER_TIMESTAMP})
        logging.info(f"Job {job_id} status updated to calling_ai.")

        token = get_auth_token()
        if not token:
             raise Exception("Failed to get auth token for AI call")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        # Construct payload according to REST API docs
        instances = [{"prompt": prompt}]
        parameters = {
            "durationSeconds": duration,
            "aspectRatio": aspect_ratio,
            "sampleCount": num_videos,
            "storageUri": storage_uri # Tell API where to save output
            # Add other params like 'generateAudio': True if needed/supported
        }
        request_body = {"instances": instances, "parameters": parameters}

        logging.info(f"Calling predictLongRunning for job {job_id}...")
        response = requests.post(PREDICT_URL, headers=headers, json=request_body, timeout=30) # Timeout for initiating call
        response.raise_for_status() # Raise exception for bad status codes
        response_data = response.json()

        operation_name = response_data.get('name')
        if not operation_name:
             raise Exception("predictLongRunning response did not contain operation 'name'")

        logging.info(f"AI operation started for job {job_id}: {operation_name}")

        # Update Firestore with operation name and new status
        doc_ref.update({
            'status': 'processing_ai', # New status indicating polling is needed
            'operation_name': operation_name,
            'updated_at': firestore.SERVER_TIMESTAMP
        })

        return jsonify({"status": "success"}), 200 # OK to Cloud Tasks

    except Exception as e:
        logging.exception(f"Error in run_task AI call/update for job {job_id}: {e}")
        if doc_ref:
             try:
                 error_details = f"{type(e).__name__}: {str(e)}"
                 doc_ref.update({'status': 'failed', 'error': error_details, 'updated_at': firestore.SERVER_TIMESTAMP})
             except Exception as db_e:
                 logging.error(f"Also failed to update Firestore after error for job {job_id}: {db_e}")
        return f"Internal Server Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
