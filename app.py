from flask import Flask, render_template, request, jsonify
import os
import uuid
from google.cloud import firestore
from google.cloud import tasks_v2
import logging
import json
import base64
import time # For polling delay
import datetime # For signed URLs

# Standard imports needed for REST API calls
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
VIDEO_MODEL_ID = "veo-3.1-generate-preview" # Model ID for REST API

# API Endpoint base
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
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.critical(f"FATAL: Error initializing Google Cloud clients: {e}")

# --- Helper Function for Dynamic URL ---
def get_cloud_run_url():
    service_url = os.getenv('SERVICE_URL')
    if service_url:
        logging.info(f"Using service URL from environment variable: {service_url}")
        return service_url if service_url.endswith('/') else service_url + '/'

    logging.warning("SERVICE_URL env var not found, attempting to fetch from metadata server.")
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
        credentials, project_id_unused = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        logging.info("Successfully obtained auth token.")
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

        # Validation
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
        storage_uri = f"gs://{BUCKET_NAME}/{job_id}/"

        doc_ref.set({
            'prompt': prompt, 'aspect_ratio': aspect_ratio, 'duration': duration,
            'num_videos': num_videos, 'status': 'pending', 'job_id': job_id,
            'storage_uri_requested': storage_uri,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} created in Firestore.")

        target_url = get_cloud_run_url()
        if not target_url:
             logging.error("Failed to get Cloud Run service URL.")
             doc_ref.update({'status': 'failed', 'error': 'Server config error: No service URL'})
             return jsonify({'error': 'Server config error: No service URL'}), 500

        worker_endpoint = target_url + 'api/run-task'
        task_payload = {
            'job_id': job_id,
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'duration': duration,
            'num_videos': num_videos,
            'storageUri': storage_uri
        }
        
        oidc_audience = target_url.rstrip('/') 

        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': worker_endpoint,
                'oidc_token': { 'service_account_email': TASK_SPN, 'audience': oidc_audience },
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
    if not db:
        logging.error("DB client not available.")
        return jsonify({'error': 'Server configuration error'}), 500

    doc_ref = None
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Job not found'}), 404

        job_data = doc.to_dict()
        current_status = job_data.get('status')
        operation_name = job_data.get('operation_name')

        if current_status == 'processing_ai' and operation_name:
            logging.info(f"Polling AI operation for job {job_id}: {operation_name}")
            token = get_auth_token()
            if not token:
                logging.error(f"Could not get auth token to poll operation for job {job_id}")
                return jsonify(job_data), 200

            headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
            poll_payload = {"operationName": operation_name}

            try:
                poll_response = requests.post(FETCH_OP_URL, headers=headers, json=poll_payload, timeout=10)
                logging.info(f"Poll request for {job_id} status: {poll_response.status_code}")
                poll_response.raise_for_status()
                op_data = poll_response.json()

                if op_data.get('done'):
                    logging.info(f"AI operation DONE for job {job_id}.")
                    ai_response_data = op_data.get('response', {})
                    video_outputs = ai_response_data.get('videos', [])
                    video_urls = []

                    if not video_outputs:
                         # This might be an RAI filter hit
                         rai_filtered = ai_response_data.get('raiMediaFilteredCount', 0)
                         rai_reasons = ai_response_data.get('raiMediaFilteredReasons', [])
                         if rai_filtered > 0:
                             error_msg = f"AI filter blocked prompt. Reason: {rai_reasons}"
                             logging.warning(f"RAI filter hit for job {job_id}: {rai_reasons}")
                             raise Exception(error_msg)
                         else:
                             logging.error(f"AI op done but no 'videos' in response for job {job_id}. Response: {ai_response_data}")
                             raise Exception("AI finished but returned no video outputs.")

                    if not storage_client:
                        logging.error("Storage client not available for processing GCS URIs.")
                        raise Exception("Storage client not initialized")

                    for i, output in enumerate(video_outputs):
                         gcs_uri = output.get('gcsUri')
                         if not gcs_uri or not gcs_uri.startswith("gs://"):
                             logging.warning(f"Output {i} for job {job_id} missing valid gcsUri. Skipping.")
                             continue
                         try:
                             bucket_name_from_uri = gcs_uri.split('/')[2]
                             object_name = '/'.join(gcs_uri.split('/')[3:])
                             if not object_name:
                                 logging.warning(f"Could not parse object name from GCS URI {gcs_uri} for job {job_id}. Skipping.")
                                 continue

                             bucket = storage_client.bucket(bucket_name_from_uri)
                             blob = bucket.blob(object_name)
                             
                             blob.make_public() # Ensure object is public
                             public_url = blob.public_url
                             video_urls.append(public_url)
                             logging.info(f"Processed GCS URI for job {job_id}: {gcs_uri} -> Public URL: {public_url}")

                         except Exception as url_err:
                              logging.error(f"Error processing GCS URI {gcs_uri} for job {job_id}: {url_err}")

                    if not video_urls:
                         raise Exception("AI finished but no valid video URLs could be constructed.")

                    # Update Firestore
                    final_status_update = { 'status': 'complete', 'updated_at': firestore.SERVER_TIMESTAMP }
                    if len(video_urls) == 1: final_status_update['video_url'] = video_urls[0]
                    else: final_status_update['video_urls'] = video_urls
                    doc_ref.update(final_status_update)
                    logging.info(f"Job {job_id} successfully completed.")
                    
                    # --- THIS IS THE FIX ---
                    # Update local job_data *without* the Sentinel object
                    job_data['status'] = 'complete'
                    if 'video_url' in final_status_update:
                        job_data['video_url'] = final_status_update['video_url']
                    if 'video_urls' in final_status_update:
                        job_data['video_urls'] = final_status_update['video_urls']
                    # We can't return the Sentinel, so we'll just return the updated job_data
                    # --- END FIX ---
                    
                    return jsonify(job_data), 200 # Return the modified job_data
                else:
                    # Operation still running
                    logging.info(f"AI operation still running for job {job_id}.")
                    return jsonify(job_data), 200 # Return current 'processing_ai' status

            except requests.exceptions.RequestException as poll_err:
                 logging.error(f"Failed to poll AI operation status for job {job_id}: {poll_err}. Response: {poll_err.response.text if poll_err.response else 'No response'}")
                 return jsonify(job_data), 200

        else:
            # Status is 'pending', 'complete', or 'failed'
            return jsonify(job_data), 200

    except Exception as e:
        logging.exception(f"Error in check_status for job {job_id}: {e}")
        try:
             if doc_ref: doc_ref.update({'status': 'failed', 'error': f'Check status failed: {str(e)}'})
        except: pass
        return jsonify({'error': 'Failed to check job status'}), 500


@app.route('/api/run-task', methods=['POST'])
def run_task():
    """ Worker endpoint: Calls the AI model via REST and stores the operation name. """
    if not db:
        logging.error("DB client not available.")
        return "Server configuration error", 500

    # OIDC Verification
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '): return "Unauthorized", 401
        token = auth_header.split(' ')[1]
        request_session = auth_requests.Request()
        audience = os.getenv('SERVICE_URL')
        if not audience: audience = get_cloud_run_url()
        if not audience: return "Config error: No audience", 500
        
        audience = audience.rstrip('/') # Apply the OIDC fix
        
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
        storage_uri = data.get('storageUri')
        if not all([job_id, prompt, aspect_ratio, duration, num_videos, storage_uri]):
            logging.error(f"Missing data in task payload for job {job_id or 'UNKNOWN'}: {data}")
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

        headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
        instances = [{"prompt": prompt}]
        parameters = {
            "durationSeconds": duration,
            "aspectRatio": aspect_ratio,
            "sampleCount": num_videos,
            "storageUri": storage_uri,
            "generateAudio": True 
        }
        request_body = {"instances": instances, "parameters": parameters}

        logging.info(f"Calling predictLongRunning for job {job_id} with body: {json.dumps(request_body)}")
        response = requests.post(PREDICT_URL, headers=headers, json=request_body, timeout=60)
        logging.info(f"predictLongRunning response status: {response.status_code} for job {job_id}")
        response.raise_for_status()
        response_data = response.json()

        operation_name = response_data.get('name')
        if not operation_name:
             logging.error(f"predictLongRunning response missing 'name' for job {job_id}. Response: {response_data}")
             raise Exception("predictLongRunning response did not contain operation 'name'")

        logging.info(f"AI operation started for job {job_id}: {operation_name}")

        doc_ref.update({
            'status': 'processing_ai',
            'operation_name': operation_name,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} status updated to processing_ai.")

        return jsonify({"status": "success, AI processing started"}), 200 # OK to Cloud Tasks

    except Exception as e:
        logging.exception(f"Error in run_task AI call/update for job {job_id}: {e}")
        if isinstance(e, requests.exceptions.RequestException) and e.response is not None:
             logging.error(f"AI API Response Text: {e.response.text}")
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
