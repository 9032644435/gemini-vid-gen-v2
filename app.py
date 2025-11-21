from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import uuid
from google.cloud import firestore, tasks_v2, storage
import logging
import json
import base64
import time
import google.auth
from google.auth.transport import requests as auth_requests
from google.oauth2 import id_token
import requests
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "gemini-vid-gen-v2")
REGION = os.getenv("GOOGLE_REGION", "us-central1")
FIRESTORE_COLLECTION = "video-generations"
TASK_QUEUE = "video-gen-queue"
TASK_SPN = os.getenv("TASK_WORKER_SA_EMAIL", f"video-gen-worker-sa@{PROJECT_ID}.iam.gserviceaccount.com")
BUCKET_NAME = os.getenv("GOOGLE_BUCKET_NAME", f"{PROJECT_ID}-video-outputs")
VIDEO_MODEL_ID = "veo-3.1-generate-preview"

API_ENDPOINT_BASE = f"https://{REGION}-aiplatform.googleapis.com/v1"
PREDICT_URL = f"{API_ENDPOINT_BASE}/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{VIDEO_MODEL_ID}:predictLongRunning"
FETCH_OP_URL = f"{API_ENDPOINT_BASE}/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{VIDEO_MODEL_ID}:fetchPredictOperation"

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

db = None
tasks_client = None
storage_client = None

# Force Update Comment
try:
    db = firestore.Client(project=PROJECT_ID)
    tasks_client = tasks_v2.CloudTasksClient()
    storage_client = storage.Client()
    vertexai.init(project=PROJECT_ID, location=REGION)
    logging.info("Clients initialized.")
except Exception as e:
    logging.critical(f"FATAL: Error initializing clients: {e}")

def get_cloud_run_url():
    service_url = os.getenv('SERVICE_URL')
    if service_url: return service_url if service_url.endswith('/') else service_url + '/'
    try:
        metadata_server_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_url"
        metadata_response = requests.get(metadata_server_url, headers={"Metadata-Flavor": "Google"}, timeout=5)
        metadata_response.raise_for_status()
        return metadata_response.text if metadata_response.text.endswith('/') else metadata_response.text + '/'
    except: return None

def get_auth_token():
    try:
        creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except: return None

@app.route('/')
def index(): return render_template('index.html')

@app.route('/remix')
def remix_page(): return render_template('remix.html')

@app.route('/api/describe-image', methods=['POST'])
def describe_image():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if ',' in image_data: image_data = image_data.split(',')[1]
        model = GenerativeModel("gemini-1.5-flash")
        image_part = Part.from_data(mime_type="image/jpeg", data=base64.b64decode(image_data))
        prompt = "Describe this image in extreme detail for a video generation prompt. Focus on lighting, style, characters, and setting. Keep it under 100 words."
        response = model.generate_content([prompt, image_part])
        return jsonify({'description': response.text})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    if not db: return jsonify({'error': 'Server config error'}), 500
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        job_id = str(uuid.uuid4())
        storage_uri = f"gs://{BUCKET_NAME}/{job_id}/"
        
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc_ref.set({
            'prompt': prompt, 'status': 'pending', 'job_id': job_id,
            'storage_uri_requested': storage_uri, 'created_at': firestore.SERVER_TIMESTAMP
        })

        target_url = get_cloud_run_url()
        if not target_url: return jsonify({'error': 'No service URL'}), 500

        worker_endpoint = target_url + 'api/run-task'
        task_payload = {
            'job_id': job_id, 'prompt': prompt, 'aspect_ratio': '16:9',
            'duration': 8, 'num_videos': 1, 'storageUri': storage_uri
        }
        
        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': worker_endpoint,
                'oidc_token': { 'service_account_email': TASK_SPN, 'audience': target_url.rstrip('/') },
                'headers': {'Content-type': 'application/json'},
                'body': json.dumps(task_payload).encode('utf-8')
            }
        }
        tasks_client.create_task(parent=tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE), task=task)
        return jsonify({'job_id': job_id, 'status': 'pending'}), 202
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/check-status/<job_id>')
def check_status(job_id):
    try:
        doc = db.collection(FIRESTORE_COLLECTION).document(job_id).get()
        if not doc.exists: return jsonify({'error': 'Job not found'}), 404
        job_data = doc.to_dict()
        
        if job_data.get('status') == 'processing_ai':
            token = get_auth_token()
            headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
            poll_payload = {"operationName": job_data.get('operation_name')}
            op_data = requests.post(FETCH_OP_URL, headers=headers, json=poll_payload, timeout=10).json()
            
            if op_data.get('done'):
                video_outputs = op_data.get('response', {}).get('videos', [])
                if video_outputs:
                    gcs_uri = video_outputs[0].get('gcsUri')
                    blob = storage_client.bucket(gcs_uri.split('/')[2]).blob('/'.join(gcs_uri.split('/')[3:]))
                    blob.make_public()
                    db.collection(FIRESTORE_COLLECTION).document(job_id).update({'status': 'complete', 'video_url': blob.public_url})
                    job_data['status'] = 'complete'
                    job_data['video_url'] = blob.public_url
                else:
                    db.collection(FIRESTORE_COLLECTION).document(job_id).update({'status': 'failed'})
                    job_data['status'] = 'failed'
        
        job_data.pop('created_at', None)
        job_data.pop('updated_at', None)
        return jsonify(job_data), 200
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/run-task', methods=['POST'])
def run_task():
    try:
        token = request.headers.get('Authorization', '').split(' ')[1]
        id_token.verify_oauth2_token(token, auth_requests.Request(), audience=get_cloud_run_url().rstrip('/'))
    except: return "Unauthorized", 401

    try:
        data = request.get_json()
        job_id = data.get('job_id')
        db.collection(FIRESTORE_COLLECTION).document(job_id).update({'status': 'calling_ai'})
        
        token = get_auth_token()
        headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
        body = {
            "instances": [{"prompt": data.get('prompt')}],
            "parameters": {"durationSeconds": 8, "aspectRatio": "16:9", "sampleCount": 1, "storageUri": data.get('storageUri'), "generateAudio": True}
        }
        response = requests.post(PREDICT_URL, headers=headers, json=body).json()
        db.collection(FIRESTORE_COLLECTION).document(job_id).update({'status': 'processing_ai', 'operation_name': response.get('name')})
        return jsonify({"status": "success"}), 200
    except Exception as e: return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

