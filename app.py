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
# Use the specific video classes (removed GenerativeModel import)
from vertexai.vision_models import VideoGenerationModel, GenerateVideosConfig
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
        metadata_response = requests.get(metadata_server_url, headers={"Metadata-Flavor": "Google"}, timeout=5)
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
            # Add specific valid durations if known for the model
            if not (1 <= duration <= 10): raise ValueError("Duration invalid")
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

        worker_endpoint = target_url + 'api/run-task'
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
                    'audience': target_url # Root URL is
