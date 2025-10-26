from flask import Flask, render_template, request, jsonify
import os
import uuid
from google.cloud import firestore
from google.cloud import tasks_v2

# --- Configuration ---
PROJECT_ID = "gemini-vid-gen-v2"
REGION = "us-central1"
FIRESTORE_COLLECTION = "video-generations"
TASK_QUEUE = "video-generation-queue"
TASK_SPN = "video-generation-sa@gemini-vid-gen-v2.iam.gserviceaccount.com"
CLOUD_RUN_URL = "https://video-generation-processor-s65g52q7ya-uc.a.run.app"
BUCKET_NAME = "gemini-vid-gen-v2-output"

# --- Client Initialization ---
app = Flask(__name__)
db = firestore.Client()
tasks_client = tasks_v2.CloudTasksClient()

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
            'job_id': job_id
        })

        # Create a Cloud Task to process the video generation
        task_parent = tasks_client.queue_path(PROJECT_ID, REGION, TASK_QUEUE)
        task = {
            'http_request': {
                'http_method': tasks_v2.HttpMethod.POST,
                'url': CLOUD_RUN_URL,
                'oidc_token': {
                    'service_account_email': TASK_SPN
                },
                'headers': {'Content-type': 'application/json'},
                'body': jsonify({'job_id': job_id}).data
            }
        }
        tasks_client.create_task(parent=task_parent, task=task)

        return jsonify({'job_id': job_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-status/<job_id>')
def check_status(job_id):
    """
    Checks the status of a video generation job.
    - Queries Firestore for the job document.
    - Returns the job status and video URL if complete.
    """
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return jsonify(doc.to_dict())
        else:
            return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
