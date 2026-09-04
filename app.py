import os
import json
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename

# Import modules from our multi-agent file
from multi_agent_rag_pipeline import get_or_create_vector_store, agent_pipeline, embeddings


from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# The SDK automatically detects the OPENAI_API_KEY environment variable
client = OpenAI()



# UPDATE TO USE WITH RAG PIPELINE FOR FRONT END AND CONNECT WITH RAG TOGETHER WITH CHAT WINDOW. DONT ADD CODE FROM TXT FILE. THIS IS CLRAN


app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_docs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TRACKED_FILES = []


def sync_tracked_files_from_disk():
    global TRACKED_FILES
    from multi_agent_rag_pipeline import REGISTRY_FILE
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r") as f:
                registry = json.load(f)
                TRACKED_FILES = [os.path.join(UPLOAD_FOLDER, fname) for fname in registry.keys() if
                                 os.path.exists(os.path.join(UPLOAD_FOLDER, fname))]
        except Exception:
            TRACKED_FILES = []


sync_tracked_files_from_disk()


# ---------------------------------------------------------------------
# CLEAN BACKEND API ROUTES
# ---------------------------------------------------------------------
@app.route('/')
def index():
    from multi_agent_rag_pipeline import DB_DIR
    db_exists = os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)
    # Renders the clean index.html file from the /templates directory
    return render_template('index.html', db_exists=db_exists)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({"error": "No file stream provided"}), 400

    files = request.files.getlist('files')
    saved_paths = []

    for file in files:
        if file.filename == '': continue
        filename = secure_filename(file.filename)
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dest_path)
        saved_paths.append(dest_path)

    vector_store = get_or_create_vector_store(saved_paths, embeddings)
    sync_tracked_files_from_disk()

    return jsonify({"status": "success", "added": saved_paths, "skipped": []})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query string empty"}), 400

    sync_tracked_files_from_disk()

    def generate_agent_stream():
        initial_state = {
            "file_paths": TRACKED_FILES,
            "user_query": user_query,
            "next_node": "",
            "retrieved_context": "",
            "draft_answer": "",
            "final_answer": "",
            "review_status": "passed",
            "retry_count": 0
        }

        for event in agent_pipeline.stream(initial_state, stream_mode="updates"):
            for node_name, output_payload in event.items():
                message = "Initiating execution workflow..."
                if node_name == "orchestrator":
                    message = f"Routing user query towards the specialized target: '{output_payload.get('next_node')}' node."
                elif node_name == "rag_search":
                    message = "Factual passages fetched from vector database. Compiling specialized response framework."
                elif node_name == "summarize":
                    message = "Bulk index structural summaries context pulled. Compiling narrative response framework."
                elif node_name == "reviewer":
                    status = output_payload.get("review_status", "passed").upper()
                    message = f"Verification engine evaluation finished. Result status: {status}."

                log_payload = {"type": "log", "node": node_name, "message": message}
                yield f"data: {json.dumps(log_payload)}\n\n"

                if "final_answer" in output_payload and output_payload["final_answer"]:
                    token_payload = {"type": "token", "text": output_payload["final_answer"]}
                    yield f"data: {json.dumps(token_payload)}\n\n"

    return Response(generate_agent_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, port=5000)




