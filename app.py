import os
import uuid
import json
import threading
import queue
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'shifu-secret-dev-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv'}

# Ensure upload dir exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# In-memory conversation store (replace with DB in production)
conversations: dict[str, list] = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_file_type(filename: str) -> str:
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext == 'pdf':
        return 'pdf'
    if ext in {'png', 'jpg', 'jpeg', 'gif', 'webp'}:
        return 'image'
    return 'document'


def run_shifu_crew(user_input: str, uploaded_files: list[dict], result_queue: queue.Queue):
    """Run the CrewAI crew and push result to queue."""
    try:
        from crew import ShifuAssistantCrew

        inputs = {"user_input": user_input}

        # Attach file paths to input context so agents can use them
        if uploaded_files:
            file_context_parts = []
            for f in uploaded_files:
                file_context_parts.append(f"[{f['type'].upper()}] {f['original_name']} -> {f['path']}")
            inputs["attached_files"] = "\n".join(file_context_parts)
            inputs["user_input"] = (
                f"{user_input}\n\nAttached files:\n{inputs['attached_files']}"
            )

        crew_instance = ShifuAssistantCrew().crew()
        result = crew_instance.kickoff(inputs=inputs)
        result_queue.put({"success": True, "content": str(result)})

    except ImportError:
        # Fallback demo mode when crew.py isn't present
        import time
        demo_response = f"""## Shifu is ready! 🥷

I received your message: **{user_input}**

> *CrewAI crew module not detected — running in demo mode.*

To activate the full agent, make sure `crew.py` is in the same directory as `app.py`.

### What I can do when fully connected:
- 🔍 **Web search** via Serper
- 📄 **Read & write files**
- 📚 **Search PDFs**
- 🗂️ **Browse directories**

```python
# Quick start
from crew import ShifuAssistantCrew
result = ShifuAssistantCrew().crew().kickoff(inputs={{"user_input": "your task"}})
```
"""
        time.sleep(1.2)
        result_queue.put({"success": True, "content": demo_response})

    except Exception as e:
        result_queue.put({"success": False, "content": f"**Error:** {str(e)}"})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/conversations', methods=['POST'])
def new_conversation():
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = []
    return jsonify({"conversation_id": conv_id})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    return jsonify({
        "success": True,
        "file_id": unique_name,
        "original_name": filename,
        "file_type": get_file_type(filename),
        "path": filepath
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    conversation_id = data.get('conversation_id')
    user_message = data.get('message', '').strip()
    uploaded_files = data.get('files', [])  # list of {file_id, original_name, path, type}

    if not user_message and not uploaded_files:
        return jsonify({"error": "Empty message"}), 400

    if not conversation_id or conversation_id not in conversations:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []

    # Store user message
    conversations[conversation_id].append({
        "role": "user",
        "content": user_message,
        "files": uploaded_files
    })

    def sse(payload: dict) -> str:
        # json.dumps escapes internal newlines so SSE frame is never split by content
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    def generate():
        result_q: queue.Queue = queue.Queue()

        # Run crew in background thread
        t = threading.Thread(
            target=run_shifu_crew,
            args=(user_message, uploaded_files, result_q),
            daemon=True
        )
        t.start()

        # Send SSE keepalives while waiting
        import time
        while t.is_alive():
            yield sse({"type": "ping"})
            time.sleep(0.8)

        t.join()
        result = result_q.get()

        if result['success']:
            conversations[conversation_id].append({
                "role": "assistant",
                "content": result['content']
            })
            yield sse({
                "type": "done",
                "content": result['content'],
                "conversation_id": conversation_id
            })
        else:
            yield sse({"type": "error", "content": result['content']})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/conversations/<conv_id>/history', methods=['GET'])
def get_history(conv_id: str):
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"messages": conversations[conv_id]})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)