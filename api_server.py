"""
GraphRAG Query API Server
Runs post_graph_inference.py with user-provided parameters and returns
the structured audit JSON + raw answer for the UI.
"""

import json
import os
import subprocess
import sys
import tempfile
import uuid

from flask import Flask, jsonify, request, stream_with_context, Response
from flask_cors import CORS

# Load project .env so subprocesses always have GROQ_API_KEY etc.
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(_env_path, override=False)  # don't override already-set shell vars
    print(f"Loaded .env from {_env_path}")
except ImportError:
    pass


app = Flask(__name__)
CORS(app)

PYTHON = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
SCRIPT = os.path.join(os.path.dirname(__file__), "post_graph_inference.py")


def _default(key: str, fallback: str) -> str:
    return os.getenv(key, fallback)


@app.route("/api/query", methods=["POST"])
def run_query():
    """
    POST /api/query
    Body (JSON):
      question       – the clinical question (required)
      top_k          – int (default 3)
      max_hops       – int (default 2)
      neo4j_url      – string (default from env)
      neo4j_username – string (default neo4j)
      neo4j_password – string (required)
      audit          – bool (default true)
    
    Returns the audit JSON + stdout as plain answer.
    """
    data = request.get_json(force=True)

    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    top_k = int(data.get("top_k", 3))
    max_hops = int(data.get("max_hops", 2))
    neo4j_url = data.get("neo4j_url") or _default("NEO4J_URL", "bolt://localhost:7687")
    neo4j_username = data.get("neo4j_username") or _default("NEO4J_USERNAME", "neo4j")
    neo4j_password = data.get("neo4j_password") or _default("NEO4J_PASSWORD", "")
    audit = bool(data.get("audit", True))

    if not neo4j_password:
        return jsonify({"error": "neo4j_password is required"}), 400

    # Write audit output to a temp file so we can read the JSON back
    tmp_audit = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, dir=os.path.dirname(__file__)
    )
    tmp_audit.close()
    audit_path = tmp_audit.name

    cmd = [
        PYTHON, SCRIPT,
        "--question", question,
        "--top-k", str(top_k),
        "--max-hops", str(max_hops),
        "--neo4j-url", neo4j_url,
        "--neo4j-username", neo4j_username,
        "--neo4j-password", neo4j_password,
    ]

    if audit:
        cmd += ["--audit", "--audit-output", audit_path]

    env = os.environ.copy()
    env["NEO4J_PASSWORD"] = neo4j_password

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            env=env,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Query timed out after 120 seconds"}), 504

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return jsonify({
            "error": f"Script exited with code {proc.returncode}",
            "stderr": stderr,
            "stdout": stdout,
        }), 500

    # Try to read audit JSON
    audit_data = None
    if audit:
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
        except Exception as e:
            audit_data = None
        finally:
            try:
                os.unlink(audit_path)
            except Exception:
                pass

    return jsonify({
        "question": question,
        "stdout": stdout,
        "stderr": stderr,
        "audit": audit_data,
    })


@app.route("/api/stream-query", methods=["POST"])
def stream_query():
    """
    POST /api/stream-query  – same params as /api/query but streams stdout lines
    as server-sent events so the UI can show live progress.
    """
    data = request.get_json(force=True)

    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    top_k = int(data.get("top_k", 3))
    max_hops = int(data.get("max_hops", 2))
    neo4j_url = data.get("neo4j_url") or _default("NEO4J_URL", "bolt://localhost:7687")
    neo4j_username = data.get("neo4j_username") or _default("NEO4J_USERNAME", "neo4j")
    neo4j_password = data.get("neo4j_password") or _default("NEO4J_PASSWORD", "")
    audit = bool(data.get("audit", True))

    if not neo4j_password:
        return jsonify({"error": "neo4j_password is required"}), 400

    tmp_audit = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, dir=os.path.dirname(__file__)
    )
    tmp_audit.close()
    audit_path = tmp_audit.name

    cmd = [
        PYTHON, SCRIPT,
        "--question", question,
        "--top-k", str(top_k),
        "--max-hops", str(max_hops),
        "--neo4j-url", neo4j_url,
        "--neo4j-username", neo4j_username,
        "--neo4j-password", neo4j_password,
    ]
    if audit:
        cmd += ["--audit", "--audit-output", audit_path]

    env = os.environ.copy()
    env["NEO4J_PASSWORD"] = neo4j_password

    def generate():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(__file__),
            env=env,
        )

        for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"

        proc.wait()

        # After process ends, send the audit JSON if available
        audit_data = None
        if audit:
            try:
                with open(audit_path, "r", encoding="utf-8") as f:
                    audit_data = json.load(f)
            except Exception:
                pass
            finally:
                try:
                    os.unlink(audit_path)
                except Exception:
                    pass

        yield f"data: {json.dumps({'done': True, 'returncode': proc.returncode, 'audit': audit_data})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "script": os.path.exists(SCRIPT)})


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "5001"))
    print(f"GraphRAG API server running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
