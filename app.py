from flask import Flask, render_template, request, jsonify
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import overconfidence_penalty

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

# ⏱️ connection timeout (seconds)
CONNECT_TIMEOUT = 5


def call_ollama(model_name, query):
    start_time = time.time()

    # ⏱️ model-specific read timeout
    read_timeout = 120
    if "deepseek" in model_name.lower():
        read_timeout = 500  # give DeepSeek more time

    payload = {
        "model": model_name,
        "prompt": query,
        "stream": False,
        # optional but recommended to limit output
        "options": {
            "num_predict": 200
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(CONNECT_TIMEOUT, read_timeout)
        )
        response.raise_for_status()

        output = response.json().get("response", "").strip()
        if not output:
            raise RuntimeError("Empty response from model")

        score = 1.0 - overconfidence_penalty(output)

        return {
            "agent": model_name,
            "response": output,
            "score": round(score, 2),
            "time": round(time.time() - start_time, 2),
            "status": "ok"
        }

    except requests.exceptions.ReadTimeout:
        return {
            "agent": model_name,
            "response": "⏱️ Timed out (model took too long)",
            "score": 0.0,
            "time": round(time.time() - start_time, 2),
            "status": "timeout"
        }

    except requests.exceptions.ConnectTimeout:
        return {
            "agent": model_name,
            "response": "❌ Could not connect to Ollama",
            "score": 0.0,
            "time": 0,
            "status": "connection_error"
        }

    except Exception as e:
        return {
            "agent": model_name,
            "response": f"❌ Error: {str(e)}",
            "score": 0.0,
            "time": 0,
            "status": "error"
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Invalid request"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    models = [
        "phi3",
        "qwen2.5:1.5b",
        "deepseek-r1:1.5b"
    ]

    answers = []

    # 🚀 PARALLEL EXECUTION
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [
            executor.submit(call_ollama, model, query)
            for model in models
        ]

        for future in as_completed(futures):
            answers.append(future.result())

    max_score = max(ans["score"] for ans in answers)
    best_agents = [ans for ans in answers if ans["score"] == max_score]

    return jsonify({
        "answers": answers,
        "best_agents": best_agents
    })


if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)
