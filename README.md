# LLM-Answer-Scoring-Hallucination-Detection-Engine
Truth-aware LLM answer scoring engine with hallucination detection and factual verification.

# 🧠 Hallucination-Aware Multi-LLM Analyzer

A Flask-based system that queries multiple Large Language Models (LLMs) in parallel and evaluates their responses using semantic similarity, factual verification, and overconfidence detection.

---

## 🚀 Features

- Parallel querying of multiple Ollama-hosted LLMs
- Hallucination-aware scoring system
- Overconfidence penalty based on language usage
- Semantic similarity validation
- Multithreaded execution for faster responses
- Simple web-based interface

---

## 📁 Project Structure

hallucination-model/
├── app.py
├── agents.py
├── scoring.py
├── utils.py
├── requirements.txt
├── README.md
└── templates/
└── index.html

---

## ⚙️ Installation & Setup

Follow the steps below in order.

---

### 1️⃣ Prerequisites

- Python 3.9 or higher
- Git
- Ollama installed and running

Start Ollama:

ollama serve

2️⃣ Clone the Repository

git clone https://github.com/YOUR_USERNAME/hallucination-model.git
cd hallucination-model

3️⃣ Create and Activate Virtual Environment (Recommended)

python -m venv .venv

Windows
.venv\Scripts\activate

Linux / macOS
source .venv/bin/activate

4️⃣ Install Python Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

5️⃣ Download Required Models

ollama pull phi3
ollama pull qwen2.5:1.5b
ollama pull deepseek-r1:1.5b

6️⃣ Run the Application

python app.py
Open your browser and visit:
http://127.0.0.1:5000

##🧪 How to Use

Enter a factual query (e.g. Who invented the incandescent light bulb?)

Click Analyze

The system will:

Query multiple LLMs in parallel

Apply hallucination and confidence penalties

Rank responses

Highlight the most reliable answer

##🧠 Why This Project?

Large Language Models often:

Hallucinate facts

Invent sources

Sound confident even when incorrect

This project evaluates truth-likelihood, not fluency.

##🔮 Future Improvements

Dynamic knowledge base integration

Topic-agnostic scoring

Model consensus scoring

Explainable score breakdown

Research-grade evaluation metrics

