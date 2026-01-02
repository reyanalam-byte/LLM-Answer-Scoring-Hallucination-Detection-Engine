import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"


def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def generate_answers(query):
    """
    Multiple agents answer the same query
    """
    agents = {
        "Agent_1": f"Answer factually and concisely:\n{query}",
        "Agent_2": f"Answer with detailed reasoning:\n{query}",
        "Agent_3": f"Answer critically and verify claims:\n{query}"
    }

    answers = {}

    for agent, prompt in agents.items():
        answers[agent] = call_ollama(prompt)

    return answers
