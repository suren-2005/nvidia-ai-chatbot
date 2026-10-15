
import os

from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load secret settings from .env
load_dotenv()

app = Flask(__name__)

api_key = os.getenv("NVIDIA_API_KEY")
model = os.getenv("NVIDIA_MODEL")

if not api_key:
    raise ValueError("NVIDIA_API_KEY is missing from .env")

if not model:
    raise ValueError("NVIDIA_MODEL is missing from .env")

# 2. Create a client for NVIDIA's hosted LLM API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# 3. Create a simple route to check whether Flask is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Chatbot Flask API is ready"
    })


# 4. Create the chatbot endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please provide a message"
        }), 400

    try:
        # 5. Send the question to NVIDIA
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful conversational AI assistant. "
                        "Answer clearly and politely."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_tokens=512,
            temperature=0.6
        )

        answer = response.choices[0].message.content

        # 6. Return the AI answer as JSON
        return jsonify({
            "answer": answer or "The model returned an empty response."
        })

    except Exception:
        app.logger.exception("NVIDIA API request failed")

        return jsonify({
            "error": "Unable to generate an answer. Check the backend logs."
        }), 502


# 7. Start the Flask development server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
