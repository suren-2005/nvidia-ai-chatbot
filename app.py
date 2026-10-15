
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from .env
load_dotenv()

# RETIRVE THE API KEY
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError(
        "NVIDIA_API_KEY is missing. Check your .env file."
    )

# Connect to NVIDIA's hosted LLM API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Instructions for the chatbot
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Answer clearly and explain technical topics "
            "in simple English."
        )
    }
]

print("NVIDIA AI Chatbot")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            max_tokens=512,         # 1 word ≈ 1-1.5 tokens & 100 words ≈ 130 tokens
            temperature=0.6         # low temp = more factual, less creative, very predictable 
        )                           # highe temp = more cretative,less predictive have chance of mistakes and hallucination

        answer = response.choices[0].message.content

        print(f"\nChatbot: {answer}\n")

        messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:
        print(f"API error: {error}\n")
        # Remove the unanswered user message so history stays valid
        messages.pop()
