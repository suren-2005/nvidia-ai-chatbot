
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="NVIDIA AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My AI Chatbot")
st.caption("Powered by an NVIDIA-hosted LLM")

# Read the API key
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("NVIDIA_API_KEY is missing from your .env file.")
    st.stop()

# Create the NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

MODEL = "openai/gpt-oss-20b"

# Store the conversation during the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display the chat input
question = st.chat_input("Ask me anything...")

if question:
    # Display and store the user's message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Ask NVIDIA for a response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful AI assistant. "
                                "Explain concepts in simple English."
                            )
                        },
                        *st.session_state.messages[-20:]
                    ],
                    max_tokens=512,
                    temperature=0.6
                )

                answer = (
                    response.choices[0].message.content
                    or "The model returned an empty response."
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Unable to get an AI response: {e}")
