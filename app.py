import os
from dotenv import load_dotenv

import streamlit as st
from google import genai

# ---------- Setup ----------
load_dotenv()  # loads GEMINI_API_KEY from .env (if present)
client = genai.Client()  # picks up GEMINI_API_KEY automatically

MODEL_NAME = "gemini-2.5-flash"  # fast + free-tier friendly

st.set_page_config(page_title="Coolest Chatbot Ever", page_icon="🤖")
st.title("🤖 *Insert cool name here*")

# Keep chat history in Streamlit session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "model", "content": "Hi! I'm the coolest LLM around. What's up?"}
    ]

# Render chat history
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"] == "model" else "user"):
        st.markdown(m["content"])

# Chat input (bottom bar)
prompt = st.chat_input("Type your message...")
if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the history in Gemini format
    history = []
    for m in st.session_state.messages[:-1]:  # all but the latest user msg
        history.append(
            {"role": m["role"], "parts": [{"text": m["content"]}]}
        )

    # Stream the model's response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        # The GenAI SDK supports server-side token streaming
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            # You can also pass system_instruction in config for persona
            contents=history + [{"role": "user", "parts": [{"text": prompt}]}],
        )
        for chunk in stream:
            if chunk.text:
                streamed_text += chunk.text
                placeholder.markdown(streamed_text)

        # Finalize assistant message in history
        st.session_state.messages.append(
            {"role": "model", "content": streamed_text or "_(no response)_"})