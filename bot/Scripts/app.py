# app.py

import streamlit as st
from main import answerquery

st.set_page_config(page_title="🩺 Medical Assistant Chatbot", layout="centered")
st.title("🩺 Medical Assistant Chatbot")
st.markdown("Ask about your symptoms to get disease predictions, medications, and precautions.\n\n**Examples:**\n- I have a sore throat and fever\n- My head hurts and I feel dizzy")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Describe your symptoms..."):
    # User message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Bot response
    with st.chat_message("assistant"):
        response = answerquery(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
