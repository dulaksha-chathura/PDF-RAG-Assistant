import streamlit as st

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})
