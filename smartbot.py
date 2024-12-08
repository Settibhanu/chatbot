import streamlit as st
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") 

# Function to query Gemini LLM
def query_gemini(prompt, max_tokens=1000):
     genai.configure(api_key=API_KEY)
     model = genai.GenerativeModel("gemini-1.5-flash")
     response = model.generate_content(prompt)
     
     return response.text

# Streamlit App
st.title("Gemini LLM Chatbot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input Area
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get LLM response
        response = query_gemini(user_input)
        
        # Add LLM response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear input
        #st.session_state.user_input = ""

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")