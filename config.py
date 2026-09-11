import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from local .env file
load_dotenv()

def init_groq_client():
    """
    Native API Key loader for VS Code / Local & Streamlit Cloud Deployment.
    1. Reads from local .env file or environment variables
    2. Reads from Streamlit Cloud Secrets (.streamlit/secrets.toml)
    """
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

    if groq_api_key:
        return Groq(api_key=groq_api_key)
    return None
    """
    Universal API Key loader.
    Supports Google Colab Secrets, local .env files, OS Environment variables, and Streamlit Secrets.
    """
    groq_api_key = None
    
    # 1. Google Colab Secrets
    try:
        from google.colab import userdata
        groq_api_key = userdata.get('GROQ_API_KEY')
    except Exception:
        pass

    # 2. Local .env / OS Env / Streamlit Secrets
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

    if groq_api_key:
        return Groq(api_key=groq_api_key)
    return None