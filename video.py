# video.py
import streamlit as st
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import time
import numpy as np
import json
import tempfile
import os
import traceback

def initialize_vertex_ai():
    credentials_path = "/tmp/gcp_credentials.json"
    with open(credentials_path, "w") as f:
        json.dump({
            "type": st.secrets["gcp"]["type"],
            "project_id": st.secrets["gcp"]["project_id"],
            "private_key_id": st.secrets["gcp"]["private_key_id"],
            "private_key": st.secrets["gcp"]["private_key"],
            "client_email": st.secrets["gcp"]["client_email"],
            "client_id": st.secrets["gcp"]["client_id"],
            "auth_uri": st.secrets["gcp"]["auth_uri"],
            "token_uri": st.secrets["gcp"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp"]["client_x509_cert_url"]
        }, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    vertexai.init(project=st.secrets["gcp"]["project_id"], location="us-central1")

def analyze_video(video_file, prompt, temperature=1.0, top_p=0.95, max_tokens=4096):
    try:
        st.info("Running Gemini video analysis...")

        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

        # Load video into Gemini API (as Part)
        model = GenerativeModel("gemini-1.5-pro-preview-0409")
        video_part = Part.from_uri(temp_video_path, mime_type="video/mp4")

        response = model.generate_content(
            [prompt, video_part],
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens
            }
        )

        st.success("Video Analysis Complete:")
        st.markdown(response.text)

    except Exception as e:
        st.error(f"❌ An error occurred during video analysis: {e}")
