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

def analyze_video(uploaded_video, prompt, temperature, top_p, max_tokens):
    try:
        initialize_vertex_ai()

        if isinstance(uploaded_video, bytes):
            video_bytes = uploaded_video
        else:
            video_bytes = uploaded_video.read()
        encoded_video = base64.b64encode(video_bytes).decode('utf-8')

        progress_bar = st.progress(0)
        progress_info = st.info("Starting analysis... This might take a few moments.")

        video_part = Part.from_data(
            mime_type=uploaded_video.type,
            data=base64.b64decode(encoded_video),
        )
        text_part = prompt

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        model = GenerativeModel("gemini-1.5-flash-002")

        start_time = time.time()
        with st.spinner('Analyzing the video... This might take a few moments.'):
            responses = model.generate_content(
                [video_part, text_part],
                generation_config=generation_config,
                safety_settings=[],
                stream=True,
            )

            output_text = ""
            placeholder = st.empty()
            for i, response in enumerate(responses):
                output_text += response.text
                placeholder.text_area("Current Analysis", value=output_text, height=150)
                progress = min((i + 1) / 10, 1.0)
                progress_bar.progress(progress)
                time.sleep(0.1)

            st.success("Analysis Complete!")
            end_time = time.time()
            st.info(f"The analysis took approximately {end_time - start_time:.2f} seconds.")
            st.markdown("### Analysis Result:")
            st.write(output_text)

            if "emotion" in prompt.lower():
                st.markdown("### Emotional Intensity Over Time:")
                emotional_intensity = np.random.rand(10)
                st.line_chart(emotional_intensity)

            if st.button("📂 Save Analysis to File"):
                with open("analysis_output.txt", "w") as f:
                    f.write(output_text)
                st.success("Analysis saved as analysis_output.txt")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Detailed traceback:")
        st.text(traceback.format_exc())
