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

# Load credentials from Streamlit secrets and write to a file
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

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Initialize Vertex AI
vertexai.init(project=st.secrets["gcp"]["project_id"], location="us-central1")

def analyze_video(uploaded_video, prompt, temperature, top_p, max_tokens):
    try:
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

            if st.button("💾 Save Analysis to File"):
                with open("analysis_output.txt", "w") as f:
                    f.write(output_text)
                st.success("Analysis saved as analysis_output.txt")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Detailed traceback:")
        st.text(traceback.format_exc())

import streamlit as st
import base64
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part

# Existing initialization
aiplatform.init(project="your-project-id", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-001")

def analyze_video_inline(video_file, prompt):
    """Analyze video by passing it inline to Gemini."""
    video_data = video_file.read()
    video_encoded = base64.b64encode(video_data).decode("utf-8")
    video_part = Part.from_data(data=video_data, mime_type=video_file.type)
    response = model.generate_content([prompt, video_part])
    return response.text

def main():
    st.title("Image and Video Analysis with Gemini")
    analysis_type = st.radio("Select analysis type:", ["Image", "Video"])

    if analysis_type == "Image":
        # Existing image analysis code
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        prompt = st.text_area("Enter your prompt:")
        if st.button("Analyze Image") and uploaded_file and prompt:
            image_part = Part.from_data(uploaded_file.read(), mime_type=f"image/{uploaded_file.type.split('/')[-1]}")
            response = model.generate_content([prompt, image_part])
            st.write(response.text)

    elif analysis_type == "Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        prompt = st.text_area("Enter your prompt for the video:")
        if st.button("Analyze Video") and uploaded_video and prompt:
            with st.spinner("Analyzing video..."):
                result = analyze_video_inline(uploaded_video, prompt)
                st.write("Video Analysis Result:")
                st.write(result)

if __name__ == "__main__":
    main()
def main():
    st.set_page_config(page_title="Marketing Media Analysis AI Assistant", layout="wide")
    st.title("🧠 Marketing Media Analysis AI Assistant")

    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis"])

    with tab1:
        st.subheader("Coming Soon: Image Analysis using Gemini")
        st.info("The image analysis feature is under development.")

    with tab2:
        st.subheader("Video Analysis with Vertex AI")

        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
        top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.95, 0.05)
        max_tokens = st.sidebar.slider("Max Output Tokens", 1000, 8192, 4096, 500)

        uploaded_video = st.file_uploader("Upload a video for analysis", type=["mp4", "mov", "avi", "mkv", "webm"], key="video")
        user_prompt = st.text_area("Enter your prompt for the video analysis", key="video_prompt")

        if uploaded_video:
            st.video(uploaded_video)

        if uploaded_video and user_prompt:
            if st.button("Analyze Video"):
                analyze_video(uploaded_video, user_prompt, temperature, top_p, max_tokens)

if __name__ == "__main__":
    main()
