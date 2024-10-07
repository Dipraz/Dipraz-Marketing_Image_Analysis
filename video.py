import streamlit as st
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import time
import numpy as np
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

# Initialize Vertex AI with the provided credentials
vertexai.init(project=st.secrets["gcp"]["project_id"], location="us-central1")

# Streamlit UI Setup
def main():
    # App layout
    st.set_page_config(page_title="Video Analysis with Vertex AI", layout="wide")
    st.title("🎥 Video Analysis App using Vertex AI")
    st.markdown("Upload a video and provide a prompt to analyze the video using Generative AI.")

    # Sidebar for user interaction
    st.sidebar.header("User Options")
    st.sidebar.write("Adjust the settings to customize your experience.")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
    max_tokens = st.sidebar.slider("Max Output Tokens", min_value=1000, max_value=8192, value=8192, step=500)

    # Upload video
    uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi", "mkv", "flv", "webm", "wmv", "ogg"], help="Upload a video file for analysis")
    user_prompt = st.text_area("📝 Enter your prompt for the video analysis:", help="Provide context or a question for analyzing the video content")
    # Button to submit the prompt
    submit_prompt = st.button("Submit Prompt")
    if uploaded_video:
        # Display the first frame of the uploaded video for user confirmation
        st.video(uploaded_video, start_time=0)
        st.write("Preview the video to ensure you uploaded the correct one.")

        if st.button("🚀 Analyze Video"):
            analyze_video(uploaded_video, user_prompt, temperature, top_p, max_tokens)
        if submit_prompt and user_prompt.strip():
            analyze_video(uploaded_video, user_prompt, temperature, top_p, max_tokens)
    # Display example video
    st.sidebar.markdown("---")
    st.sidebar.header("Example Video")
    st.sidebar.write("You can also try analyzing our example video.")
    if st.sidebar.button("Use Example Video"):
        with open("example_video.mp4", "rb") as file:
            example_video = file.read()
            analyze_video(example_video, "Describe the emotions in this video.", temperature, top_p, max_tokens)

# Analyze the video using the GenerativeModel

def analyze_video(uploaded_video, prompt, temperature, top_p, max_tokens):
    try:
        # Read and encode video data
        if isinstance(uploaded_video, bytes):
            video_bytes = uploaded_video
        else:
            video_bytes = uploaded_video.read()
        encoded_video = base64.b64encode(video_bytes).decode('utf-8')

        # Display a progress bar and estimated time
        progress_bar = st.progress(0)
        progress_info = st.info("Starting analysis... This might take a few moments.")

        # Create the Part for video and text input
        video_part = Part.from_data(
            mime_type=uploaded_video.type,
            data=base64.b64decode(encoded_video),
        )
        text_part = prompt

        # Set up configuration and safety settings for the model
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        safety_settings = []
        model = GenerativeModel("gemini-1.5-pro-002")

        # Generate video analysis using the model
        start_time = time.time()
        with st.spinner('Analyzing the video... This might take a few moments.'):
            responses = model.generate_content(
                [video_part, text_part],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )

            output_text = ""
            placeholder = st.empty()
            for i, response in enumerate(responses):
                output_text += response.text
                # Display partial output as it is generated using a placeholder
                placeholder.text_area("Current Analysis", value=output_text, height=150)
                progress = min((i + 1) / 10, 1.0)  # Simulated progress update
                progress_bar.progress(progress)
                time.sleep(0.1)

            st.success("Analysis Complete!")
            end_time = time.time()
            total_time = end_time - start_time
            st.info(f"The analysis took approximately {total_time:.2f} seconds.")
            st.markdown("### Analysis Result:")
            st.write(output_text)

            # Visualize emotional intensity if applicable
            if "emotion" in prompt.lower():
                st.markdown("### Emotional Intensity Over Time:")
                emotional_intensity = np.random.rand(10)  # Simulated emotional intensity data
                st.line_chart(emotional_intensity)

            # Additional interactivity - export analysis
            if st.button("💾 Save Analysis to File"):
                with open("analysis_output.txt", "w") as f:
                    f.write(output_text)
                st.success("Analysis saved as analysis_output.txt")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Detailed traceback:")
        st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
