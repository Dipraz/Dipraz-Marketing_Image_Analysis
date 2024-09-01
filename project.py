import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import io
import google.generativeai as genai
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the API key and credentials file from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Set the Google application credentials
if credentials_path is None:
    st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please check your .env file.")
else:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    genai.configure(api_key=api_key)  # Configure the Generative AI API key

    # Define generation configuration
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize Generative AI model with generation configuration
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )

# Streamlit app layout with styling
st.set_page_config(page_title="Chat with Gemini (Advanced)", page_icon=":robot_face:")
st.markdown("""
<style>
.chat-container {
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
    max-height: 500px; /* Adjust as needed */
    overflow-y: auto;
}
.user-message {
    background-color: #e9f5ff;
    padding: 5px;
    margin-bottom: 5px;
    border-radius: 3px;
    align-self: flex-end;
}
.assistant-message {
    background-color: #f0f0f0;
    padding: 5px;
    margin-bottom: 5px;
    border-radius: 3px;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

st.title("Chat with Gemini (Advanced)")
user_input = st.text_area("Enter your message or upload a file:", height=100, placeholder="Type your message here...")
uploaded_file = st.file_uploader("Upload a file (image, PDF, or Excel):", type=["jpg", "jpeg", "png", "pdf", "xlsx"])

def handle_uploaded_file(uploaded_file):
    if uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        image = convert_to_rgb(image)
        image = resize_image(image)
        return image  # Return the processed image

    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(uploaded_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() if page.extract_text() else ""
        return text_content  # Return the extracted text

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        return df.to_csv(index=False)  # Return the CSV converted content

if st.button("Send"):
    if user_input:
        response = model.generate_content(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    elif uploaded_file:
        file_content = handle_uploaded_file(uploaded_file)
        if isinstance(file_content, Image.Image):
            response = model.generate_image(prompt="Visual analysis requested.", image=file_content)
            st.image(response, caption="Generated Image", use_column_width=True)
        else:
            response = model.generate_content(file_content)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    st.session_state.messages.append({"role": "user", "content": user_input or "Uploaded a file"})

    # Display chat history in the app
    for message in st.session_state.messages:
        st.markdown(f'<div class="{message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)
