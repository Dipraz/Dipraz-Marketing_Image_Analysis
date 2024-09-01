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

# Check if credentials_path is set
if credentials_path is None:
    st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please check your .env file.")
else:
    # Set the Google application credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Configure the Generative AI API key
    genai.configure(api_key=api_key)

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

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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

    # User input area with placeholder and styling
    user_input = st.text_area("Enter your message or upload a file:", height=100, placeholder="Type your message here...")

    # File uploader with styling
    uploaded_file = st.file_uploader("Upload a file (image, PDF, or Excel):", type=["jpg", "jpeg", "png", "pdf", "xlsx"],
                                    accept_multiple_files=False,
                                    label_visibility="collapsed")

    # Process user input and generate response with visual feedback
    if st.button("Send") or (user_input and st.session_state.get('enter_pressed')):
        if user_input or uploaded_file:
            file_content = None

            # Handle file uploads
            if uploaded_file:
                if uploaded_file.type.startswith("image/"):
                    image = Image.open(uploaded_file)
                    image = convert_to_rgb(image)
                    image = resize_image(image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    file_content = image

                elif uploaded_file.type == "application/pdf":
                    from PyPDF2 import PdfReader
                    pdf_reader = PdfReader(uploaded_file)
                    file_content = ""
                    for page in pdf_reader.pages:
                        file_content += page.extract_text()

                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    df = pd.read_excel(uploaded_file)
                    file_content = df.to_csv(index=False)

            # Generate Gemini response (include file content if available)
            if file_content:
                if isinstance(file_content, Image.Image):
                    response = model.generate_image(prompt=user_input, image=file_content)
                    st.image(response, caption="Generated Image", use_column_width=True)
                else:
                    response = model.generate_content([user_input, file_content])
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            else:
                response = model.generate_content(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

        # Add user input/file to chat history
        st.session_state.messages.append({"role": "user", "content": user_input or "Uploaded a file"})

        # Display messages in the chat container with styling
        with st.container():
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    if isinstance(response, str):
                        with st.spinner("Generating response..."):
                            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.image(response, caption="Generated Image", use_column_width=True)

    # Clear the input area and reset the 'enter_pressed' state
    st.session_state['enter_pressed'] = False
    user_input = st.empty()

# Detect if the Enter key was pressed in the input area
if user_input and st.session_state.get('enter_pressed') is None:
    st.session_state['enter_pressed'] = True

# Image processing functions
def convert_to_rgb(image):
    """Convert an image to RGB format if it is not already."""
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def resize_image(image, max_size=(300, 250)):
    image.thumbnail(max_size)
    return image
