import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd
import easyocr
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Get the API key and credentials file from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# Function to clear chat history
def clear_chat():
    st.session_state["messages"] = []
    st.session_state["uploaded_files"] = []

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Excel files
def extract_text_from_excel(file):
    df = pd.read_excel(file)
    text = df.to_string(index=False)
    return text

# Initialize the easyocr reader
reader = easyocr.Reader(['en'])  # Specify the language(s) you want to support

# Function to extract text from image files using easyocr
def extract_text_from_image(file):
    image = Image.open(file)
    result = reader.readtext(image)
    text = " ".join([res[1] for res in result])  # Concatenate detected text
    return text

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    file_texts = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            file_texts.append(extract_text_from_pdf(file))
        elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            file_texts.append(extract_text_from_excel(file))
        elif file.type in ["image/jpeg", "image/png"]:
            file_texts.append(extract_text_from_image(file))
        else:
            st.warning(f"Unsupported file type: {file.type}")
    return "\n".join(file_texts)

# Check if credentials_path is set
if credentials_path is None:
    st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please check your .env file.")
else:
    # Set the Google application credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Configure the Generative AI API key
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.6,  # Encourage creativity and diversity in responses
        "top_p": 0.8,        # Sample from the top 80% most likely tokens, balancing focus and variety
        "top_k": 40,          # Consider the top 40 tokens at each step
        "max_output_tokens": 2048, # Limit responses to a reasonable length to maintain interactivity
        "response_mime_type": "text/plain",
    }

    # Initialize Generative AI model and chat session
    try:
        chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
        ).start_chat(history=[])
    except Exception as e:
        st.error(f"Failed to initialize the Generative AI model: {e}")
        chat_session = None

    def generate_response(prompt, uploaded_files):
        if chat_session is None:
            return "Error: Model not initialized."

        # Extract text from the uploaded files
        file_contents = process_uploaded_files(uploaded_files)

        # Construct the prompt with chat history and file content
        full_prompt = "\n".join(
            [f"User: {message['content']}" for message in st.session_state.messages]
            + [f"Content from uploaded files:\n{file_contents}"]  # Include file contents in the prompt
            + [f"User: {prompt}"]
        )

        try:
            # Send message and get response using the chat session
            response = chat_session.send_message(full_prompt)
            return response.text
        except AttributeError as e:
            st.error(f"An error occurred: {e}")
            return "Error: Could not generate response."
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return "Error: Could not generate response."

    # Title of the app
    st.title("📄 Chat with Gemini Pro 1.5")
    st.markdown("Welcome to the Gemini Pro 1.5 Chat Interface! You can upload PDFs, Excel files, or images, and start chatting based on their contents.")

    # Sidebar for navigation
    with st.sidebar:
        st.header("Options")
        if st.button("Clear Chat"):
            clear_chat()
            st.success("Chat history cleared!")

    # File uploader widget
    st.markdown("### Upload Files")
    uploaded_files = st.file_uploader(
        "You can upload files here (PDF, Excel, Images).",
        type=["pdf", "xls", "xlsx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Process files and give feedback
    if uploaded_files:
        st.session_state.uploaded_files.extend(uploaded_files)
        st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")

    # Display chat history
    st.markdown("### Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if prompt := st.chat_input("Your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, st.session_state.uploaded_files)
                st.write(response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}")
