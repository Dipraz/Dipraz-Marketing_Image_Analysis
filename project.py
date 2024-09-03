import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

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
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize Generative AI model with generation configuration
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )

    # Initialize chat history and documents in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents" not in st.session_state:
        st.session_state.documents = []

    st.title("Chat with Gemini Pro 1.5")

    # User input
    user_input = st.text_input("You:", "")

    # Document upload
    uploaded_file = st.file_uploader("Upload a document (optional)", type=["txt", "pdf"])
    if uploaded_file is not None:
        # Process the uploaded file and add its content to the documents list
        # You'll need to implement the file processing logic based on the file type
        # For example, you could use libraries like PyPDF2 for PDF processing or simply read the text from a .txt file
        st.session_state.documents.append(uploaded_file.read())

    # Generate response
    if st.button("Send") and user_input:
        # Construct the prompt with chat history and documents
        prompt = "\n".join(st.session_state.chat_history + st.session_state.documents) + "\nUser: " + user_input

        # Generate response from Gemini Pro
        response = model.generate_text(prompt)

        # Add user input and response to chat history
        st.session_state.chat_history.append("User: " + user_input)
        st.session_state.chat_history.append("Gemini: " + response.text)

    # Display chat history
    for message in st.session_state.chat_history:
        st.write(message)
