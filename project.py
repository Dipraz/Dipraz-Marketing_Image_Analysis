import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
from PyPDF2 import PdfReader
import docx
import pandas as pd
import google.generativeai as genai
import io

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

    # Streamlit interface setup
    st.title("Gemini Pro 1.5 Chat Interface")
    st.write("Chat with the Gemini Pro 1.5 model. You can also upload documents to analyze.")

    # Initialize chat history in session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Text input for user messages
    user_input = st.text_input("You: ", key='input')

    # Send user message to Gemini Pro 1.5 API and get a response
    if user_input:
        chat_session = model.start_chat(history=st.session_state['history'])
        response = chat_session.send_message(user_input)
        response_message = response.text

        # Store the interaction in session state
        st.session_state['history'].append({"role": "user", "parts": [user_input]})
        st.session_state['history'].append({"role": "model", "parts": [response_message]})

    # Display chat history
    for message in st.session_state['history']:
        if message['role'] == 'user':
            st.write(f"You: {message['parts'][0]}")
        else:
            st.write(f"Gemini Pro 1.5: {message['parts'][0]}")

    # File uploader for document input
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, Excel, Image)", type=['pdf', 'docx', 'xlsx', 'jpg', 'jpeg', 'png'])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            # Process PDF
            pdf_reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            st.write("Document content (PDF):")
            st.write(text)
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Process DOCX
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            st.write("Document content (DOCX):")
            st.write(text)
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # Process Excel
            df = pd.read_excel(uploaded_file)
            st.write("Document content (Excel):")
            st.dataframe(df)
        
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            # Process Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Optional: Perform image-related operations using AI model

        # Optional: Send content to Gemini Pro 1.5 for analysis or summary
        if uploaded_file.type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            response = model.generate_content(text, generation_config=generation_config)
            st.write("Analysis by Gemini Pro 1.5:")
            st.write(response.text)
