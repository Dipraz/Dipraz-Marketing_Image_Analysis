import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai
import tempfile
import pandas as pd
import PyPDF2

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

    st.title("💬 Interactive AI Chat and File Analysis with Gemini API")
    st.markdown("""
        Welcome to the **Gemini AI Chat and File Processor**! 
        You can chat with our AI, upload files for processing, and enjoy a seamless interaction experience. 🚀
    """)

    # Chat section
    st.subheader("🤖 Chat with Gemini")
    chat_history = st.session_state.get('chat_history', [])
    user_input = st.text_input("Ask me anything:", key="chat_input")
    
    if st.button("Send", key="send_button"):
        if user_input:
            # Update chat history with user message
            chat_history.append(("You", user_input))
            st.session_state.chat_history = chat_history
            
            # Show typing animation
            with st.spinner("Gemini is thinking..."):
                response = model.start_chat().send_message(user_input)
                chat_history.append(("Gemini", response.text))
                st.session_state.chat_history = chat_history

    # Display chat history
    for sender, message in chat_history:
        if sender == "You":
            st.write(f"**🧑‍💻 You:** {message}")
        else:
            st.write(f"**🤖 Gemini:** {message}")

    # File upload section
    st.subheader("📂 Upload a File")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["png", "jpg", "jpeg", "pdf", "xlsx", "xls"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # Display progress bar
        progress_bar = st.progress(0)

        if file_type in ["image/png", "image/jpeg"]:
            progress_bar.progress(30)
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            progress_bar.progress(60)

            with st.spinner("Analyzing image..."):
                response = model.generate_content(["Describe this image.", image], generation_config=generation_config)
                st.write(response.text)
            progress_bar.progress(100)

        elif file_type == "application/pdf":
            progress_bar.progress(30)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            st.write(text)
            progress_bar.progress(60)

            with st.spinner("Summarizing PDF content..."):
                response = model.generate_content(f"Summarize the following content: {text}", generation_config=generation_config)
                st.write(response.text)
            progress_bar.progress(100)

        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            progress_bar.progress(30)
            excel_data = pd.read_excel(uploaded_file)
            st.write(excel_data)
            progress_bar.progress(60)

            with st.spinner("Analyzing Excel data..."):
                response = model.generate_content(f"Analyze the following data: {excel_data.head().to_json()}", generation_config=generation_config)
                st.write(response.text)
            progress_bar.progress(100)

        # Reset progress bar after processing
        progress_bar.empty()

    st.markdown("---")
    st.markdown("Developed with ❤️ by your AI assistant team.")
