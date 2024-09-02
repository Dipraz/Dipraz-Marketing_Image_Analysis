import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai
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
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 64,
        "max_output_tokens": 2000,
        "response_mime_type": "text/plain",
    }

    # Initialize Generative AI model with generation configuration
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )

    st.title("🖼️📄🤖 Interactive AI Assistant with Files")
    st.markdown("""
        Talk with your files using the power of AI! Upload images, PDFs, or Excel files, 
        and ask specific questions to interact with the content in a dynamic way. 🚀
    """)

    # Chat section
    st.subheader("🤖 General Chat with Gemini")
    chat_history = st.session_state.get('chat_history', [])
    user_input = st.text_input("Ask Gemini a general question:", key="chat_input")

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
    st.subheader("📂 Upload a File and Chat with it")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["png", "jpg", "jpeg", "pdf", "xlsx", "xls"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            file_chat_input = st.text_input("Ask a question about the image:", key="image_chat_input")
            if st.button("Send Image Question", key="image_send_button"):
                with st.spinner("Analyzing the image..."):
                    response = model.generate_content([f"Question: {file_chat_input}", image], generation_config=generation_config)
                    st.write(f"**Gemini:** {response.text}")

        elif file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            st.write("PDF content loaded successfully!")

            file_chat_input = st.text_area("Ask a question about the PDF content:", key="pdf_chat_input")
            if st.button("Send PDF Question", key="pdf_send_button"):
                with st.spinner("Analyzing the PDF..."):
                    response = model.generate_content(f"PDF Content: {text}\nQuestion: {file_chat_input}", generation_config=generation_config)
                    st.write(f"**Gemini:** {response.text}")

        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            excel_data = pd.read_excel(uploaded_file)
            st.write("Excel data loaded successfully!")
            st.write(excel_data)

            file_chat_input = st.text_input("Ask a question about the Excel data:", key="excel_chat_input")
            if st.button("Send Excel Question", key="excel_send_button"):
                with st.spinner("Analyzing the Excel data..."):
                    response = model.generate_content(f"Excel Data: {excel_data.head().to_json()}\nQuestion: {file_chat_input}", generation_config=generation_config)
                    st.write(f"**Gemini:** {response.text}")

    st.markdown("---")
    st.markdown("Developed by Dipraz...")
