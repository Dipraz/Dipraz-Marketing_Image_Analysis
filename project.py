import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

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
        "temperature": 0.5,  # Adjust as needed
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192,
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

        # Construct the prompt including uploaded files information
        file_descriptions = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                file_descriptions.append(f"PDF file '{file.name}' is available for reference.")
            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                file_descriptions.append(f"Excel file '{file.name}' is available for reference.")
            elif file.type.startswith("image/"):
                file_descriptions.append(f"Image file '{file.name}' is available for reference.")
            else:
                file_descriptions.append(f"File '{file.name}' of type '{file.type}' is available, but might not be fully supported.")

        # Construct the full prompt including chat history and file references
        full_prompt = "\n".join(
            [f"User: {message['content']}" for message in st.session_state.messages]
            + file_descriptions
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
    st.title("Chat with Gemini Pro 1.5")

    # Clear chat button
    if st.button("Clear Chat"):
        clear_chat()

    # File uploader widget
    uploaded_files = st.file_uploader(
        "Upload files (optional)",
        type=["txt", "pdf", "xls", "xlsx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Store uploaded files in session state
    if uploaded_files:
        st.session_state.uploaded_files.extend(uploaded_files)

    # Display chat history
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
        st.session_state.messages.append({"role": "assistant", "content": response})
