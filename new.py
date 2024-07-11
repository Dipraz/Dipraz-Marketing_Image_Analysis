import streamlit as st
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_image(prompt):
    """Generates a single image using the DALL-E API."""
    try:
        client = openai.OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,  # Specify n=1 as per API requirement
            size="1024x1024"
        )
        # Extract the first image URL from the response data
        image_url = response.data[0].url
        return image_url

    except openai.OpenAIError as e:
        st.error(f"Error generating image: {e}")
        return None

# Streamlit app layout
st.title("AI-Powered Image Generator")

st.markdown("""
### Explore Your Imagination
Enter a prompt and generate creative images using the power of AI.
""")

# Text input for the prompt
prompt = st.text_area("Enter your prompt here:", height=150)

# Sidebar for advanced settings
with st.sidebar:
    st.write("## Advanced Settings")
    st.markdown("""
    Adjust settings to fine-tune the image generation process or explore additional features.
    """)
    image_count = st.number_input("Number of images to generate", min_value=1, max_value=10, value=1)

# Button to generate images
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating images..."):
            # Generate the specified number of images
            for i in range(image_count):
                image_url = generate_image(prompt)
                if image_url:
                    st.image(image_url, caption=f"Generated Image {i+1}")
                    st.markdown(f"[Download Image {i+1}]({image_url})", unsafe_allow_html=True)
    else:
        st.warning("Please enter a prompt.")
