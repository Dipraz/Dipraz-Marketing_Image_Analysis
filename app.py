import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import openai
import base64
import io

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_image(image):
    """
    Encode a PIL image to a base64 string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image_base64(image_base64):
    """
    Perform a basic analysis of the image using OpenAI's GPT-4.
    """
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Analyze this marketing image: data:image/png;base64,{image_base64}",
        max_tokens=1000,
        temperature=0.3
    )
    raw_response = response.choices[0].text.strip()
    attributes = ["text_amount", "color_usage", "visual_cues", "emotion", "focus", "customer_centric", "credibility", "user_interaction", "cta_presence", "cta_clarity"]
    values = raw_response.split(',')
    structured_response = {attr: val.strip() for attr, val in zip(attributes, values)}
    return structured_response

def detailed_marketing_analysis(image_base64):
    """
    Perform a detailed marketing analysis of the image.
    """
    prompt = (
        "Analyze the provided image for marketing effectiveness. Provide a score from 1 to 5 "
        "(1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. "
        "The results should be presented in a table format (Aspect, Score, Explanation, Improvement). Here are the aspects to consider:\n"
        "1. Attraction and Focus\n"
        "2. Distinction\n"
        "3. Purpose and Value\n"
        "4. Headline Review\n"
        "5. Engagement\n"
        "6. Trust\n"
        "7. Motivation and Influence\n"
        "8. Calls to Action\n"
        "9. Experience and Memorability\n"
        "10. Attention\n"
        "11. Distinction\n"
        "12. Memory\n"
        "13. Effort\n"
    )
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"{prompt}\ndata:image/png;base64,{image_base64}",
        max_tokens=1500,
        temperature=0.5
    )
    return response.choices[0].text.strip()

def cognitive_load_analysis(image_base64):
    """
    Perform a cognitive load and trust analysis of the image.
    """
    prompt = (
        "Analyze the provided image for its marketing effectiveness based on the following detailed criteria. "
        "For each criterion, provide a score from 1 to 5 (1 being poor and 5 being excellent) and a short explanation "
        "with some improvement suggestions:\n"
        "- Prioritization of important information\n"
        "- Use of visual cues and color\n"
        "- Clarity of labels and buttons\n"
        "- Attention capture and structure\n"
        "- Headline clarity and impact\n"
        "- Engagement level\n"
        "- Trustworthiness of content"
    )
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"{prompt}\ndata:image/png;base64,{image_base64}",
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].text.strip()

# Streamlit app setup
st.title('Marketing Image Analysis AI Assistant (GPT-4)')

uploaded_file = st.file_uploader("Upload your marketing image here:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_base64 = encode_image(image)

    if st.button('Basic Analysis'):
        with st.spinner("Analyzing..."):
            basic_analysis_result = analyze_image_base64(image_base64)
            st.subheader("Basic Analysis Results:")
            st.json(basic_analysis_result)

    if st.button('Detailed Marketing Analysis'):
        with st.spinner("Analyzing..."):
            detailed_result = detailed_marketing_analysis(image_base64)
            st.subheader("Detailed Marketing Analysis Results:")
            st.text(detailed_result)

    if st.button('Cognitive Load and Trust Analysis'):
        with st.spinner("Analyzing..."):
            cognitive_result = cognitive_load_analysis(image_base64)
            st.subheader("Cognitive Load and Trust Analysis Results:")
            st.text(cognitive_result)
