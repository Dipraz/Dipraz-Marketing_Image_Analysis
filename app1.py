import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import openai
import base64
import io

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resize_image(image, max_size=(512, 512)):
    image.thumbnail(max_size, Image.ANTIALIAS)
    return image

# Function to analyze the image description
def analyze_image_base64(image_base64):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that provides concise structured analysis of marketing images."},
            {"role": "user", "content": f"data:image/png;base64,{image_base64}"}
        ],
        max_tokens=300,
        temperature=0.3
    )

    # Try accessing content from either 'content' or 'message.content' property
    try:
        raw_response = response.choices[0].content.strip()
    except AttributeError:
        raw_response = response.choices[0].message["content"].strip()

    attributes = ["text_amount", "color_usage", "visual_cues", "emotion", "focus", "customer_centric", "credibility", "user_interaction", "cta_presence", "cta_clarity"]
    values = raw_response.split(',')
    structured_response = {attr: val.strip() for attr, val in zip(attributes, values)}
    return structured_response

# Function for detailed marketing analysis
def detailed_marketing_analysis(image_base64):
    prompt = "Analyze the provided image for various marketing aspects. For each aspect, provide a score from 1 to 5, a short explanation, and suggestions for improvement:\n"
    prompt += "1. Attraction and Focus: Does the content prioritize important information and draw attention effectively?\n"
    prompt += "2. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
    prompt += "3. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
    prompt += "4. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
    prompt += "5. Engagement: Discuss the text amount, reading age, grouping, lists, and customer value.\n"
    prompt += "6. Trust: Assess the credibility, reliability, and intimacy conveyed by the content. Is the content brand or customer-centric?\n"
    prompt += "7. Motivation and Influence: Examine if the content aligns with user motivators, demonstrates authority, uses scarcity, and provides social proof.\n"
    prompt += "8. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
    prompt += "9. Experience and Memorability: Comment on the user interaction, content difficulty, emotion created, participation encouragement, learning styles, interactivity, context, reinforcement, practical value, and social currency.\n"
    prompt += "10. Attention: Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc).\n"
    prompt += "11. Distinction: Is the image a stock image or something unique?\n"
    prompt += "12. Memory: Does the content include a range of learning styles (e.g., image, text, infographics, video, etc)?\n"
    prompt += "13. Effort: Does the content have multiple messages and lots of text? Is it long and difficult to read?\n"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that provides detailed marketing analysis of images."},
            {"role": "user", "content": f"{prompt} data:image/png;base64,{image_base64}"}
        ],
        max_tokens=1000,
        temperature=0.5
    )
    # Access response content using .content property within .choices[0]
    try:
        return response.choices[0].content.strip()
    except AttributeError:
        return response.choices[0].message.content.strip()

# Function for cognitive load and trust analysis
def cognitive_load_analysis(image_base64):
    prompt = "Analyze the provided image based on detailed cognitive load, attention, engagement, and trust criteria:\n"
    prompt += "- Does the image prioritize important information effectively?\n"
    prompt += "- Does it use visual cues and color to highlight important information?\n"
    prompt += "- Does the image contain labels, buttons, and if so, are they clearly labeled?\n"
    prompt += "- How does the image capture and hold attention? Is it structured to draw attention to the most important areas?\n"
    prompt += "- Discuss the clarity and impact of the headline if present.\n"
    prompt += "- Evaluate the engagement level including text quantity and grouping.\n"
    prompt += "- Assess the trustworthiness of the content based on visual and textual cues."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that provides detailed cognitive load and trust analysis of images."},
            {"role": "user", "content": f"{prompt} data:image/png;base64,{image_base64}"}
        ],
        max_tokens=500,
        temperature=0.5
    )
    # Access response content using .content property within .choices[0]
    try:
        return response.choices[0].content.strip()
    except AttributeError:
        return response.choices[0].message["content"].strip()

# Streamlit app setup
st.title('Marketing Image Analysis AI Assistant')
uploaded_file = st.file_uploader("Upload your marketing image here:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_base64 = encode_image(image)

    if st.button('Basic Analysis'):
        basic_analysis_result = analyze_image_base64(image_base64)
        st.write("Basic Analysis Results:")
        st.json(basic_analysis_result)

    if st.button('Detailed Marketing Analysis'):
        detailed_result = detailed_marketing_analysis(image_base64)
        st.write("Detailed Marketing Analysis Results:")
        st.write(detailed_result)

    if st.button('Cognitive Load and Trust Analysis'):
        cognitive_result = cognitive_load_analysis(image_base64)
        st.write("Cognitive Load and Trust Analysis Results:")
        st.write(cognitive_result)
