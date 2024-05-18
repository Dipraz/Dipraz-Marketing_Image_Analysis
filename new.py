import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
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

    # Initialize Generative AI model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    # Initialize session state variables
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'result' not in st.session_state:
        st.session_state.result = None

    # Function to resize image (if necessary)
    def resize_image(image, max_size=(500, 500)):
        image.thumbnail(max_size)
        return image

    def analyze_image(uploaded_file):
        prompt = (
            "Analyze the provided image for various marketing aspects and respond in single words or short phrases separated by commas, covering: "
            "text amount, color usage, visual cues, emotion, focus, customer centric, credibility, user interaction, CTA presence, CTA clarity."
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])
        attributes = ["text_amount", "color_usage", "visual_cues", "emotion", "focus", "customer_centric", "credibility", "user_interaction", "cta_presence", "cta_clarity"]
        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            values = raw_response.split(',')
            if len(attributes) == len(values):
                structured_response = {attr: val.strip() for attr, val in zip(attributes, values)}
                return structured_response
            else:
                st.error("Unexpected response structure from the model. Please check prompt and model output format.")
                return None
        else:
            st.error("Unexpected response structure from the model.")
            return None

    def detailed_marketing_analysis(uploaded_file):
        prompt = (
            "Analyze the provided image for marketing effectiveness. Provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format (Aspect, Score, Explanation, Improvement). Here are the aspects to consider:\n"
            "1. Attraction and Focus: Does the content prioritize important information and draw attention effectively?\n"
            "2. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
            "3. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "4. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
            "5. Engagement: Discuss the text amount, reading age, grouping, lists, and customer value.\n"
            "6. Trust: Assess the credibility, reliability, and intimacy conveyed by the content. Is the content brand or customer-centric?\n"
            "7. Motivation and Influence: Examine if the content aligns with user motivators, demonstrates authority, uses scarcity, and provides social proof.\n"
            "8. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "9. Experience and Memorability: Comment on the user interaction, content difficulty, emotion created, participation encouragement, learning styles, interactivity, context, reinforcement, practical value, and social currency.\n"
            "10. Attention: Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc).\n"
            "11. Distinction: Is the image a stock image or something unique?\n"
            "12. Memory: Does the content include a range of learning styles (e.g., image, text, infographics, video, etc)?\n"
            "13. Effort: Does the content have multiple messages and lots of text? Is it long and difficult to read?\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])
        
        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Raw response:", raw_response)  # Print raw response for debugging
            
            # Split the response into individual aspects and their details
            aspects = raw_response.split("\n\n")
            st.write("Aspects:", aspects)  # Debugging

            # Parse each aspect data further
            results = []
            for aspect in aspects:
                data = aspect.split("\n")
                st.write("Data:", data)  # Debugging
                if len(data) >= 4:  # Check if all information is present
                    try:
                        # Extract the fields
                        aspect_name = data[0].split(": ")[1]
                        score = data[1].split(": ")[1]
                        explanation = data[2].split(": ")[1]
                        improvement = data[3].split(": ")[1]
                        results.append({"Aspect": aspect_name, "Score": score, "Explanation": explanation, "Improvement": improvement})
                    except IndexError as e:
                        st.error(f"Error parsing aspect: {aspect} - {e}")

    def marketing_effectiveness(uploaded_file):
        prompt = (
            "Analyze the provided image for its marketing effectiveness based on the following detailed criteria. For each criterion, provide a score from 1 to 5 (1 being poor and 5 being excellent) and a short explanation with some improvement suggestions:\n"
            "1. Information Prioritization: Does the image effectively highlight the most important information?\n"
            "2. Visual Cues and Color Usage: Does the image use visual cues and colors to draw attention to key elements?\n"
            "3. Labeling and Button Clarity: Are any labels or buttons present clearly labeled and easy to understand?\n"
            "4. Attention Capturing: How well does the image capture and hold the viewer's attention?\n"
            "5. Headline Clarity and Impact: If a headline is present, how clear and impactful is it?\n"
            "6. Engagement Level: Evaluate the text quantity and how well the information is grouped. Is the text engaging?\n"
            "7. Trustworthiness: Assess the trustworthiness of the content based on visual and textual elements.\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            st.error("Unexpected response structure from the model.")
            return None

    # Streamlit app setup
    st.title('Marketing Image Analysis AI Assistant')

    with st.sidebar:
        st.header("Options")
        basic_analysis = st.button('Basic Analysis')
        detailed_analysis = st.button('Detailed Marketing Analysis')
        marketing_success = st.button('Marketing Success Analysis')

    col1, col2 = st.columns(2)
    uploaded_file = col1.file_uploader("Upload your marketing image here:")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = resize_image(image)
        col2.image(image, caption="Uploaded Image", use_column_width=True)

        if basic_analysis:
            with st.spinner("Performing basic analysis..."):
                uploaded_file.seek(0)
                basic_analysis_result = analyze_image(uploaded_file)
                if basic_analysis_result:
                    st.write("## Basic Analysis Results:")
                    st.json(basic_analysis_result)

        if detailed_analysis:
            with st.spinner("Performing detailed marketing analysis..."):
                uploaded_file.seek(0)
                detailed_result = detailed_marketing_analysis(uploaded_file)
                if detailed_result:
                    st.write("## Detailed Marketing Analysis Results:")
                    st.write(detailed_result)

        if marketing_success:
            with st.spinner("Performing marketing effectiveness and success analysis..."):
                uploaded_file.seek(0)
                marketing_success = marketing_effectiveness(uploaded_file)
                if marketing_success:
                    st.write("## Marketing Effectiveness Analysis Results:")
                    st.write(marketing_success)
