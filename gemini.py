import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai
import pandas as pd

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

    def resize_image(image, max_size=(500, 500)):
        image.thumbnail(max_size)
        return image

    def analyze_image(uploaded_file):
        prompt = (
            "Analyze the provided image for various marketing aspects. Respond in single words or short phrases separated by commas for each attribute: "
            "text amount (High or Low), color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), focus (Central message or Scattered), "
            "customer-centric (Yes or No), credibility (High or Low), user interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
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
            "Analyze the provided image for marketing effectiveness. Provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format (Aspect, Score, Explanation, Improvement). Under the table, please write the total sum of scores. Here are the aspects to consider:\n"
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
            st.write(f"Raw response: {raw_response}")  # Debug: display raw response
            aspects = raw_response.split("\n\n")

            results = []
            for aspect in aspects:
                lines = aspect.split("\n")
                aspect_dict = {
                    "Aspect": "N/A",
                    "Score": "N/A",
                    "Explanation": "N/A",
                    "Improvement": "N/A"
                }
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        if key.startswith("Aspect"):
                            aspect_dict["Aspect"] = value
                        elif key.startswith("Score"):
                            aspect_dict["Score"] = value
                        elif key.startswith("Explanation"):
                            aspect_dict["Explanation"] = value
                        elif key.startswith("Improvement"):
                            aspect_dict["Improvement"] = value
                if any(value != "N/A" for value in aspect_dict.values()):
                    results.append(aspect_dict)

            if results:
                st.write("Detailed Marketing Analysis Results:")
                st.table(pd.DataFrame(results))
            else:
                st.error("Error: Unable to parse detailed marketing analysis results.")
        else:
            st.error("Unexpected response structure from the model.")
        return None

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

    def headline_analysis(uploaded_file):
        prompt = (
            "Analyze the provided image headline for its effectiveness based on the following criteria. For each criterion, provide a score from 1 to 5 (1 being poor and 5 being excellent) and a short, concise explanation. Also, provide 3 possible short suggestions for improved headlines. The results should be presented in a table format (Criterion, Score, Explanation, Improvement). Under the table, please write the total sum of scores. Here are the criteria to consider:\n"
            "1. Clarity & Concision: How clearly does the headline convey the main point?\n"
            "2. Customer Focus: Does the headline emphasize a customer-centric approach?\n"
            "3. Relevance: How accurately does the headline reflect the content?\n"
            "4. Keywords: Are relevant SEO keywords included naturally?\n"
            "5. Emotional Appeal: Does the headline evoke curiosity or an emotional response?\n"
            "6. Uniqueness: How original and creative is the headline?\n"
            "7. Urgency & Curiosity: Does the headline create a sense of urgency or pique curiosity?\n"
            "8. Benefit-Driven: Does the headline convey a clear benefit or value proposition?\n"
            "9. Target Audience: Is the headline tailored to resonate with the specific target audience?\n"
            "10. Length & Format: Does the headline fall within an ideal length of 6-12 words?\n"
            "11. Numbers & Lists: Does the headline effectively use numbers or a list format?\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write(f"Raw response: {raw_response}")  # Debug: display raw response
            criterias = raw_response.split("\n\n")

            results = []
            for criteria in criterias:
                lines = criteria.split("\n")
                criteria_dict = {
                    "Criterion": "N/A",
                    "Score": "N/A",
                    "Explanation": "N/A",
                    "Improvement": "N/A"
                }
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        if key.startswith("Criterion"):
                            criteria_dict["Criterion"] = value
                        elif key.startswith("Score"):
                            criteria_dict["Score"] = value
                        elif key.startswith("Explanation"):
                            criteria_dict["Explanation"] = value
                        elif key.startswith("Improvement"):
                            criteria_dict["Improvement"] = value
                if any(value != "N/A" for value in criteria_dict.values()):
                    results.append(criteria_dict)

            if results:
                st.write("Headline Analysis Results:")
                st.table(pd.DataFrame(results))
            else:
                st.error("Error: Unable to parse detailed headline analysis results.")
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
        headline_analysis_button = st.button('Headline Analysis')

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
                marketing_success_result = marketing_effectiveness(uploaded_file)
                if marketing_success_result:
                    st.write("## Marketing Effectiveness Analysis Results:")
                    st.write(marketing_success_result)

        if headline_analysis_button:
            with st.spinner("Performing headline analysis..."):
                uploaded_file.seek(0)
                headline_result = headline_analysis(uploaded_file)
                if headline_result:
                    st.write("## Headline Analysis Results:")
                    st.table(pd.DataFrame(headline_result))
