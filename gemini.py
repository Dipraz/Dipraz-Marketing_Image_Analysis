import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import io
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
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize Generative AI model with generation configuration
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )

    def resize_image(image, max_size=(500, 500)):
        image.thumbnail(max_size)
        return image

    def analyze_image(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image for various marketing aspects and ensure your results remain consistent for each aspect, regardless of how many times you analyze the image. Respond in single words or short phrases separated by commas for each attribute: "
            "text amount (High or Low), color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), focus (Central message or Scattered), "
            "customer-centric (Yes or No), credibility (High or Low), user interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
        )
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            attributes = ["text_amount", "color_usage", "visual_cues", "emotion", "focus", "customer_centric", "credibility", "user_interaction", "cta_presence", "cta_clarity"]
            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                values = raw_response.split(',')
                if len(attributes) == len(values):
                    structured_response = {attr: val.strip() for attr, val in zip(attributes, values)}
                    return structured_response
                else:
                    st.error("Unexpected response structure from the model. Please check the prompt and model output format.")
                    return None
            else:
                st.error("Unexpected response structure from the model.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    def combined_marketing_analysis_V6(uploaded_file):
        prompt = (
            "Imagine you are a UX design and marketing analysis consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. First, provide detailed responses for the following:\n"
            "\n"
            "1. Asset Type: Clearly identify and describe the type of marketing asset. Examples include email, social media posts, advertisements, flyer, brochures, landing pages, etc.\n"
            "2. Purpose: Clearly state the specific purpose of this marketing asset. Provide a detailed explanation of how it aims to achieve this purpose. Examples include selling a product, getting more signups, driving traffic to a webpage, increasing brand awareness, engaging with customers, etc.\n"
            "3. Asset Audience: Identify the target audience for this marketing asset. Describe the demographics, interests, and needs of this audience. Examples include age group, gender, location, income level, education, interests, behaviors, etc.\n"
            "\n"
            "Then, for each aspect listed below, provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format with the columns: Aspect, Score, Explanation, and Improvement. After the table, provide the total sum of the scores and a concise explanation with overall improvement suggestions. Ensure that this analysis process yields consistent scoring results, regardless of how many times or when it is run. Here are the aspects to consider:\n"
            "\n"
            "The aspects to consider are:\n"
            "1. Creative Score: Assess the creativity of the design. Does it stand out and capture attention through innovative elements?\n"
            "2. Attention: Evaluate the order of content consumption in the uploaded image. Start by identifying and analyzing the headline for its prominence and position. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"
            "3. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
            "4. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "5. Clarity: Evaluate the clarity of the design elements. Are the visuals and text easy to understand?\n"
            "6. First Impressions: Analyze the initial impact of the design. Does it create a strong positive first impression?\n"
            "7. Cognitive Demand: Evaluate the cognitive load required to understand and navigate the design. Is it intuitive and easy to use?\n"
            "8. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
            "9. Headline SEO keywords and emotional appeal: Does the headline incorporate SEO keywords and evoke an emotional response?\n"
            "10. Visual Cues and Color Usage: Does the image use visual cues and colors to draw attention to key elements? Analyze how color choices, contrast, and elements like arrows or frames guide the viewer's attention.\n"
            "11. Labeling and Button Clarity: Are any labels or buttons present clearly labeled and easy to understand? Evaluate the use of text size, font choice, and placement for optimal readability.\n"
            "12. Engagement: Assess the engagement level of the user experience. Is the UX design captivating and satisfying to interact with?\n"
            "13. Trust: Assess the trustworthiness of the content based on visual and textual elements. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)? Assess the credibility, reliability, and intimacy conveyed by the content.\n"
            "14. Motivation: Assess the design's ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof?\n"
            "15. Influence: Analyze the influence of the design. Does it effectively persuade users and drive desired behaviors?\n"
            "16. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "17. Experience: Assess the overall user experience. How well does the design facilitate a smooth and enjoyable interaction?\n"
            "18. Memorability: Evaluate how memorable the design is. Does it leave a lasting impression?\n"
            "19. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? (1: Very Dense & Difficult, 5: Clear & Easy to Understand)\n"
        )
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Combined Marketing Analysis Results_V6:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    def combined_marketing_analysis_V7(uploaded_file, asset_type, asset_purpose, asset_audience):
        prompt = (
            f"Imagine you are a UX design and marketing analysis consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. "
            f"First, provide detailed responses for the following:\n\n"
            f"1. Asset Type: {asset_type}\n"
            f"2. Purpose: {asset_purpose}\n"
            f"3. Asset Audience: {asset_audience}\n\n"
            f"Then, for each aspect listed below, provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. "
            f"The results should be presented in a table format with the columns: Aspect, Score, Explanation, and Improvement. After the table, provide the total sum of the scores and a concise explanation with overall improvement suggestions. "
            f"Ensure that this analysis process yields consistent scoring results, regardless of how many times or when it is run. Here are the aspects to consider:\n\n"
            f"1. Creative Score: Assess the creativity of the design considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'. Does it stand out and capture attention through innovative elements?\n"
            f"2. Attention: Evaluate the order of content consumption in the uploaded image. Start by identifying and analyzing the headline for its prominence and position, considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"
            f"3. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}' when evaluating this aspect.\n"
            f"4. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric? Evaluate this considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"5. Clarity: Evaluate the clarity of the design elements. Are the visuals and text easy to understand? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"6. First Impressions: Analyze the initial impact of the design. Does it create a strong positive first impression? Take into account the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"7. Cognitive Demand: Evaluate the cognitive load required to understand and navigate the design. Is it intuitive and easy to use? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"8. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words, considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"9. Headline SEO keywords and emotional appeal: Does the headline incorporate SEO keywords and evoke an emotional response? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"10. Visual Cues and Color Usage: Does the image use visual cues and colors to draw attention to key elements? Analyze how color choices, contrast, and elements like arrows or frames guide the viewer's attention. Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"11. Labeling and Button Clarity: Are any labels or buttons present clearly labeled and easy to understand? Evaluate the use of text size, font choice, and placement for optimal readability, considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"12. Engagement: Assess the engagement level of the user experience. Is the UX design captivating and satisfying to interact with? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"13. Trust: Assess the trustworthiness of the content based on visual and textual elements. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)? Assess the credibility, reliability, and intimacy conveyed by the content, considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"14. Motivation: Assess the design's ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"15. Influence: Analyze the influence of the design. Does it effectively persuade users and drive desired behaviors? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"16. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs, considering the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"17. Experience: Assess the overall user experience. How well does the design facilitate a smooth and enjoyable interaction? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"18. Memorability: Evaluate how memorable the design is. Does it leave a lasting impression? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
            f"19. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? Consider the asset type '{asset_type}', its purpose '{asset_purpose}', and the target audience '{asset_audience}'.\n"
        )

        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Combined Marketing Analysis Results_V7:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    def text_analysis(uploaded_file):
        prompt = (
            "Imagine you are a UX design and marketing analysis consultant reviewing the text on a marketing asset (excluding the headline) for a client. Analyze the provided text using the following criteria. For each aspect, provide a score from 1 to 5 (1 being low, 5 being high) along with a concise explanation and suggestions for improvement. Present the results in a table format with the columns: Aspect, Score, Explanation, and Improvements. After the table, provide the total sum of the scores and a concise explanation with overall improvement suggestions. Ensure your scoring remains consistent for each aspect, regardless of how many times you analyze the image. Here are the aspects to consider:\n"
            "1. Clarity: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy?\n"
            "2. Customer Focus: Does the text emphasize a customer-centric approach?\n"
            "3. Engagement: Discuss the text amount, readability, grouping, use of lists, and value to the customer.\n"
            "4. Effort: Evaluate the effort required to read the text. Is it broken into manageable chunks? Does it use lists or bullet points? Is it relatively short and easy to read?\n"
            "5. Purpose and Value: Assess the clarity of the text’s purpose and value proposition. Is the message clear within a few seconds of reading?\n"
            "6. Motivation: Assess the text's ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof?\n"
            "7. Influence: Analyze the influence of the text. Does it effectively persuade users and drive desired behaviors?\n"
            "8. Depth: Does the text provide a range of depth of information (including links to further information) to address different interest levels of customers?\n"
            "9. Trust: Assess the trustworthiness of the content based on the text. Evaluate the credibility, reliability, and intimacy conveyed. Is the content brand-centric or customer-centric?\n"
            "10. Memorability: Evaluate how memorable the text is. Does it leave a lasting impression?\n"
            "11. Emotional Appeal: Does the text evoke an emotional response?\n"
            "12. Uniqueness: Is the text original and creative?\n"
            "13. Urgency & Curiosity: Does the text create a sense of urgency or pique curiosity?\n"
            "14. Benefit-Driven: Does the text convey clear benefits or value propositions?\n"
            "15. Target Audience: Is the text tailored to resonate with the specific target audience?\n"
            "16. Cognitive Demand: Evaluate the cognitive load required to read, understand, and navigate the text. Is it intuitive and easy to use?\n"
            "17. Reading Age: Assess the reading age level required to understand the text.\n"
            "Conclude with three alternative versions of the text that align better with the image content and effectively address the identified weaknesses." 
        )
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Text Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    def headline_analysis(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image and its headline for a client. Analyze the provided image content alongside the headline text to assess the headline's effectiveness. Rate each criterion on a scale from 1 to 5 (1 being poor, 5 being excellent), and provide a concise explanation for each score. Also, suggest improvements for each criterion. Present your results in a table format with columns labeled: Criterion, Score, Explanation, Improvements. Below the table, calculate and display the total sum of all scores. Ensure that this analysis process yields consistent scoring results, regardless of how often or when it is run. Conclude with three possible improved headlines that better align with the image content. The improved headlines should not contain colons (':') and should vary in structure and style.\n"
            "The criteria to assess are:\n"
            "1. Clarity: How clearly does the headline convey the main point?\n"
            "2. Customer Focus: Does the headline emphasize a customer-centric approach?\n"
            "3. Relevance: How accurately does the headline reflect the content of the image?\n"
            "4. Emotional Appeal: Does the headline evoke curiosity or an emotional response, considering the image content?\n"
            "5. Uniqueness: How original and creative is the headline?\n"
            "6. Urgency & Curiosity: Does the headline create a sense of urgency or pique curiosity, considering the image?\n"
            "7. Benefit-Driven: Does the headline convey a clear benefit or value proposition, aligned with the image content?\n"
            "8. Target Audience: Is the headline tailored to resonate with the specific target audience, considering the image's visual cues?\n"
            "9. Length & Format: Does the headline fall within an ideal length of 6-12 words?\n"
        )
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Headline Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    def headline_detailed_analysis(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image and its headline for a client. As an expert, you are assessing the headline's effectiveness. Analyze the headline text extracted from the image and present the results in a table format with the following columns: Criteria, Assessment, and Explanation. Ensure the analysis is consistent across multiple runs. The criteria to assess are:\n"
            "1. Word Count: Provide the total number of words in the headline.\n"
            "2. Letter Count: Provide the total number of letters in the headline, excluding spaces and special characters. Count only alphabetic characters.\n"
            "3. Common Words: Count the number of frequently used words in the headline.\n"
            "4. Uncommon Words: Count the number of less frequently used words in the headline.\n"
            "5. Emotional Words: Count the number of words that convey emotions (positive, negative, etc.).\n"
            "6. Power Words: Count the number of words used to grab attention or influence.\n"
            "7. Sentiment: Assess the overall sentiment of the headline (positive, negative, neutral).\n"
            "8. Reading Grade Level: Provide the reading grade level of the headline text.\n"
            "After the table, provide an overall summary, including a concise explanation and some improvement suggestions."
        )
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Headline Optimization Report Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    def flash_analysis(uploaded_file):
        prompt = (
            "Imagine you are a visual content analyst reviewing an image for a client. Analyze the provided image and generate a detailed description that captures the key elements and information relevant to marketing purposes. Focus on objective details like objects, people, colors, text, and their arrangement. Additionally, identify any potential cultural references or symbols that might be relevant to the target audience. Ensure the description is consistent across multiple runs, avoiding subjective interpretations or emotional responses."
        )
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([prompt, image])
            if response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                st.error("Unexpected response structure from the model.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the image: {e}")
            return None

    # Streamlit app setup
    st.title('Marketing Image Analysis AI Assistant')
    with st.sidebar:
        st.header("Options")
        basic_analysis = st.button('Basic Analysis')
        combined_analysis_V6 = st.button('Combined Detailed Marketing Analysis V6')
        combined_analysis_V7 = st.button('Combined Detailed Marketing Analysis V7', on_click=lambda: st.session_state.update({"submitted": True}))
        text_analysis_button = st.button('Text Analysis')
        headline_analysis_button = st.button('Headline Analysis')
        detailed_headline_analysis_button = st.button('Headline Optimization Report') 
        flash_analysis_button = st.button('Flash Analysis')

    col1, col2 = st.columns(2)
    uploaded_files = col1.file_uploader("Upload your marketing image here:", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image = resize_image(image)
            col2.image(image, caption="Uploaded Image", use_column_width=True)

            if st.session_state.submitted:
                st.write("Please provide additional details for analysis:")
                st.session_state.asset_type = st.text_input("Asset Type (e.g., email, social media post):", value=st.session_state.asset_type)
                st.session_state.asset_purpose = st.text_input("Asset Purpose (e.g., selling a product, increasing brand awareness):", value=st.session_state.asset_purpose)
                st.session_state.asset_audience = st.text_input("Target Audience (e.g., age group, gender, location):", value=st.session_state.asset_audience)

                if st.button("Submit Analysis"):
                    if st.session_state.asset_type and st.session_state.asset_purpose and st.session_state.asset_audience:
                        with st.spinner("Performing detailed marketing analysis..."):
                            uploaded_file.seek(0)
                            combined_result = combined_marketing_analysis_V7(
                                uploaded_file, 
                                st.session_state.asset_type, 
                                st.session_state.asset_purpose, 
                                st.session_state.asset_audience
                            )
                            if combined_result:
                                st.write("## Detailed Marketing Analysis Results:")
                                st.markdown(combined_result)
                        # Reset submitted state after processing
                        st.session_state.submitted = False
                    else:
                        st.error("Please provide all the required details for the analysis.")

            if basic_analysis:
                with st.spinner("Performing basic analysis..."):
                    uploaded_file.seek(0)
                    basic_analysis_result = analyze_image(uploaded_file)
                    if basic_analysis_result:
                        st.write("## Basic Analysis Results:")
                        st.json(basic_analysis_result)

            if combined_analysis_V6:
                with st.spinner("Performing combined marketing analysis_V6..."):
                    uploaded_file.seek(0)
                    detailed_result_V6 = combined_marketing_analysis_V6(uploaded_file)
                    if detailed_result_V6:
                        st.write("## Combined Marketing Analysis_V6 Results:")
                        st.markdown(detailed_result_V6)

            if text_analysis_button:
                with st.spinner("Performing text analysis..."):
                    uploaded_file.seek(0)
                    text_result = text_analysis(uploaded_file)
                    if text_result:
                        st.write("## Text Analysis Results:")
                        st.markdown(text_result)                

            if headline_analysis_button:
                with st.spinner("Performing headline analysis..."):
                    uploaded_file.seek(0)
                    headline_result = headline_analysis(uploaded_file)
                    if headline_result:
                        st.write("## Headline Analysis Results:")
                        st.markdown(headline_result)

            if detailed_headline_analysis_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    uploaded_file.seek(0)
                    detailed_headline_result = headline_detailed_analysis(uploaded_file)
                    if detailed_headline_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(detailed_headline_result)

            if flash_analysis_button:
                with st.spinner("Performing Flash analysis..."):
                    uploaded_file.seek(0)
                    flash_result = flash_analysis(uploaded_file)
                    if flash_result:
                        st.write("## Flash Analysis Results:")
                        st.markdown(flash_result)
