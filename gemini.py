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

    # Define generation configuration
    generation_config = {
        "temperature": 0.3,
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
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image for various marketing aspects. Respond in single words or short phrases separated by commas for each attribute: "
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
                st.error("Unexpected response structure from the model. Please check the prompt and model output format.")
                return None
        else:
            st.error("Unexpected response structure from the model.")
            return None

    def detailed_marketing_analysis(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. For each aspect listed below, provide a score from 1 to 5 (1 being low, 5 being high) along with a concise explanation for all aspects and suggestions for improvement. Present the results in a table format with the columns: Aspect, Score, Explanation, and Improvement. Also, after the table, write the total sum of the scores and a concise explanation with some overall improvement suggestions. Ensure your scoring remains consistent for each aspect, regardless of how many times you analyze the image.\n"
            "The aspects to consider are:\n"
            "1. Order of Content Consumption: Analyze the uploaded image to understand the order of content consumption. Start by identifying and analyzing the headline for its prominence and position. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"
            "2. Visual Appeal: Does the image contain visually appealing elements that grab user attention? Does it appeal on its own, without text?\n"
            "3. Purpose & Value Clarity: Is the purpose and value proposition of the content clear within 3 seconds of viewing the image? Is the content product or customer-centric?\n"
            "4. Headline (if present):\n"
            "   - Clarity & Conciseness: Evaluate the headline's clarity and conciseness.\n"
            "   - Customer Focus: Does the headline emphasize a customer-centric approach?\n"
            "   - Other sub-questions like SEO keywords, emotional appeal, etc.\n"
            "5. Text Analysis (if present):\n"
            "   - Text Quantity: Evaluate the amount of text present in the image.\n"
            "   - Text Grouping & Readability: How is the text grouped and formatted for readability?\n"
            "6. Trustworthiness: Assess the trustworthiness of the content based on visual and textual elements. Evaluate the credibility, reliability, and intimacy conveyed by the content. Determine if the content is brand-centric or customer-centric, noting that customer-centric content generally has higher trustworthiness.\n"
            "7. Motivation & Influence: Examine if the content aligns with user motivators, demonstrates authority, uses scarcity, and provides social proof.\n"
            "8. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "9. Experience & Memorability: Comment on the user interaction, content difficulty, emotion created, participation encouragement, learning styles, interactivity, context, reinforcement, practical value, and social currency.\n"
            "10. Attraction & Focus: Does the content prioritize important information and draw attention effectively?\n"
            "11. Memory: Does the content include a range of learning styles (e.g., image, text, infographics, video, etc)?\n"
            "12. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? (1: Very Dense & Difficult, 5: Clear & Easy to Understand)\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Detailed Marketing Analysis Results:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None
    
    def detailed_marketing_analysis_v1(uploaded_file):
        prompt = (
            "Analyze the provided image for marketing effectiveness. For each aspect listed below, provide a score from 1 to 5 (1 being low, 5 being high) along with a concise explanation and suggestions for improvement. Present the results in a table format with the columns: Aspect, Score, Explanation, and Improvement. Also after the table, please write the total sum of the scores and a concise explanation with some improvement overall suggestions. Ensure that this analysis method delivers consistent results mainly score, regardless of how many times or when it is run. The aspects to consider are: \n"
            "1. Attention: Evaluate the order of content consumption in the uploaded image. Start by identifying and analyzing the headline for its prominence and position. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"
            "2. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
            "3. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "4. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
            "5. Engagement: Discuss the text amount, reading age, grouping, lists, and customer value.\n"
            "6. Trust: Assess the trustworthiness of the content based on visual and textual elements. Evaluate the credibility, reliability, and intimacy conveyed by the content. Determine if the content is brand-centric or customer-centric, noting that customer-centric content generally has higher trustworthiness? \n"
            "7. Motivation & Influence: Examine if the content aligns with user motivators, demonstrates authority, uses scarcity, and provides social proof.\n"
            "8. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "9. Experience and Memorability: Comment on the user interaction, content difficulty, emotion created, participation encouragement, learning styles, interactivity, context, reinforcement, practical value, and social currency.\n"
            "10. Attraction and Focus: Does the content prioritize important information and draw attention effectively?\n"
            "11. Memory: Does the content include a range of learning styles (e.g., image, text, infographics, video, etc)?\n"
            "12. Effort: Does the content have multiple messages and lots of text? Is it long and difficult to read?\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Detailed Marketing Analysis Results_V1:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None
    
    def detailed_marketing_analysis_v2(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. Provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format (Aspect, Score, Explanation, Improvement).Also after the table, please write the total sum of the scores and a concise explanation with some improvement overall suggestions. Here are the aspects to consider:\n"
            "1. Attention: How well does the image capture and hold the viewer's attention? Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc). Does the content prioritize important information and draw attention effectively?\n"
            "2. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
            "3. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "4. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
            "5. Engagement: Discuss the text amount, reading age, grouping, lists, and customer value.\n"
            "6. Trust: Assess the trustworthiness of the content based on visual and textual elements. Assess the credibility, reliability, and intimacy conveyed by the content. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)?\n"
            "7. Motivation and Influence: Examine if the content aligns with user motivators, demonstrates authority, uses scarcity, and provides social proof.\n"
            "8. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "9. Experience and Memorability: Comment on the user interaction, content difficulty, emotion created, participation encouragement, learning styles, interactivity, context, reinforcement, practical value, and social currency.\n"
            "10. Attraction and Focus: Does the content prioritize important information and draw attention effectively?\n"
            "11. Distinction: Is the image a stock image or something unique?\n"
            "12. Memory: Does the content include a range of learning styles (e.g., image, text, infographics, video, etc)?\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Detailed Marketing Analysis Results_V1:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None
    
    def combined_marketing_analysis_V5(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. First, provide detailed responses for the following:\n"
            "\n"
            "1. Asset Type: Clearly identify and describe the type of marketing asset. Examples include email, social media post, advertisement, flyer, brochure, landing page, etc.\n"
            "2. Purpose: Clearly state the specific purpose of this marketing asset. Examples include selling a product, getting more signups, driving traffic to a webpage, increasing brand awareness, engaging with customers, etc.\n"
            "\n"
            "Then, for each aspect listed below, provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format with the columns: Aspect, Score, Explanation, and Improvement. After the table, provide the total sum of the scores and a concise explanation with some overall improvement suggestions. Here are the aspects to consider:\n"
            "\n"
            "The aspects to consider are:\n"
            "1. Attention: Evaluate the order of content consumption in the uploaded image. Start by identifying and analyzing the headline for its prominence and position. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"
            "2. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
            "3. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "4. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
            "5. Engagement: Discuss the text amount, reading age, grouping, lists, and customer value.\n"
            "6. Trust: Assess the trustworthiness of the content based on visual and textual elements. Assess the credibility, reliability, and intimacy conveyed by the content. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)?\n"
            "7. Motivation & Influence: Examine if the content aligns with user motivators, demonstrates authority, uses scarcity, and provides social proof.\n"
            "8. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "9. Experience & Memorability: Comment on the user interaction, content difficulty, emotion created, participation encouragement, learning styles, interactivity, context, reinforcement, practical value, and social currency.\n"
            "10. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? (1: Very Dense & Difficult, 5: Clear & Easy to Understand)\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])
        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Detailed Marketing Analysis Results_V1:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None
    
    def combined_marketing_analysis_V6(uploaded_file):
        prompt = (
            "Imagine you are a UX design and marketing analysis consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. First, provide detailed responses for the following:\n"
            "\n"
            "1. Asset Type: Clearly identify and describe the type of marketing asset. Examples include email, social media posts, advertisements, flyer, brochures, landing pages, etc.\n"
            "2. Purpose: Clearly state the specific purpose of this marketing asset. Examples include selling a product, getting more signups, driving traffic to a webpage, increasing brand awareness, engaging with customers, etc.\n"
            "\n"
            "Then, for each aspect listed below, provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format with the columns: Aspect, Score, Explanation, and Improvement. After the table, provide the total sum of the scores and a concise explanation with some overall improvement suggestions. Here are the aspects to consider:\n"
            "\n"
            "The aspects to consider are:\n"
            "1. Creative Score: Assess the creativity of the design. Does it stand out and capture attention through innovative elements?\n"
            "2. Attention: Evaluate the order of content consumption in the uploaded image. Start by identifying and analyzing the headline for its prominence and position. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text,or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"            
            "3. Distinction: Determine if the design differentiates itself from competitors. Is it unique and memorable?\n"
            "4. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "5. Clarity: Evaluate the clarity of the design elements. Are the visuals and text easy to understand?\n"
            "6. First Impressions: Analyze the initial impact of the design. Does it create a strong positive first impression?\n"
            "7. Cognitive Demand: Evaluate the cognitive load required to understand and navigate the design. Is it intuitive and easy to use?\n"
            "8. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting,length, use of numbers/lists, brand consistency, and power words.\n"
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
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])
        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Combined Marketing Analysis Results_V6:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None
    
    def detailed_marketing_analysis_v3(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image for marketing effectiveness. Provide a score from 1 to 5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format (Aspect, Score, Explanation, Improvement).Also after the table, please write the total sum of the scores and a concise explanation with some improvement overall suggestions. Here are the aspects to consider:\n"
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
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Detailed Marketing Analysis Results_V1:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None

    def marketing_effectiveness(uploaded_file):
        prompt = (
            "Imagine you are a marketing consultant reviewing an image for a client. Analyze the provided image and assess its marketing effectiveness based on the following criteria. For each criterion, provide a score from 1 to 5 (1 being poor, 5 being excellent) along with a concise explanation that highlights how the image content contributes to the score. After evaluating each criterion, present a bulleted summary with improvement suggestions for each aspect that scored lower than 4. Ensure that this analysis process yields consistent scoring results, regardless of how many times or when it is run.\n"
            "The criteria to consider are:\n"
            "1. Information Prioritization: Does the image effectively highlight the most important information? Consider factors like size, placement, and visual hierarchy.\n"
            "2. Visual Cues and Color Usage: Does the image use visual cues and colors to draw attention to key elements? Analyze how color choices, contrast, and elements like arrows or frames guide the viewer's attention.\n"
            "3. Labeling and Button Clarity: Are any labels or buttons present clearly labeled and easy to understand? Evaluate the use of text size, font choice, and placement for optimal readability.\n"
            "4. Attention Capturing: How well does the image capture and hold the viewer's attention? Consider elements like composition, negative space, and use of contrasting elements.\n"
            "5. Headline Clarity and Impact (if applicable): If a headline is present, how clear and impactful is it? Assess the headline's readability, relevance to the image content, and potential to evoke curiosity or action.\n"
            "6. Engagement Level: Evaluate the text quantity and how well the information is grouped. Is the text engaging? Analyze the text-to-image ratio, use of bullet points or call-to-action elements, and overall readability.\n"
            "7. Trustworthiness: Assess the trustworthiness of the content based on visual and textual elements. Consider factors like image quality, professionalism, and consistency with brand identity.\n"
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
            "Imagine you are a marketing consultant reviewing an image and its headline for a client. Analyze the provided image content alongside the headline text to assess the headline's effectiveness. Rate each criterion on a scale from 1 to 5 (1 being poor, 5 being excellent), and provide a concise explanation for each score based on the synergy between the image and headline. Present your results in a table format with columns labeled: Criterion, Score, Explanation. Below the table, calculate and display the total sum of all scores. Ensure that this analysis process yields consistent scoring results, regardless of how many times or when it is run. Conclude with three possible improved headlines that better align with the image content. The criteria to assess are:\n"
            "1. Clarity & Conciseness: How clearly does the headline convey the main point?\n"
            "2. Customer Focus: Does the headline emphasize a customer-centric approach?\n"
            "3. Relevance: How accurately does the headline reflect the content of the image?\n"
            "4. Keywords: Are relevant SEO keywords included naturally?\n"
            "5. Emotional Appeal: Does the headline evoke curiosity or an emotional response, considering the image content?\n"
            "6. Uniqueness: How original and creative is the headline?\n"
            "7. Urgency & Curiosity: Does the headline create a sense of urgency or pique curiosity, considering the image?\n"
            "8. Benefit-Driven: Does the headline convey a clear benefit or value proposition, aligned with the image content?\n"
            "9. Target Audience: Is the headline tailored to resonate with the specific target audience, considering the image's visual cues?\n"
            "10. Length & Format: Does the headline fall within an ideal length of 6-12 words?\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Headline Analysis Results:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None

    def headline_detailed_analysis(uploaded_file):
        prompt = (
            "Imagine you are analyzing a marketing image for a human client. You are an expert assessing the headline's effectiveness. Analyze the headline text extracted from the image and present the results in a table format with the following columns: Criteria, Assessment, and Explanation. After the table, provide an overall summary, including a concise explanation and some improvement suggestions. Ensure the analysis is consistent across multiple runs.\n"
            "The criteria to assess are:\n"
            "1. Word Count: Number of words in the headline.\n"
            "2. Character Count: Total number of characters, including spaces.\n"
            "3. Common Words: Count of frequently used words.\n"
            "4. Uncommon Words: Count of less frequent words.\n"
            "5. Emotional Words: Number of words conveying emotions (positive, negative, etc.).\n"
            "6. Power Words: Number of words used to grab attention or influence.\n"
            "7. Sentiment: Overall sentiment of the headline (positive, negative, neutral).\n"
            "8. Reading Grade Level: Reading grade level of the headline text.\n"
        )
        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("Headline Optimization Report Results:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None

    def flash_analysis(uploaded_file):
        prompt = (
            "Imagine you are a visual content analyst reviewing an image for a client. Analyze the provided image and generate a detailed description that captures the key elements and information relevant to marketing purposes. Focus on objective details like objects, people, colors, text, and their arrangement. Additionally, identify any potential cultural references or symbols that might be relevant to the target audience. Ensure the description is consistent across multiple runs, avoiding subjective interpretations or emotional responses."
        )

        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            st.error("Unexpected response structure from the model.")
            return None

    def ux_marketing_analysis(uploaded_file):
        prompt = (
            "Imagine you are a UX design and marketing analysis consultant reviewing a UX design image for a client. Analyze the provided image for its effectiveness in both UX design and marketing contexts. For each aspect listed below, provide a score from 1 to 5 (1 being poor, 5 being excellent) along with a concise explanation for all aspects and suggestions for improvement. Present the results in a table format with the columns: Aspect, Score, Explanation, and Improvement. Also, after the table, write the total sum of the scores and a concise explanation with some overall improvement suggestions. Ensure your scoring remains consistent for each aspect, regardless of how many times you analyze the image.\n"
            "The aspects to consider are:\n"
            "1. General Analysis: Evaluate the overall effectiveness of the UX design. How well does the design communicate its intended message?\n"
            "2. Creative Score: Assess the creativity of the design. Does it stand out and capture attention through innovative elements?\n"
            "3. Attract: Analyze the ability of the design to attract users. Are the visual elements appealing and engaging?\n"
            "4. Focus: Evaluate how well the design directs user attention to key areas. Does it guide users effectively to important information?\n"
            "5. Distinction: Determine if the design differentiates itself from competitors. Is it unique and memorable?\n"
            "6. Purpose and Value: Assess the clarity of the design’s purpose and value proposition. Is the message clear within a few seconds of viewing?\n"
            "7. Headline (if present):\n"
            "   - Clarity & Conciseness: Evaluate the headline's clarity and conciseness.\n"
            "   - Customer Focus: Does the headline emphasize a customer-centric approach?\n"
            "   - SEO keywords and emotional appeal: Does the headline incorporate SEO keywords and evoke an emotional response?\n"
            "8. Calls to Action (CTAs): Analyze the presence, prominence, benefits, and language of CTAs. Are they clear and compelling?\n"
            "9. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy?\n"
            "10. Engage: Assess the level of user engagement the design promotes. Does it encourage interaction and participation?\n"
            "11. Clarity: Evaluate the clarity of the design elements. Are the visuals and text easy to understand?\n"
            "12. First Impressions: Analyze the initial impact of the design. Does it create a strong positive first impression?\n"
            "13. Engagement: Assess the engagement level of the user experience. Is the UX design captivating and satisfying to interact with?\n"
            "14. Cognitive Demand: Evaluate the cognitive load required to understand and navigate the design. Is it intuitive and easy to use?\n"
            "15. Experience: Assess the overall user experience. How well does the design facilitate a smooth and enjoyable interaction?\n"
            "16. Trust: Analyze the trustworthiness of the design. Does it convey credibility and reliability?\n"
            "17. Act: Evaluate how well the design encourages users to take action. Are the calls to action effective and persuasive?\n"
            "18. Motivation: Assess the designs ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof?\n"
            "19. Influence: Analyze the influence of the design. Does it effectively persuade users and drive desired behaviors?\n"
            "20. Memorability: Evaluate how memorable the design is. Does it leave a lasting impression?\n"
        )

        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("UX and Marketing Analysis Results:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
        return None
    def custom_prompt_analysis(uploaded_file, custom_prompt, model):
        prompt = custom_prompt

        image = Image.open(uploaded_file)
        response = model.generate_content([prompt, image])

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            st.write("## User Prompt Analysis Results:")
            st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
        else:
            st.error("Unexpected response structure from the model.")
            return None
    # Streamlit app setup
    st.title('Marketing Image Analysis AI Assistant')

    with st.sidebar:
        st.header("Options")
        basic_analysis = st.button('Basic Analysis')
        detailed_analysis = st.button('Detailed Marketing Analysis')
        detailed_analysis_V1 = st.button('Detailed Marketing Analysis V1')
        detailed_analysis_V2 = st.button('Detailed Marketing Analysis V2')
        detailed_analysis_V3 = st.button('Detailed Marketing Analysis V3')
        combined_analysis_V5 = st.button('Combined Detailed Marketing Analysis V5')
        combined_analysis_V6 = st.button('Combined Detailed Marketing Analysis V6')
        marketing_success = st.button('Marketing Success Analysis')
        headline_analysis_button = st.button('Headline Analysis')
        detailed_headline_analysis_button = st.button('Headline Optimization Report') 
        flash_analysis_button = st.button('Flash Analysis') 
        ux_marketing_analysis_button = st.button('UX Design and Marketing Analysis')
        st.text_area("Enter your custom prompt for image analysis:", key="custom_prompt")
        custom_prompt_button = st.button('Custom Prompt Analysis')

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
                    st.markdown(detailed_result)
        if detailed_analysis_V1:
            with st.spinner("Performing detailed marketing analysis_V1..."):
                uploaded_file.seek(0)
                detailed_result_V1 = detailed_marketing_analysis_v1(uploaded_file)
                if detailed_result_V1:
                    st.write("## Detailed Marketing Analysis_V1 Results:")
                    st.markdown(detailed_result_V1)
        if detailed_analysis_V2:
            with st.spinner("Performing detailed marketing analysis_V2..."):
                uploaded_file.seek(0)
                detailed_result_V2 = detailed_marketing_analysis_v2(uploaded_file)
                if detailed_result_V2:
                    st.write("## Detailed Marketing Analysis_V2 Results:")
                    st.markdown(detailed_result_V2)
        if combined_analysis_V5:
            with st.spinner("Performing combined marketing analysis_V5..."):
                uploaded_file.seek(0)
                detailed_result_V5 = combined_marketing_analysis_V5(uploaded_file)
                if detailed_result_V5:
                    st.write("## Combined Marketing Analysis_V5 Results:")
                    st.markdown(detailed_result_V5)
        if combined_analysis_V6:
            with st.spinner("Performing combined marketing analysis_V6..."):
                uploaded_file.seek(0)
                detailed_result_V6 = combined_marketing_analysis_V6(uploaded_file)
                if detailed_result_V6:
                    st.write("## Combined Marketing Analysis_V6 Results:")
                    st.markdown(detailed_result_V6)
        if detailed_analysis_V3:
            with st.spinner("Performing detailed marketing analysis_V3..."):
                uploaded_file.seek(0)
                detailed_result_V3 = detailed_marketing_analysis_v3(uploaded_file)
                if detailed_result_V3:
                    st.write("## Combined Marketing Analysis_V3 Results:")
                    st.markdown(detailed_result_V3)                    
        if marketing_success:
            with st.spinner("Performing marketing effectiveness and success analysis..."):
                uploaded_file.seek(0)
                marketing_success_result = marketing_effectiveness(uploaded_file)
                if marketing_success_result:
                    st.write("## Marketing Effectiveness Analysis Results:")
                    st.markdown(marketing_success_result)

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

        if ux_marketing_analysis_button:
            with st.spinner("Performing UX and Marketing Analysis..."):
                uploaded_file.seek(0)
                ux_result = ux_marketing_analysis(uploaded_file)
                if ux_result:
                    st.write("## UX Design Analysis Results:")
                    st.markdown(ux_result)
                    
        if custom_prompt_button:
            with st.spinner("Performing custom prompt analysis..."):
                uploaded_file.seek(0)
                custom_prompt = st.session_state.custom_prompt
                custom_prompt_result = custom_prompt_analysis(uploaded_file, custom_prompt, model)
                if custom_prompt_result:
                    st.write("## Custom Prompt Analysis Results:")
                    st.markdown(custom_prompt_result)
