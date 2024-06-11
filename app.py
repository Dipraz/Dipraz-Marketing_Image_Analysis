import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import io
import google.generativeai as genai
import cv2
import tempfile

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

    def extract_frames(video_file_path, num_frames=5):
        """Extracts frames from a video file using OpenCV."""
        cap = cv2.VideoCapture(video_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(total_frames // num_frames, 1) 
        frames = []
        for i in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Explicitly convert color space and create a PIL Image from bytes
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            pil_image = Image.open(io.BytesIO(buffer))
            
            frames.append(pil_image)

        cap.release()
        return frames
    def analyze_media(uploaded_file, is_image=True):
        prompt = (
            "Imagine you are a marketing consultant reviewing a visual asset (image or video) for a client. Analyze the provided asset for various marketing aspects and ensure your results remain consistent for each aspect, regardless of how many times you analyze the image. Respond in single words or short phrases separated by commas for each attribute: "
            "text amount (High or Low), color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), focus (Central message or Scattered), "
            "customer-centric (Yes or No), credibility (High or Low), user interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
        )
        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([prompt, frames[0]])  # Using the first frame

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
            st.error(f"Failed to read or process the media: {e}")
            return None

    def combined_marketing_analysis_V6(uploaded_file, is_image=True):
        prompt = (
            "Imagine you are a UX design and marketing analysis consultant reviewing a visual asset (image or video) for a client. Analyze the provided asset for marketing effectiveness. Firstly, provide detailed responses for the following:\n"
            "\n"
            "1. Asset Type: Clearly identify and describe the type of marketing asset. Examples include email, social media posts, advertisements, flyers, brochures, landing pages, etc.\n"
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
            "9. Visual Cues and Color Usage: Does the image use visual cues and colors to draw attention to key elements? Analyze how color choices, contrast, and elements like arrows or frames guide the viewer's attention.\n"
            "10. Labeling and Button Clarity: Are any labels or buttons present that are clearly labeled and easy to understand? Evaluate the use of text size, font choice, and placement for optimal readability.\n"
            "11. Engagement: Assess the engagement level of the user experience. Is the UX design captivating and satisfying to interact with?\n"
            "12. Trust: Assess the trustworthiness of the content based on visual and textual elements. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)? Assess the credibility, reliability, and intimacy conveyed by the content.\n"
            "13. Motivation: Assess the design's ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof?\n"
            "14. Influence: Analyze the influence of the design. Does it effectively persuade users and drive desired behaviors?\n"
            "15. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "16. Experience: Assess the overall user experience. How well does the design facilitate a smooth and enjoyable interaction?\n"
            "17. Memorability: Evaluate how memorable the design is. Does it leave a lasting impression?\n"
            "18. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? (1: Very Dense & Difficult, 5: Clear & Easy to Understand)\n"
        )

        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Combined Marketing Analysis Results_V6:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def text_analysis(uploaded_file, is_image=True):
        prompt = (
            "Imagine you are a UX design and marketing analysis consultant reviewing the text content of a marketing asset (image or video, excluding the headline) for a client. "
            "1. Text Extraction: If analyzing an image, extract all visible text. If analyzing a video, extract text from the most representative frame and note any significant textual changes in other frames.\n"
            "2. Text Analysis: Analyze the extracted text using the following criteria. For each aspect, provide a score from 1 to 5 (1 being low, 5 being high) along with a concise explanation and actionable suggestions for improvement. Present the results in a table format with the columns: Aspect, Score, Explanation, and Improvements.\n"
            "Aspects to Consider:\n"
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
            "17. Reading Age: Assess the reading grade level required to understand the text.\n"
            "Conclusion and Recommendations: Summarize the overall strengths and weaknesses of the text. Calculate the total score across all aspects. Provide three alternative text versions that address the identified weaknesses, align with the visual content, and resonate better with the target audience."
        )

        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Text Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def headline_analysis(uploaded_file, is_image=True):
        prompt = """
        Imagine you are a marketing consultant reviewing a visual asset (image or video) and its headline(s) for a client. Your goal is to provide a comprehensive analysis of the headline's effectiveness across various key criteria.
        1. Content Analysis:
           - If analyzing an image, examine the visual elements such as key objects, colors, composition, and style, along with the underlying messages or themes conveyed.
           - If analyzing a video, focus on the most representative frame and consider the overall visual and auditory elements that contribute to the message.
        
        2. Headline Identification:
           - Clearly identify the main headline and any supporting headlines in the asset.
           - Differentiate the main headline from other text elements.
        
        3. Headline Evaluation (Main Headline Only):
           Evaluate the main headline against the following criteria, rating each on a scale from 1 to 5 (1 being poor, 5 being excellent), and provide a concise explanation for each score:
           - Clarity: How easily and quickly does the headline convey the main point?
           - Customer Focus: Does the headline emphasize a customer-centric approach, addressing their needs or interests?
           - Relevance: How well does the headline reflect the visual content of the image or video?
           - Emotional Appeal: Does the headline evoke curiosity, excitement, or other emotions that resonate with the target audience?
           - Uniqueness: How original and memorable is the headline compared to typical marketing messages?
           - Urgency & Curiosity: Does the headline create a sense of urgency or pique curiosity to learn more?
           - Benefit-Driven: Does the headline clearly communicate a specific benefit or value proposition to the audience?
           - Target Audience: Is the headline's language, tone, and style tailored to the specific target audience?
           - Length & Format: Is the headline concise (ideally 6-12 words) and does it use formatting effectively?
        
        4. Present Results:
           - Display the main headline's evaluation in a table format with columns: Criterion, Score, Explanation, and Improvements. Ensure every cell in the table is filled, and if no improvements are needed, note that.
        
        5. Supporting Headline Evaluation (Optional):
           - If applicable, briefly assess any supporting headlines and note if they require further analysis. Consider creating a separate table if a more in-depth analysis is needed.
        
        6. Total Score:
           - Calculate and display the total score for the main headline based on the evaluations.
        
        7. Improved Headlines:
           - Provide three alternative headlines for the main headline that address any weaknesses identified. Ensure these headlines are free of colons, diverse in structure and style, and aligned with the visual content and the target audience.
        
        Note: If analyzing a video, mention any notable changes in headlines or messaging throughout the video.
        """

        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Headline Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def headline_detailed_analysis(uploaded_file, is_image=True):
        prompt = """
        Imagine you are a marketing consultant reviewing the headline text of a marketing asset (image or video) for a client. Your task is to assess the headline's effectiveness based on various linguistic and marketing criteria.
        
        1. Headline Extraction:
           - If analyzing an image, extract the main headline from the image.
           - If analyzing a video, extract the main headline from the most representative frame and note any significant textual changes in other frames.
        
        2. Headline Analysis:
           Analyze the extracted headline text and present the results in a table format with the following columns: Criteria, Assessment, and Explanation. Ensure the analysis is consistent across multiple runs.
        
        Criteria to Assess:
           - Word Count: Provide the total number of words in the headline.
           - Letter Count: Provide the total number of letters in the headline, excluding spaces and special characters. Count only alphabetic characters.
           - Common Words: Count the number of frequently used words in the headline.
           - Uncommon Words: Count the number of less frequently used words in the headline.
           - Emotional Words: Count the number of words that convey emotions (positive, negative, etc.).
           - Power Words: Count the number of words known to grab attention or influence.
           - Sentiment: Assess the overall sentiment of the headline (positive, negative, neutral).
           - Reading Grade Level: Estimate the reading grade level required to understand the headline text.
        
        3. Overall Summary and Suggestions:
           After the table, provide an overall assessment of the headline's effectiveness. Consider the following aspects:
           - Clarity and Conciseness: Is the headline easy to understand at a glance? Does it convey the key message effectively?
           - Impact and Engagement: Does the headline grab attention and make the reader curious? Does it evoke an emotional response?
           - Relevance to the Target Audience: Is the headline's language and style appropriate for the intended audience? Does it speak to their interests or needs?
           - Alignment with Visual Content: If applicable (for videos), does the headline align with the visual content and message of the video?
        
        Based on your analysis, provide three alternative headline suggestions that could improve the headline's overall effectiveness. Ensure the suggestions are clear, concise, attention-grabbing, relevant to the target audience, and aligned with the visual content (if applicable).
        """

        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Headline Optimization Report Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def flash_analysis(uploaded_file, is_image=True):
        prompt = (
            "Imagine you are a visual content analyst reviewing an image for a client. Analyze the provided image and generate a detailed description that captures the key elements and information relevant to marketing purposes. Focus on objective details like objects, people, colors, text, and their arrangement. Additionally, identify any potential cultural references or symbols that might be relevant to the target audience. Ensure the description is consistent across multiple runs, avoiding subjective interpretations or emotional responses."
        )

        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

            if response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                st.error("Unexpected response structure from the model.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def custom_prompt_analysis(uploaded_file, custom_prompt, is_image=True):
        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([custom_prompt, image])
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:  # Check if frames were extracted successfully
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                response = model.generate_content([custom_prompt, frames[0]])  # Using the first frame for analysis

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                return raw_response
            else:
                st.error("Unexpected response structure from the model.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    # Streamlit app setup
    st.title('Marketing Media Analysis AI Assistant')
    with st.sidebar:
        st.header("Options")
        basic_analysis = st.button('Basic Analysis')
        combined_analysis_V6 = st.button('Combined Detailed Marketing Analysis V6')
        text_analysis_button = st.button('Text Analysis')
        headline_analysis_button = st.button('Headline Analysis')
        detailed_headline_analysis_button = st.button('Headline Optimization Report')
        flash_analysis_button = st.button('Flash Analysis')

        st.header("Custom Prompt")
        custom_prompt = st.text_area("Enter your custom prompt here:")
        custom_prompt_button = st.button('Send')

    col1, col2 = st.columns(2)
    uploaded_files = col1.file_uploader("Upload your marketing media here:", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            is_image = uploaded_file.type in ['image/png', 'image/jpg', 'image/jpeg']
            if is_image:
                image = Image.open(uploaded_file)
                image = resize_image(image)
                col2.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                col2.video(uploaded_file, format="video/mp4")

            if basic_analysis:
                with st.spinner("Performing basic analysis..."):
                    uploaded_file.seek(0)
                    basic_analysis_result = analyze_media(uploaded_file, is_image)
                    if basic_analysis_result:
                        st.write("## Basic Analysis Results:")
                        st.json(basic_analysis_result)

            if combined_analysis_V6:
                with st.spinner("Performing combined marketing analysis_V6..."):
                    uploaded_file.seek(0)
                    detailed_result_V6 = combined_marketing_analysis_V6(uploaded_file, is_image)
                    if detailed_result_V6:
                        st.write("## Combined Marketing Analysis_V6 Results:")
                        st.markdown(detailed_result_V6)

            if text_analysis_button:
                with st.spinner("Performing text analysis..."):
                    uploaded_file.seek(0)
                    text_result = text_analysis(uploaded_file, is_image)
                    if text_result:
                        st.write("## Text Analysis Results:")
                        st.markdown(text_result)

            if headline_analysis_button:
                with st.spinner("Performing headline analysis..."):
                    uploaded_file.seek(0)
                    headline_result = headline_analysis(uploaded_file, is_image)
                    if headline_result:
                        st.write("## Headline Analysis Results:")
                        st.markdown(headline_result)

            if detailed_headline_analysis_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    uploaded_file.seek(0)
                    detailed_headline_result = headline_detailed_analysis(uploaded_file, is_image)
                    if detailed_headline_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(detailed_headline_result)

            if flash_analysis_button:
                with st.spinner("Performing Flash analysis..."):
                    uploaded_file.seek(0)
                    flash_result = flash_analysis(uploaded_file, is_image)
                    if flash_result:
                        st.write("## Flash Analysis Results:")
                        st.markdown(flash_result)

            if custom_prompt_button:
                with st.spinner("Performing custom prompt analysis..."):
                    uploaded_file.seek(0)
                    custom_result = custom_prompt_analysis(uploaded_file, custom_prompt, is_image)
                    if custom_result:
                        st.write("## Custom Prompt Analysis Results:")
                        st.markdown(custom_result)
