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
        # General prompt for both images and videos
        prompt = (
            "Analyze the media (image or video frame) for various marketing aspects, ensuring consistent results for each aspect. "
            "Respond in single words or short phrases separated by commas for each attribute: text amount (High or Low), "
            "color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), "
            "focus (Central message or Scattered), customer-centric (Yes or No), credibility (High or Low), "
            "user interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
        )
        try:
            if is_image:
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([prompt, image])  # Assuming model.generate_content handles image input
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
    
                frames = extract_frames(tmp_path)  # Assuming extract_frames extracts frames from video
                if frames is None or not frames:
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None
    
                response = model.generate_content([prompt, frames[0]])  # Analyzing the first frame
    
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
                st.error("Model did not provide a response.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
    def combined_marketing_analysis_V6(uploaded_file, is_image=True):
        prompt = """
        Imagine you are a UX design and marketing analysis consultant reviewing a visual asset (image or video) for a client.
        
        1. Asset Identification:
            - Clearly identify and describe the type of marketing asset (e.g., email, social media post, advertisement, flyer, brochure, landing page).
        
        2. Purpose Analysis:
            - Clearly state the specific purpose of the asset (e.g., selling a product, increasing brand awareness, driving website traffic).
            - Explain in detail how the asset aims to achieve its purpose, referencing specific elements or strategies used.
        
        3. Audience Identification:
            - Identify the target audience for the asset.
            - Describe the demographics, interests, needs, and pain points of this audience.
        
        4. Evaluation and Scoring:
            For each aspect listed below, provide:
                - A score from 1 to 5 (1 being low, 5 being high).
                - A concise explanation justifying the score.
                - Specific, actionable suggestions for improvement.
            Present the results in a table with columns: Aspect, Score, Explanation, and Improvement.
        
        Aspects to Consider:
        
            - Creative Impact: Does the design stand out and capture attention with innovative elements?
            - Attention & Hierarchy:
                - Image: Is the order of content consumption clear and effective? (e.g., headline, body text, visuals, CTA)
                - Video: Does the video's structure and editing guide the viewer's focus? Are key messages highlighted?
            - Distinction: 
                - Image: Do the visuals grab attention? Do they appeal to the viewer with and without text?
                - Video: Does the video use compelling visuals and storytelling to differentiate itself?
            - Purpose & Value: Is the asset's purpose and value proposition clear within 3 seconds? Is it customer-centric?
            - Clarity: Are the visuals and text easy to understand? Is the message conveyed effectively?
            - First Impression: Does the asset make a positive first impression? Is it visually appealing and inviting?
            - Cognitive Load: Is the asset easy to process and understand? Does it avoid overwhelming the viewer with too much information or complexity?
            - Headline Effectiveness:
                - Image: Evaluate the headline for clarity, conciseness, customer focus, emotional appeal, uniqueness, urgency, benefit-driven messaging, target audience relevance, and length/format.
                - Video: Evaluate the opening message/hook, and consider if on-screen text supports the video's narrative effectively.
            - Visual Cues & Color Usage:
                - Image: How effectively do visual cues and colors guide attention to key elements?
                - Video: How do color choices and transitions contribute to the overall mood and message?
            - Labeling & Button Clarity (if applicable): Are labels and buttons clear, easy to understand, and visually distinct?
            - Engagement: 
                - Image: Does the design encourage interaction or further exploration?
                - Video: Does the video hold the viewer's attention throughout? Are there elements that drive engagement?
            - Trustworthiness: Do the visual and textual elements create a sense of credibility, reliability, and intimacy? Is it brand or customer-centric?
            - Motivation: Does the asset appeal to user motivators? Does it use authority, social proof, or other persuasive techniques effectively?
            - Influence & Persuasion: Does the asset effectively persuade viewers and lead them towards a desired action?
            - Call to Action (CTA): If present, is the CTA prominent, clear, and compelling? Does it communicate the benefits of taking action?
            - Overall Experience (UX): How smooth and enjoyable is the user experience? Is it easy to navigate and understand the information presented?
            - Memorability: Will the asset leave a lasting impression on the viewer? Does it have elements that are unique or surprising?
            - Textual Effort: 
                - Image: Is the text clear, concise, and easy to read?
                - Video: Is the on-screen text minimal, readable, and well-timed? Does it complement the spoken message effectively?
        
        5. Overall Assessment:
            - Summarize the key findings from your analysis.
            - Calculate the total score across all aspects.
            - Provide concrete recommendations for improving the asset's overall marketing effectiveness, taking into account its specific type, purpose, and target audience.
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
                st.write("Combined Marketing Analysis Results_V6:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def text_analysis(uploaded_file, is_image=True):
        prompt = """
        Imagine you are a UX design and marketing analysis consultant reviewing the text content of a marketing asset (image or video, excluding the headline) for a client. Your goal is to provide a comprehensive analysis of the text's effectiveness and offer actionable recommendations for improvement.
        
        1. Text Extraction and Contextualization:
           - **Image Analysis:** Carefully extract and analyze all visible text within the image. Consider the text's placement, font choices, and relationship to visual elements.
           - **Video Analysis:**
              * Identify the most representative frame(s) that showcase the primary text content.
              * Extract and analyze the text from these frames.
              * Pay attention to any significant textual changes or patterns across different parts of the video.
              * Consider how the text interacts with the video's visuals and audio elements.
        
        2. Textual Assessment:
           Evaluate the extracted text based on the following criteria. For each aspect, provide:
              * A score from 1 to 5 (1 being low, 5 being high).
              * A clear and concise explanation justifying your score, focusing on both strengths and weaknesses.
              * Actionable suggestions for improvement, tailored to the specific text, the overall asset, and the target audience.
        
            Present your evaluation in a well-organized table with these columns: Aspect, Score, Explanation, and Improvement.
        
        Aspects to Evaluate:
        
           - Clarity and Conciseness: Is the text easy to understand? Does it avoid unnecessary jargon and get to the point quickly?
           - Customer Focus: Does the text center the customer's needs, desires, and pain points? Is it written from their perspective?
           - Engagement: Is the text compelling and attention-grabbing? Does it use storytelling, persuasive language, or humor to connect with the audience? Evaluate the text's length, readability, formatting (e.g., use of lists, bullet points), and overall value proposition.
           - Reading Effort: Is the text easy to scan and digest? Is it broken into manageable chunks? Does it avoid overly complex sentence structures?
           - Purpose and Value: Is the purpose of the text immediately clear? Does it effectively communicate the value proposition of the product or service?
           - Motivation & Persuasion: Does the text inspire the target audience to take action? Does it effectively appeal to their emotions, needs, or desires? Does it utilize social proof, authority, or other persuasive techniques?
           - Depth and Detail: Does the text provide enough information to satisfy different levels of audience interest? Are there opportunities to include additional details or links to further information?
           - Trustworthiness: Does the text build trust and credibility? Does it use language that is honest, transparent, and relatable? Is it more customer-centric or brand-centric?
           - Memorability: Are there any unique phrases, word choices, or storytelling elements that make the text stand out and memorable?
           - Emotional Appeal: Does the text evoke positive emotions relevant to the product or service? Are these emotions likely to resonate with the target audience?
           - Uniqueness & Differentiation: Does the text differentiate the brand or product from competitors? Does it have a distinct voice and style?
           - Urgency and Curiosity: Does the text create a sense of urgency or FOMO (fear of missing out)? Does it pique the audience's curiosity and encourage them to learn more?
           - Benefit Orientation: Does the text clearly articulate the benefits of the product or service to the customer? Are these benefits compelling and relevant to the target audience?
           - Target Audience Relevance: Is the text's language, tone, and style appropriate for the intended audience? Does it speak to their specific interests, needs, and preferences?
           - Cognitive Demand: How much mental effort is required to understand the text? Is the information presented in a way that is easy to digest and remember?
           - Reading Level: What is the estimated reading level of the text? Is it appropriate for the target audience's comprehension abilities?
        
        3. Conclusion and Recommendations:
        
           - Provide a concise summary of your analysis, highlighting the text's strengths, weaknesses, and overall effectiveness.
           - Calculate the total score based on the individual aspect scores.
           - Offer three alternative text versions that address the identified weaknesses and enhance the overall impact of the marketing message. These revisions should be tailored to the specific asset type, its purpose, and the target audience.
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
           - Clearly identify the main headline, Image Headline and any supporting headlines in the asset.
           - Differentiate the main headline from other text elements.
        
        3. Headline Evaluation (Main Headline Only):
           Evaluate the main headline against the following criteria, rating each on a scale from 0 to 10 (0 being poor, 10 being excellent), and provide a concise explanation for each score:
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
        prompt = """
        Imagine you are a visual content analyst reviewing a marketing asset (image or video) for a client. Your goal is to provide a detailed, objective description that captures essential information relevant to marketing decisions.
        
        Instructions:
        
        1. Asset Identification:
            - Clearly identify the type of asset (image or video).
        
        2. Detailed Description:
            - For images:
                - Describe the prominent visual elements (objects, people, animals, settings).
                - Note the dominant colors and their overall effect.
                - Mention any text, its content, font style, size, and placement.
                - Describe the composition and layout of the elements.
            - For videos:
                - Describe the key scenes, actions, and characters.
                - Note the visual style, color palette, and editing techniques.
                - Mention any text overlays, captions, or speech, transcribing if possible.
                - Identify the background music or sound effects, if present.
        
        3. Cultural References and Symbolism:
            - Identify any cultural references, symbols, or visual metaphors that could be significant to the target audience.
            - Explain how these elements might be interpreted or resonate with the audience.
        
        4. Marketing Implications:
            - Briefly summarize the potential marketing implications based on the visual and textual elements.
            - Consider how the asset might appeal to different demographics or interests.
            - Mention any potential positive or negative associations it may evoke.
        
        5. Additional Notes:
            - If analyzing a video, focus on the most representative frame(s) for the initial description.
            - Mention any significant changes or variations in visuals or text throughout the video.
        
        Please ensure your description is:
        
        - Objective: Focus on factual details and avoid subjective interpretations or opinions.
        - Detailed: Provide enough information for the client to understand the asset's visual and textual content.
        - Marketing-Oriented: Highlight elements that are relevant to marketing strategy and decision-making.
        - Consistent: Provide similar descriptions for the same asset, regardless of how many times you analyze it.
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
