import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import io
import google.generativeai as genai
import cv2
import tempfile
import re
import imageio
import json
import xml.etree.ElementTree as ET
import base64

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

    def resize_image(image, max_size=(300, 250)):
        image.thumbnail(max_size)
        return image

    def extract_frames(video_file_path, num_frames=5):
        """Extracts frames from a video file using OpenCV."""
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            st.error(f"Failed to open video file {video_file_path}. Check if the file is corrupt or format is unsupported.")
            return None
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(total_frames // num_frames, 1)
        frames = []
    
        for i in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
    
            # Convert color space and create a PIL Image from bytes
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
        cap.release()
        if len(frames) == 0:
            st.error("No frames were extracted, possibly due to an error in reading the video.")
            return None
        return frames
    
    def analyze_video(uploaded_file):
        """Analyzes video by extracting frames and performing model inference on the first frame."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
    
            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                st.error("No frames were extracted from the video. Please check the video format.")
                return None
    
            prompt = (
                "Analyze the media (image or video frame) for various marketing aspects, ensuring consistent results for each aspect. "
                "Respond in single words or short phrases separated by commas for each attribute: text amount (High or Low), "
                "Color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), "
                "Focus (Central message or Scattered), customer-centric (Yes or No), credibility (High or Low), "
                "User interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
            )
    
            response = model.generate_content([prompt, frames[0]])  # Analyzing the first frame
            if response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                st.error("Model did not provide a response.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    # Initialize session state variables for headlines and analysis results
    if 'headlines' not in st.session_state:
        st.session_state.headlines = {}
    if 'headline_result' not in st.session_state:
        st.session_state.headline_result = None    

    def analyze_media(uploaded_file, is_image=True):
        # General prompt for both images and videos
        prompt = (
            "Analyze the media (image or video frame) for various marketing aspects, ensuring consistent results for each aspect. "
            "Respond in single words or short phrases separated by commas for each attribute: text amount (High or Low), "
            "Color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), "
            "Focus (Central message or Scattered), customer-centric (Yes or No), credibility (High or Low), "
            "User interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
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

**Part 1: Text Extraction and Contextualization**

*   **Image Analysis:** 
    1.  **Text Extraction:** Thoroughly identify and extract ALL visible text within the image, including headlines, body copy, captions, calls to action, taglines, logos, and any other textual elements.
    2.  **Presentation:** Present the extracted text in a clear, bulleted list format, maintaining the original order and structure to the extent possible.
    3.  **Visual Analysis:**
        *   **Placement:**  For each text element, specify its location within the image (e.g., top left, centered, bottom right). Note any instances of overlapping text or elements that might hinder readability.
        *   **Font Choices:** Describe the font style (serif, sans-serif, script, etc.), weight (bold, regular, light), size, and color of each distinct text element.
        *   **Visual Relationships:** Explain how the text interacts with other visual elements (images, graphics, colors) and how it contributes to the overall message and hierarchy of information.

*   **Video Analysis:**
    1.  **Key Frame Identification:** Select the most representative frame(s) that showcase the primary text content.
    2.  **Text Extraction:** Extract and present the text from these key frames in a clear, bulleted list format.
    3.  **Temporal Analysis:** Briefly describe any significant textual changes or patterns that occur throughout the video. 
    4.  **Integration with Visuals and Audio:** Analyze how the text interacts with the video's visuals (scenes, characters, actions) and audio (dialogue, music, sound effects).


**Part 2: Textual Assessment**
Evaluate the extracted text based on the following criteria. For each aspect, provide:
    *   **Score:** A rating from 1 (poor) to 5 (excellent)
    *   **Explanation:** A concise justification of the score, highlighting strengths and weaknesses.
    *   **Improvement:** Specific, actionable suggestions for enhancing the text's effectiveness.

| Aspect                     | Score (1-5) | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                      | Improvement                                                                                                                                                                                                                                                                                                                                                                                                          |
| :-------------------------- | ----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clarity and Conciseness     |             | Assess how easy it is to understand the text. Does it convey the message directly and without unnecessary jargon? Is it free of ambiguity or confusing language? Consider sentence structure, vocabulary, and overall flow.                                                                                                                                                                                          | Suggest ways to simplify or clarify the language, eliminate jargon, or shorten sentences.  Provide examples of how the text could be made more concise and direct.                                                                                                                                                                                                                                |
| Customer Focus             |             | Evaluate if the text is written from the customer's perspective. Does it address their needs, desires, and pain points? Does it use language that resonates with them? Are the benefits of the product/service clearly articulated in a way that matters to the target audience?                                                                                                                                                      | Offer suggestions for incorporating the customer's voice or perspective into the text. Recommend ways to highlight benefits that are most relevant to the target audience.                                                                                                                                                                                                                              |
| Engagement                 |             | Assess how compelling and interesting the text is. Does it grab attention and hold the reader's interest? Consider the use of storytelling techniques, persuasive language, humor, rhetorical questions, or other engaging elements. Evaluate the length of the text, readability, formatting (e.g., use of bullet points, lists, headings), and the overall value proposition presented. | Propose ways to make the text more captivating. This could involve incorporating storytelling elements, using stronger verbs, adding humor, or highlighting the unique value proposition in a more compelling way. Suggest ways to improve formatting for better readability and scannability.                                                                                                   |
| Reading Effort             |             | Evaluate how easy it is to read and understand the text. Does it flow smoothly? Are the sentences well-structured? Is the vocabulary appropriate for the target audience?  Consider the use of technical terms, passive voice, or complex sentence structures.                                                                                                                                                                    | Suggest ways to improve readability by using simpler sentence structures, active voice, or more accessible vocabulary.                                                                                                                                                                                                                                                                              |
| Purpose and Value          |             | Is the purpose of the text immediately clear to the reader? Does it effectively communicate the value proposition of the product or service? Does it answer the question "What's in it for me?" from the customer's point of view?                                                                                                                                                                                          | If the purpose or value is unclear, recommend ways to make it more explicit. This could involve restating the key message more directly or highlighting the most compelling benefit.                                                                                                                                                                                                        |
| Motivation & Persuasion    |             | Assess the text's persuasive power. Does it create a sense of urgency or desire for the product/service? Does it use strong calls to action? Does it effectively leverage social proof, authority, or other persuasive techniques?                                                                                                                                                                                            | Suggest ways to strengthen the persuasive elements. This could involve adding a stronger call to action, incorporating testimonials or statistics, or using more persuasive language.                                                                                                                                                                                                                  |
| Depth and Detail           |             | Determine if the text provides sufficient information to meet the needs of the target audience. Is there enough detail to satisfy those who want to learn more? Are there opportunities to provide additional context, links to further resources, or options for deeper engagement?                                                                                                                                          | If the text lacks depth, suggest areas where more detail could be added. If it's too detailed, recommend condensing or prioritizing information. Consider whether additional resources (like a link to a landing page) would enhance the message.                                                                                                                                  |
| Trustworthiness             |             | Evaluate how trustworthy and credible the text appears. Does it use language that is honest, transparent, and relatable? Does it focus on building trust with the customer or primarily on promoting the brand? Are there opportunities to incorporate trust signals like testimonials, guarantees, or data?                                                                                                                 | Suggest ways to enhance the text's trustworthiness. This could involve softening the sales pitch, highlighting customer testimonials, or using language that emphasizes transparency and authenticity.                                                                                                                                                                                            |
| Memorability                |             | Assess if there are any elements in the text that make it stand out and easy to remember. This could include catchy phrases, unique word choices, or interesting storytelling techniques.                                                                                                                                                                                                                               | If the text lacks memorability, recommend incorporating memorable phrases, wordplay, or anecdotes that tie into the product/service.                                                                                                                                                                                                                                                                 |
| Emotional Appeal           |             | Determine if the text evokes emotions that align with the desired brand image and message. Are these emotions likely to resonate with the target audience? Are there missed opportunities to create a stronger emotional connection?                                                                                                                                                                                               | Suggest incorporating language that evokes specific emotions (joy, excitement, etc.). Consider the use of sensory words or vivid imagery to make the text more emotionally impactful.                                                                                                                                                                                                            |
| Uniqueness & Differentiation |             | Analyze whether the text effectively differentiates the brand or product from competitors. Does it have a distinct voice and style? Is the message unique and memorable, or does it blend in with other marketing messages?                                                                                                                                                                                                       | If the text lacks differentiation, suggest ways to make it more unique. This could involve developing a stronger brand voice, using unexpected language or metaphors, or focusing on a specific aspect that sets the brand or product apart.                                                                                                                                                   |
| Urgency and Curiosity       |             | Assess if the text creates a sense of urgency or FOMO (fear of missing out). Does it pique the audience's curiosity and entice them to learn more or take action? Are there any missed opportunities to create a stronger sense of urgency or intrigue?                                                                                                                                                               | Suggest ways to create urgency (e.g., limited-time offers) or spark curiosity (e.g., asking intriguing questions).                                                                                                                                                                                                                                                                                |
| Benefit Orientation         |             | Evaluate whether the text clearly articulates the benefits of the product or service to the customer. Are these benefits specific, relevant, and compelling to the target audience? Does the text answer the question "What's in it for me?" from the customer's perspective?                                                                                                                                              | If the benefits are unclear or not well-articulated, recommend making them more explicit and customer-centric. Focus on outcomes or solutions that the product/service provides, rather than just features.                                                                                                                                                                                 |
| Target Audience Relevance  |             | Determine if the text's language, tone, and style are appropriate and appealing to the intended audience. Does it speak to their specific interests, needs, and preferences? Is it culturally relevant and sensitive to their values?                                                                                                                                                                                               | If the text
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
        prompt = f"""
    Imagine you are a marketing consultant reviewing the headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. 
    Your task is to assess the headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Headline Extraction and Context**
  **Image/Video:**
    1. **Headline Identification:**
        * **Main Headline:** Clearly state the main headline extracted from the image or video.
        * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
        * **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.

**Part 2: Headline Analysis**
Analyze the extracted headline(s) and present the results in a well-formatted table:

| Criterion             | Main Headline Assessment | Image Headline Assessment (if applicable) | Supporting Headline Assessment (if applicable) | Main Headline Explanation | Image Headline Explanation (if applicable) | Supporting Headline Explanation (if applicable) | Main Headline Improvement | Image Headline Improvement (if applicable) | Supporting Headline Improvement (if applicable) |
|-----------------------|--------------------------|------------------------------------------|-----------------------------------------------|-----------------------------|---------------------------|--------------------------------------------|------------------------------------------------|
| Clarity               | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Clarity]_ |  _[Explanation for Image Headline Clarity]_ | _[Explanation for Supporting Headline Clarity]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Customer Focus        | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Customer Focus]_ |  _[Explanation for Image Headline Customer Focus]_ | _[Explanation for Supporting Headline Customer Focus]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Relevance             | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Relevance]_ |  _[Explanation for Image Headline Relevance]_ | _[Explanation for Supporting Headline Relevance]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Emotional Appeal      | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Emotional Appeal]_ |  _[Explanation for Image Headline Emotional Appeal]_ | _[Explanation for Supporting Headline Emotional Appeal]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Uniqueness            | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Uniqueness]_ |  _[Explanation for Image Headline Uniqueness]_ | _[Explanation for Supporting Headline Uniqueness]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Urgency & Curiosity   | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Urgency & Curiosity]_ |  _[Explanation for Image Headline Urgency & Curiosity]_ | _[Explanation for Supporting Headline Urgency & Curiosity]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Benefit-Driven        | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Benefit-Driven]_ |  _[Explanation for Image Headline Benefit-Driven]_ | _[Explanation for Supporting Headline Benefit-Driven]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Target Audience       | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Target Audience]_ |  _[Explanation for Image Headline Target Audience]_ | _[Explanation for Supporting Headline Target Audience]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Length & Format       | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Length & Format]_ |  _[Explanation for Image Headline Length & Format]_ | _[Explanation for Supporting Headline Length & Format]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
| Overall Effectiveness | _[Rate from 1 to 5]_     | _[Rate from 1 to 5]_                      | _[Rate from 1 to 5]_                           | _[Explanation for Main Headline Overall Effectiveness]_ |  _[Explanation for Image Headline Overall Effectiveness]_ | _[Explanation for Supporting Headline Overall Effectiveness]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |  _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ | _[Suggest improvement if possible, or explain why it's already effective if the score is 5]_ |
**Part 3: Improved Headline Suggestions**
Provide three alternative headlines for EACH of the following, along with a brief explanation for each option:

* **Main Headline:**
  * **Option 1:** [Headline] - [Explanation]
  * **Option 2:** [Headline] - [Explanation]
  * **Option 3:** [Headline] - [Explanation]
* **Image Headline:**
  * **Option 1:** [Headline] - [Explanation]
  * **Option 2:** [Headline] - [Explanation]
  * **Option 3:** [Headline] - [Explanation]
* **Supporting Headline (if applicable):**
  * **Option 1:** [Headline] - [Explanation]
  * **Option 2:** [Headline] - [Explanation]
  * **Option 3:** [Headline] - [Explanation]
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
                st.session_state.headline_result = raw_response
                st.write("Headline Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)

                headline_matches = re.findall(r'(Main Headline|Image Headline):\s*(.*)', raw_response)
                extracted_headlines = {headline_type: headline_text for headline_type, headline_text in headline_matches}

                if "Image Headline" in extracted_headlines:
                    st.session_state.headlines = extracted_headlines
                else:
                    st.warning("Image headline not found in the results. Further analysis cannot be performed.")
                    st.session_state.headlines = None

            else:
                st.error("Unexpected response structure from the model.")
                return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def headline_detailed_analysis(uploaded_file, is_image=True):
        prompt = """
        Imagine you are a marketing consultant reviewing the headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. 
        Your task is to assess the headline's effectiveness based on various linguistic and marketing criteria.
        
        **Part 1: Headline Extraction and Context**
          **Image/Video:**
            1. **Headline Identification:**
                * **Main Headline:** Clearly state the main headline extracted from the image or video.
                * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
                * **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.
                
        **Part 2: Headline Analysis**
        Analyze the extracted headline(s) and present the results in a well-formatted table:
        
        | Criterion             | Main Headline Assessment                                    | Image Headline Assessment (if applicable)                                  | Supporting Headline Assessment (if applicable)                                  | Explanation | Improvement |
        |-----------------------|-------------------------------------------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------|-------------|
        | Word Count            | _[Number of words in the headline]_                         | _[Number of words in the headline]_                                        | _[Number of words in the headline]_                                             | _Example: The headline has X words, which is [concise/lengthy/appropriate] for a [type of asset]._ | _Example: To improve, consider adjusting the word count to better suit the [type of asset]._ |
        | Keyword Relevance     | _[Assessment of how well the headline incorporates relevant keywords or phrases]_ | _[Assessment of how well the headline incorporates relevant keywords or phrases]_ | _[Assessment of how well the headline incorporates relevant keywords or phrases]_ | _Example: The headline includes [X number] of relevant keywords, such as "[keyword 1]", "[keyword 2]", etc. This may help with [SEO/search engine visibility/targeting a specific audience]._ | _Example: To improve, consider incorporating additional relevant keywords such as "[keyword 3]". |
        | Common Words          | _[Number of common words]_                                  | _[Number of common words]_                                                 | _[Number of common words]_                                                      | _Example: The headline uses X common words, which may [enhance readability/make it less memorable/be appropriate for the target audience]._ | _Example: To improve, consider using fewer common words to make the headline more unique._ |
        | Uncommon Words        | _[Number of uncommon words]_                                | _[Number of uncommon words]_                                               | _[Number of uncommon words]_                                                    | _Example: The headline includes X uncommon words, which can [make it more memorable/potentially confuse some readers/be appropriate for a niche audience]._ | _Example: To improve, consider balancing uncommon words with common ones to enhance readability._ |
        | Emotional Words       | _[Number of words conveying emotion (positive, negative, neutral)]_ | _[Number of words conveying emotion (positive, negative, neutral)]_        | _[Number of words conveying emotion (positive, negative, neutral)]_             | _Example: The headline contains X emotional words, indicating a [positive/negative/neutral] tone. This may [elicit a strong response/be appropriate for the context/be overly dramatic]._ | _Example: To improve, consider using more emotional words to elicit a stronger response._ |
        | Power Words           | _[Number of words with strong persuasive potential]_        | _[Number of words with strong persuasive potential]_                       | _[Number of words with strong persuasive potential]_                            | _Example: The headline includes X power words, which are designed to [grab attention/persuade/create urgency]. However, overuse can make the headline seem [pushy/manipulative/less authentic]._ | _Example: To improve, consider using power words sparingly to avoid sounding pushy._ |
        | Sentiment             | _[Overall sentiment: positive, negative, or neutral]_       | _[Overall sentiment: positive, negative, or neutral]_                      | _[Overall sentiment: positive, negative, or neutral]_                           | _Example: The headline has a [positive/negative/neutral] sentiment, which [aligns well with the message/could be adjusted for better impact]._ | _Example: To improve, consider aligning the sentiment more closely with the overall message._ |
        | Reading Grade Level   | _[Estimated grade level required to understand the headline]_ | _[Estimated grade level required to understand the headline]_              | _[Estimated grade level required to understand the headline]_                   | _Example: The headline is written at a [grade level] reading level, making it [easy/challenging] for the target audience to comprehend. This is [appropriate/inappropriate] considering their [assumed education level/interests/background]._ | _Example: To improve, consider adjusting the reading grade level to better match the target audience._ |
        | Clarity & Conciseness | _[Assessment of how clear and easy to understand the headline is]_ | _[Assessment of how clear and easy to understand the headline is]_         | _[Assessment of how clear and easy to understand the headline is]_              | _Example: The headline is [clear and concise/somewhat vague/difficult to understand]. It [effectively/ineffectively] communicates the main message. The use of [specific words/phrases] contributes to its [clarity/lack of clarity]._ | _Example: To improve, consider simplifying the language for better clarity._ |
        | Impact & Engagement   | _[Assessment of how well the headline grabs attention and evokes emotion]_ | _[Assessment of how well the headline grabs attention and evokes emotion]_  | _[Assessment of how well the headline grabs attention and evokes emotion]_       | _Example: The headline is [attention-grabbing/forgettable/mildly interesting]. It [does/does not] effectively create a sense of urgency or curiosity. The use of [emotional language/power words/humor] [successfully/unsuccessfully] evokes a response in the reader._ | _Example: To improve, consider using more engaging language to evoke a stronger response._ |
        | Relevance             | _[Assessment of how well the headline aligns with the visual content and target audience]_ | _[Assessment of how well the headline aligns with the visual content and target audience]_ | _[Assessment of how well the headline aligns with the visual content and target audience]_ | _Example: The headline is [highly relevant/somewhat relevant/irrelevant] to the visual content and the target audience. It [successfully/partially/unsuccessfully] connects with their [interests/needs/pain points] by using [language/tone/style] that is [appropriate/inappropriate] for them._ | _Example: To improve, consider adjusting the language to better align with the target audience's interests._ |
        | Overall Effectiveness | _[Overall rating of the headline's effectiveness based on all criteria, 1 (poor) to 5 (excellent)]_ | _[Overall rating of the headline's effectiveness based on all criteria, 1 (poor) to 5 (excellent)]_ | _[Overall rating of the headline's effectiveness based on all criteria, 1 (poor) to 5 (excellent)]_ | _Example: Considering all factors, the headline receives an overall score of X out of 5. It is [highly effective/moderately effective/ineffective] due to its [strengths/weaknesses] in [specific criteria]._ | _Example: To improve, consider addressing the identified weaknesses in specific criteria._ |
        
        **Part 3: Improved Headline Suggestions**
        Provide three alternative headlines that improve upon the original while maintaining relevance to the visual content and the target audience:
        * **Option 1:**
        * **Option 2:**
        * **Option 3:**        
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
    def main_headline_detailed_analysis(uploaded_file, is_image=True):
        prompt =  f"""
Imagine you are a marketing consultant reviewing the main headline text of a marketing asset ({'image' if is_image else 'video'}) for a client.
Your task is to assess the main headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Headline Extraction and Context**
**Image/Video:**
1. **Headline Identification:**
  * **Main Headline:** Clearly state the main headline extracted from the image or video.
  * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
  * **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.

**Part 2: Headline Analysis**
Analyze the extracted Main Headline and present the results in a well-formatted table:

Headline being analyzed: [Main Headline]

| Criterion               | Score | Explanation                                       | Main Headline Improvement               |
|-------------------------|-------|---------------------------------------------------|-----------------------------------------|
| Clarity                 | _[1-5]_ | _[Explanation for clarity of the main headline]_   | _[Suggested improvement or reason it's effective]_ |
| Customer Focus          | _[1-5]_ | _[Explanation for customer focus of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Relevance               | _[1-5]_ | _[Explanation for relevance of the main headline]_  | _[Suggested improvement or reason it's effective]_ |
| Emotional Appeal        | _[1-5]_ | _[Explanation for emotional appeal of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Uniqueness              | _[1-5]_ | _[Explanation for uniqueness of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Urgency & Curiosity     | _[1-5]_ | _[Explanation for urgency & curiosity of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Benefit-Driven          | _[1-5]_ | _[Explanation for benefit-driven nature of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Target Audience         | _[1-5]_ | _[Explanation for target audience focus of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Length & Format         | _[1-5]_ | _[Explanation for length & format of the main headline]_ | _[Suggested improvement or reason it's effective]_ |
| Overall Effectiveness   | _[1-5]_ | _[Explanation for overall effectiveness of the main headline]_ | _[Suggested improvement or reason it's effective]_ |

Total Score: _[Sum of all scores]_

**Part 3: Improved Headline Suggestions**
Provide three alternative headlines for the main headline, along with a brief explanation for each option:

* **Option 1:** [Headline] - [Explanation]
* **Option 2:** [Headline] - [Explanation]
* **Option 3:** [Headline] - [Explanation]
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
    def image_headline_detailed_analysis(uploaded_file, is_image=True):
        prompt = f"""
Imagine you are a marketing consultant reviewing the image headline text of a marketing asset ({'image' if is_image else 'video'}) for a client.
Your task is to assess the image headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Headline Extraction and Context**
**Image/Video:**
1. **Headline Identification:**
  * **Main Headline:** Clearly state the main headline extracted from the image or video.
  * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
  * **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.

**Part 2: Headline Analysis**
Analyze the extracted Image Headline and present the results in a well-formatted table:

Headline being analyzed: [Image Headline]

| Criterion               | Score | Explanation                                       | Image Headline Improvement              |
|-------------------------|-------|---------------------------------------------------|-----------------------------------------|
| Clarity                 | _[1-5]_ | _[Explanation for clarity of the image headline]_   | _[Suggested improvement or reason it's effective]_ |
| Customer Focus          | _[1-5]_ | _[Explanation for customer focus of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Relevance               | _[1-5]_ | _[Explanation for relevance of the image headline]_  | _[Suggested improvement or reason it's effective]_ |
| Emotional Appeal        | _[1-5]_ | _[Explanation for emotional appeal of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Uniqueness              | _[1-5]_ | _[Explanation for uniqueness of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Urgency & Curiosity     | _[1-5]_ | _[Explanation for urgency & curiosity of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Benefit-Driven          | _[1-5]_ | _[Explanation for benefit-driven nature of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Target Audience         | _[1-5]_ | _[Explanation for target audience focus of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Length & Format         | _[1-5]_ | _[Explanation for length & format of the image headline]_ | _[Suggested improvement or reason it's effective]_ |
| Overall Effectiveness   | _[1-5]_ | _[Explanation for overall effectiveness of the image headline]_ | _[Suggested improvement or reason it's effective]_ |

Total Score: _[Sum of all scores]_

**Part 3: Improved Headline Suggestions**
Provide three alternative headlines for the image headline, along with a brief explanation for each option:

* **Option 1:** [Headline] - [Explanation]
* **Option 2:** [Headline] - [Explanation]
* **Option 3:** [Headline] - [Explanation]
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
    def supporting_headline_detailed_analysis(uploaded_file, is_image=True):
        prompt = f"""
Imagine you are a marketing consultant reviewing the supporting headline text of a marketing asset ({'image' if is_image else 'video'}) for a client.
Your task is to assess the supporting headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Headline Extraction and Context**
**Image/Video:**
1. **Headline Identification:**
  * **Main Headline:** Clearly state the main headline extracted from the image or video.
  * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
  * **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.

**Part 2: Headline Analysis**
Analyze the extracted Supporting Headline and present the results in a well-formatted table:

Headline being analyzed: [Supporting Headline]

| Criterion               | Score | Explanation                                       | Supporting Headline Improvement         |
|-------------------------|-------|---------------------------------------------------|-----------------------------------------|
| Clarity                 | _[1-5]_ | _[Explanation for clarity of the supporting headline]_   | _[Suggested improvement or reason it's effective]_ |
| Customer Focus          | _[1-5]_ | _[Explanation for customer focus of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Relevance               | _[1-5]_ | _[Explanation for relevance of the supporting headline]_  | _[Suggested improvement or reason it's effective]_ |
| Emotional Appeal        | _[1-5]_ | _[Explanation for emotional appeal of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Uniqueness              | _[1-5]_ | _[Explanation for uniqueness of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Urgency & Curiosity     | _[1-5]_ | _[Explanation for urgency & curiosity of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Benefit-Driven          | _[1-5]_ | _[Explanation for benefit-driven nature of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Target Audience         | _[1-5]_ | _[Explanation for target audience focus of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Length & Format         | _[1-5]_ | _[Explanation for length & format of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |
| Overall Effectiveness   | _[1-5]_ | _[Explanation for overall effectiveness of the supporting headline]_ | _[Suggested improvement or reason it's effective]_ |

Total Score: _[Sum of all scores]_

**Part 3: Improved Headline Suggestions**
Provide three alternative headlines for the supporting headline, along with a brief explanation for each option:

* **Option 1:** [Headline] - [Explanation]
* **Option 2:** [Headline] - [Explanation]
* **Option 3:** [Headline] - [Explanation]
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
        
    def meta_profile(uploaded_file, is_image=True):
        prompt = f"""
Analyze the uploaded image to develop detailed personas for Facebook advertising, focusing on the specific targeting options available on the platform. The analysis should reveal potential persona types that are likely to engage with the advertisement based on discernible visual elements and implied context within the image. The personas should be tailored to leverage Facebook's detailed targeting capabilities, including location, age, gender, interests, and more.

**Persona Development Instructions:**

1. **Persona Types Identification:**
   - Identify 4 distinct persona types that are most likely to respond to the advertisement, based on the image analysis.
   - Present these in a table format with columns for 'Persona Type' and 'Description'.

2. **Detailed Persona Profiles:**
   - For each identified persona type, create a comprehensive profile. Each profile should be presented in a table with the following columns:
      - **Persona Type:** The general category of the persona.
      - **Description:** A brief overview based on the image analysis.
      - **Analysis:** Detailed characteristics and expected reactions to the ad, tailored to Facebook’s targeting options like age, location, gender, interests, behaviors, device usage, and more.

**Important Considerations:**

* **Base on Visual Evidence:** Ensure that all persona deductions are supported by visible elements in the image.
* **Utilize Facebook’s Targeting Options:** Include specifics such as:
   - **Location:** Suggestions on countries, cities, or areas.
   - **Age and Gender:** Appropriate age range and gender, if discernible.
   - **Languages:** Potential languages spoken by the persona.
   - **Interests:** Deductions based on visible activities or preferences.
   - **Behaviors and Purchase Behavior:** Inferred from any visible consumer goods, technology, or lifestyle cues.
   - **Device Usage:** Likely devices used based on the context or settings in the image.
   - **Connections to Pages, Apps, or Events:** If there are implications of prior interactions with similar content.
   - **Life Events, Education Level, Fields of Study, Job Titles, and Industries:** Speculate based on attire, setting, or activities.

**Example Persona Tables Based on an Image of People at a Tech Conference:**

**Persona Types Table:**

| Persona Type     | Description                            |
|------------------|----------------------------------------|
| Tech Enthusiast  | Engaged with the latest technology and gadgets. |
| Startup Founder  | Interested in entrepreneurship and startup culture. |
| Software Developer | Likely works in software development, interested in coding and new tech. |
| Business Professional | Engaged in corporate roles, interested in networking and professional development. |

**Detailed Persona Profiles Table:**

| Persona Type           | Description               | Analysis                                                    |
|------------------------|---------------------------|-------------------------------------------------------------|
| Tech Enthusiast        | Early adopters of technology, often attends tech events | Target with ads for new gadgets, tech conferences, online tech tutorials |
| Startup Founder        | Young entrepreneurs, engaged in startup ecosystems | Focus on ads for business tools, networking events, investor connections |
| Software Developer     | Professionals in software development, interested in coding and software updates | Advertise coding bootcamps, new software tools, tech job openings |
| Business Professional  | Works in a corporate setting, interested in career advancement | Target with professional development courses, seminars, and corporate solutions |

This structured approach should guide the generation of actionable persona insights that can be directly applied in Facebook advertising strategies.
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
                st.write("Meta (Facebook) targeting Profile Result::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
        
    def linkedin_profile(uploaded_file, is_image=True):
        prompt = f"""
Analyze the uploaded image to develop detailed personas for LinkedIn advertising, focusing on the specific targeting options available on the platform. The analysis should reveal potential persona types that are most likely to engage with the advertisement based on discernible visual elements and implied context within the image. The personas should be tailored to leverage LinkedIn's detailed targeting capabilities, including company industry, job functions, and more.

**Persona Development Instructions:**

1. **Persona Types Identification:**
   - Identify 4 distinct persona types that are most likely to respond to the advertisement, based on the image analysis.
   - Present these in a table format with columns for 'Persona Type' and 'Description'.

2. **Detailed Persona Profiles:**
   - For each identified persona type, create a comprehensive profile. Each profile should be presented in a table with the following columns:
      - **Persona Type:** The general category of the persona.
      - **Description:** A brief overview based on the image analysis.
      - **Analysis:** Detailed characteristics and expected reactions to the ad, tailored to LinkedIn’s targeting options like job seniority, skills, member groups, interests, and more.

**Important Considerations:**

* **Base on Visual Evidence:** Ensure that all persona deductions are supported by visible elements in the image.
* **Utilize LinkedIn’s Targeting Options:** Include specifics such as:
   - **Location:** Suggestions on countries, cities, or regions.
   - **Company Industry and Size:** Likely industries and company sizes discernible from attire, accessories, or context.
   - **Job Functions and Job Titles:** Deductions based on attire or activities in the image.
   - **Job Seniority and Years of Experience:** Inferred from visual cues indicating professional maturity or youth.
   - **Schools, Degrees, Fields of Study:** Speculate based on any educational material or context.
   - **Skills:** Likely skills based on the activities or settings depicted.
   - **Member Groups and Interests:** Connections to professional groups or interests based on the image context.
   - **Traits:** Any discernible behaviors that align with LinkedIn's trait targeting.

**Example Persona Tables Based on an Image of a Corporate Event:**

**Persona Types Table:**

| Persona Type         | Description                               |
|----------------------|-------------------------------------------|
| Corporate Executive  | High-level decision-makers in large corporations. |
| Tech Innovator       | Professionals in cutting-edge technology sectors. |
| HR Specialist        | Human resources professionals focused on talent management and recruitment. |
| Marketing Guru       | Experts in digital marketing and brand strategy. |

**Detailed Persona Profiles Table:**

| Persona Type         | Description               | Analysis                                                                       |
|----------------------|---------------------------|--------------------------------------------------------------------------------|
| Corporate Executive  | Senior leaders at large firms | Target with content on leadership, enterprise solutions, and executive education. |
| Tech Innovator       | Professionals in tech industries | Focus ads on latest tech trends, innovations, and tech networking events. |
| HR Specialist        | Engaged in recruitment and HR policies | Advertise tools for recruitment, HR software solutions, and professional HR events. |
| Marketing Guru       | Skilled in marketing strategies | Target with digital marketing tools, analytics software, and marketing seminars. |

This structured approach should guide the generation of actionable persona insights that can be directly applied in LinkedIn advertising strategies.
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
                st.write("linkedin targeting Profile Result:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
        
    def x_profile(uploaded_file, is_image=True):
        prompt = f"""
Analyze the uploaded image to identify and describe potential advertising personas for X (formerly Twitter), focusing specifically on discernible visual elements and contextual cues within the image. The analysis should provide insights into user characteristics, preferences, and behaviors, aiming to enhance advertising strategies on X by pinpointing key personas who are likely to engage with tailored content. Emphasize the translation of visual and contextual observations into actionable marketing insights, suitable for developing targeted advertising campaigns on X.

**Persona Development Instructions:**

1. **Persona Types Table:**
   - Summarize 2-3 potential customer categories in a table, derived from the image analysis.
   - Columns should include 'Persona Type' and 'Description'.

2. **Detailed Persona Profiles Table:**
   - Expand on each persona type with a detailed profile in table format.
   - Columns should include:
      - **Name:** A hypothetical name representing the persona.
      - **Location:** General area (e.g., urban, suburban, rural) as suggested by the image.
      - **Gender:** If discernable from the image; otherwise, specify 'Any'.
      - **Language:** Likely primary language(s) inferred from the image.
      - **Interests:** Interests deduced from elements within the image.
      - **Behaviors:** Predicted X usage patterns (e.g., types of accounts followed, content interaction).
      - **Keywords:** Related hashtags or search terms.
      - **Topics:** Types of topics they might engage on X.
      - **Analysis:** Strategies on how to effectively target this persona on X.

**Important Considerations:**

* **Base on Visual Evidence:** Ensure all deductions about personas are supported by visible elements or scenarios within the image.
* **X-Specific Insights:** Emphasize insights that are specifically useful for targeting and engaging these personas on X.

**Example Output (Based on an image of a person engaging in urban photography):**

**Persona Types Table:**

| Persona Type     | Description                                  |
|------------------|----------------------------------------------|
| Urban Explorer   | Enjoys city life and urban photography.      |
| Tech Enthusiast  | Interested in the latest gadgets and tech.   |
| Social Influencer| Engages frequently with followers on social media. |

**Detailed Persona Profiles Table:**

| Name        | Location    | Gender | Language | Interests        | Behaviors                          | Keywords            | Topics            | Analysis                                      |
|-------------|-------------|--------|----------|------------------|------------------------------------|---------------------|-------------------|-----------------------------------------------|
| Alex        | Urban       | Any    | English  | Photography, tech| Posts tech reviews, cityscapes     | #UrbanLife, #Tech   | Technology, Photography | Use ads for tech gadgets, photography courses. |
| Jamie       | Urban       | Female | English  | Social media, fashion | Influences fashion trends, engages actively | #Fashion, #SocialMedia | Fashion, Lifestyle | Target with fashion brand collaborations, social media tools. |
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
                st.write("X (formerly Twitter) targeting Profile Result::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def flash_analysis(uploaded_file, is_image=True):
        prompt = f"""
        Imagine you are a visual content analyst reviewing a marketing asset ({'image' if is_image else 'video'}) for a client. Your goal is to provide a detailed, objective description that captures essential information relevant to marketing decisions.

        Instructions:

        1. Detailed Description:
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

        2. Cultural References and Symbolism:
            - Identify any cultural references, symbols, or visual metaphors that could be significant to the target audience.
            - Explain how these elements might be interpreted or resonate with the audience.

        3. Marketing Implications:
            - Briefly summarize the potential marketing implications based on the visual and textual elements.
            - Consider how the asset might appeal to different demographics or interests.
            - Mention any potential positive or negative associations it may evoke.

        4. Additional Notes:
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
                # Handle images directly
                image = Image.open(io.BytesIO(uploaded_file.read()))
                response = model.generate_content([custom_prompt, image])
                if response.candidates and len(response.candidates[0].content.parts) > 0:
                    return response.candidates[0].content.parts[0].text.strip()
                else:
                    st.error("Model did not provide a valid response or the response structure was unexpected.")
                    return None
            else:
                # Handle videos by processing each extracted frame
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)  # Ensure frames are extracted correctly
                if frames is None or not frames:
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None

                responses = []
                valid_responses = False  # Track if any valid responses were obtained
                for frame in frames:
                    response = model.generate_content([custom_prompt, frame])  # Pass frame in the correct format
                    if response.candidates and len(response.candidates[0].content.parts) > 0:
                        responses.append(response.candidates[0].content.parts[0].text.strip())
                        valid_responses = True
                    else:
                        responses.append("No valid response for this frame.")

                # Clean up the temporary file
                os.remove(tmp_path)

                if not valid_responses:
                    return "No valid responses were obtained from any of the frames."

                # Combine valid responses from all frames
                return "\n".join(responses)

        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def convert_to_json(results):
        return json.dumps(results, indent=4)

    def convert_to_xml(results):
        root = ET.Element("Results")
        for key, value in results.items():
            item = ET.SubElement(root, key)
            item.text = str(value)
        return ET.tostring(root, encoding='unicode')

    def create_download_link(data, file_format, filename):
        b64 = base64.b64encode(data.encode()).decode()
        return f'<a href="data:file/{file_format};base64,{b64}" download="{filename}">Download {file_format.upper()}</a>'

    st.title("Marketing Media Analysis AI Assistant")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Analysis Options")

        # Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["Basic", "Detailed", "Headlines", "Persona"])

        with tab1:
            basic_analysis = st.button("Basic Analysis")
            flash_analysis_button = st.button("Flash Analysis")

        with tab2:
            combined_analysis_V6 = st.button("Combined Marketing Analysis V6")
            text_analysis_button = st.button("Text Analysis")

        with tab3:
            headline_analysis_button = st.button("Headline Analysis")
            detailed_headline_analysis_button = st.button("Headline Optimization Report")
            main_headline_analysis_button = st.button("Main Headline Analysis")
            image_headline_analysis_button = st.button("Image Headline Analysis")
            supporting_headline_analysis_button = st.button("Supporting Headline Analysis")
            
        with tab4:
            meta_profile_button = st.button("Facebook targeting")
            linkedin_profile_button = st.button("LinkedIn targeting")
            x_profile_button = st.button("X (formerly Twitter) targeting")
            
        st.markdown("---")
        custom_prompt = st.text_area("Custom Prompt (Optional):")
        custom_prompt_button = st.button("Analyze with Custom Prompt")

    # --- Main Content Area ---

    # File Uploader with Enhanced UI
    uploaded_files = st.file_uploader(
        "Upload Marketing Media (Image or Video):", 
        accept_multiple_files=True, 
        type=["png", "jpg", "jpeg", "mp4", "avi"],
        help="Supported formats: PNG, JPG, JPEG, MP4, AVI",
    )

    # Display Uploaded Media (Responsive Design)
    for uploaded_file in uploaded_files:
        is_image = uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]

        with st.container():  # Use container for better layout
            col1, col2 = st.columns([1, 2])  # Adjust column ratio as needed

            with col1:
                if is_image:
                    image = Image.open(uploaded_file)
                    image = resize_image(image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                else:
                    st.video(uploaded_file, format="video/mp4")

            with col2:
                uploaded_file.seek(0)

            # Use the 'is_image' flag correctly in function calls
            if basic_analysis:
                with st.spinner("Performing basic analysis..."):
                    basic_analysis_result = analyze_media(uploaded_file, is_image)
                    if basic_analysis_result:
                        st.write("## Basic Analysis Results:")
                        st.json(basic_analysis_result)
                        json_data = convert_to_json(basic_analysis_result)
                        xml_data = convert_to_xml(basic_analysis_result)
                        st.markdown(create_download_link(json_data, "json", "basic_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "basic_analysis.xml"), unsafe_allow_html=True)

            if combined_analysis_V6:
                with st.spinner("Performing combined marketing analysis V6..."):
                    detailed_result_V6 = combined_marketing_analysis_V6(uploaded_file, is_image)
                    if detailed_result_V6:
                        st.write("## Combined Marketing Analysis V6 Results:")
                        st.markdown(detailed_result_V6)
                        json_data = convert_to_json(detailed_result_V6)
                        xml_data = convert_to_xml(detailed_result_V6)
                        st.markdown(create_download_link(json_data, "json", "combined_analysis_v6.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "combined_analysis_v6.xml"), unsafe_allow_html=True)

            if text_analysis_button:
                with st.spinner("Performing text analysis..."):
                    text_result = text_analysis(uploaded_file, is_image)
                    if text_result:
                        st.write("## Text Analysis Results:")
                        st.markdown(text_result)
                        json_data = convert_to_json(text_result)
                        xml_data = convert_to_xml(text_result)
                        st.markdown(create_download_link(json_data, "json", "text_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "text_analysis.xml"), unsafe_allow_html=True)

            if headline_analysis_button:
                with st.spinner("Performing headline analysis..."):
                    headline_result = headline_analysis(uploaded_file, is_image)
                    if headline_result:
                        st.write("## Headline Analysis Results:")
                        st.markdown(headline_result)
                        json_data = convert_to_json(headline_result)
                        xml_data = convert_to_xml(headline_result)
                        st.markdown(create_download_link(json_data, "json", "headline_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "headline_analysis.xml"), unsafe_allow_html=True)

            if main_headline_analysis_button:
                with st.spinner("Performing Main Headline Analysis..."):
                    main_headline_result = main_headline_detailed_analysis(uploaded_file, is_image)
                    if main_headline_result:
                        st.write("## Main Headline Analysis Results:")
                        st.markdown(main_headline_result)
                        json_data = convert_to_json(main_headline_result)
                        xml_data = convert_to_xml(main_headline_result)
                        st.markdown(create_download_link(json_data, "json", "main_headline_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "main_headline_analysis.xml"), unsafe_allow_html=True)

            if image_headline_analysis_button:
                with st.spinner("Performing Image Headline Analysis..."):
                    image_detailed_headline_result = image_headline_detailed_analysis(uploaded_file, is_image)
                    if image_detailed_headline_result:
                        st.write("## Image Headline Analysis Results:")
                        st.markdown(image_detailed_headline_result)
                        json_data = convert_to_json(image_detailed_headline_result)
                        xml_data = convert_to_xml(image_detailed_headline_result)
                        st.markdown(create_download_link(json_data, "json", "image_headline_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "image_headline_analysis.xml"), unsafe_allow_html=True)

            if supporting_headline_analysis_button:
                with st.spinner("Performing Supporting Headline Analysis..."):
                    supporting_detailed_headline_result = supporting_headline_detailed_analysis(uploaded_file, is_image)
                    if supporting_detailed_headline_result:
                        st.write("## Supporting Headline Analysis Report Results:")
                        st.markdown(supporting_detailed_headline_result)
                        json_data = convert_to_json(supporting_detailed_headline_result)
                        xml_data = convert_to_xml(supporting_detailed_headline_result)
                        st.markdown(create_download_link(json_data, "json", "supporting_headline_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "supporting_headline_analysis.xml"), unsafe_allow_html=True)

            if detailed_headline_analysis_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    detailed_headline_result = headline_detailed_analysis(uploaded_file, is_image)
                    if detailed_headline_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(detailed_headline_result)
                        json_data = convert_to_json(detailed_headline_result)
                        xml_data = convert_to_xml(detailed_headline_result)
                        st.markdown(create_download_link(json_data, "json", "headline_optimization_report.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "headline_optimization_report.xml"), unsafe_allow_html=True)

            if flash_analysis_button:
                with st.spinner("Performing Flash analysis..."):
                    flash_result = flash_analysis(uploaded_file, is_image)
                    if flash_result:
                        st.write("## Flash Analysis Results:")
                        st.markdown(flash_result)
                        json_data = convert_to_json(flash_result)
                        xml_data = convert_to_xml(flash_result)
                        st.markdown(create_download_link(json_data, "json", "flash_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "flash_analysis.xml"), unsafe_allow_html=True)

            if custom_prompt_button:
                with st.spinner("Performing custom prompt analysis..."):
                    custom_result = custom_prompt_analysis(uploaded_file, custom_prompt, is_image)
                    if custom_result:
                        st.write("## Custom Prompt Analysis Results:")
                        st.markdown(custom_result)
                        json_data = convert_to_json(custom_result)
                        xml_data = convert_to_xml(custom_result)
                        st.markdown(create_download_link(json_data, "json", "custom_prompt_analysis.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "custom_prompt_analysis.xml"), unsafe_allow_html=True)

            if meta_profile_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    meta_profile_result = meta_profile(uploaded_file, is_image)
                    if meta_profile_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(meta_profile_result)
                        json_data = convert_to_json(meta_profile_result)
                        xml_data = convert_to_xml(meta_profile_result)
                        st.markdown(create_download_link(json_data, "json", "headline_optimization_report.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "headline_optimization_report.xml"), unsafe_allow_html=True)
                        
            if linkedin_profile_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    linked_profile_result = linkedin_profile(uploaded_file, is_image)
                    if linked_profile_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(linked_profile_result)
                        json_data = convert_to_json(linked_profile_result)
                        xml_data = convert_to_xml(linked_profile_result)
                        st.markdown(create_download_link(json_data, "json", "headline_optimization_report.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "headline_optimization_report.xml"), unsafe_allow_html=True)
                    
            if x_profile_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    x_profile_result = x_profile(uploaded_file, is_image)
                    if x_profile_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(x_profile_result)
                        json_data = convert_to_json(x_profile_result)
                        xml_data = convert_to_xml(x_profile_result)
                        st.markdown(create_download_link(json_data, "json", "headline_optimization_report.json"), unsafe_allow_html=True)
                        st.markdown(create_download_link(xml_data, "xml", "headline_optimization_report.xml"), unsafe_allow_html=True)
