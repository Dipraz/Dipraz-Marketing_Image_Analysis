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
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize Generative AI model with generation configuration
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )
    def convert_to_rgb(image):
        """Convert an image to RGB format if it is not already."""
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image
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

    def overall_analysis(uploaded_file, is_image=True):
        prompt = """
Analyze the provided image for marketing effectiveness. First, provide detailed responses for the following:\n"
            "\n"
            "1. Asset Type: Clearly identify and describe the type of marketing asset. Examples include email, social media posts, advertisements, flyers, brochures, landing pages, etc.\n"
            "2. Purpose: Clearly state the specific purpose of this marketing asset. Provide a detailed explanation of how it aims to achieve this purpose. Examples include selling a product, getting more signups, driving traffic to a webpage, increasing brand awareness, engaging with customers, etc.\n"
            "3. Asset Audience: Identify the target audience for this marketing asset. Describe the demographics, interests, and needs of this audience. Examples include age group, gender, location, income level, education, interests, behaviors, etc.\n"
            "\n"
            "Then, for each aspect listed below, provide a score from 1 to 5 in increments of 0.5 (1 being low, 5 being high) and a concise explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format with the columns: Aspect, Score, Explanation, and Improvement. After the table, provide a concise explanation with suggestions for overall improvement. Here are the aspects to consider:\n"
            "\n"
            "The aspects to consider are:\n"
            "1. Creative Score: Assess the creativity of the design. Does it stand out and capture attention through innovative elements?\n"
            "2. Attention: Evaluate the order of content consumption in the uploaded image. Start by identifying and analyzing the headline for its prominence and position. Next, evaluate any additional text for visibility and reader engagement sequence. Assess the positioning of images in relation to the text, followed by an examination of interactive elements such as buttons. Discuss the order in which the content is consumed (e.g., headline first, then text, or image then text then button, etc.). Determine if the content prioritizes important information, and draws and holds attention effectively.\n"
            "3. Distinction: Does the content contain pictures that grab user attention? Does it appeal to the primal brain with and without text?\n"
            "4. Purpose and Value: Is the purpose and value clear within 3 seconds? Is the content product or customer-centric?\n"
            "5. Clarity: Evaluate the clarity of the design elements. Are the visuals and text easy to understand?\n"
            "6. First Impressions: Analyze the initial impact of the design. Does it create a strong positive first impression?\n"
            "7. Headline Review: Evaluate the headline for clarity, conciseness, customer centricity, SEO keyword integration, emotional appeal, uniqueness, urgency, benefit to the reader, audience targeting, length, use of numbers/lists, brand consistency, and power words.\n"
            "8. Headline keywords and emotional appeal: Does the headline incorporate keywords and evoke an emotional response?\n"
            "9. Visual Cues and Color Usage: Does the image use visual cues and colors to draw attention to key elements? Analyze how color choices, contrast, and elements like arrows or frames guide the viewer's attention.\n"
            "10. Button Clarity: Are any CTA buttons present clearly labelled and easy to understand? Evaluate the use of text size, font choice, and placement for optimal readability.\n"
            "11. Engagement: Assess the engagement level of the user experience. Is the UX design captivating and satisfying to interact with?\n"
            "12. Trust: Assess the trustworthiness of the content based on visual and textual elements. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)? Assess the credibility, reliability, and intimacy conveyed by the content.\n"
            "13. Motivation: Assess the design's ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof?\n"
            "14. Influence: Analyze the influence of the design. Does the asset effectively persuade viewers and lead them towards a desired action?\n"
            "15. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "16. Experience: Assess the overall user experience. How well does the design facilitate a smooth and enjoyable interaction?\n"
            "17. Memorability: Evaluate how memorable the design is. Does it leave a lasting impression?\n"
            "18. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? (1: Very Dense & Difficult, 5: Clear & Easy to Understand)\n"
             "19. Tone: Is the tone used to increase the effectiveness of the asset effectively?\n"
            "20. Framing: Is framing of the message used to increase the effectiveness of the asset effectively?\n"
            "21. Content Investment: Blocks containing paragraphs of text will not be consumed by busy users and would require time to read – this is negative, as the users will not spend the time. Is the amount of content presented kept short and clear?\n"
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
As a UX design and marketing analysis consultant, you are tasked with reviewing the text content of a marketing asset (image or video, excluding the headline) for a client. Your goal is to provide a comprehensive analysis of the text's effectiveness and offer actionable recommendations for improvement.

**Part 1: Text Extraction and Contextualization**

* **Image Analysis:**
  1. **Text Extraction:** Thoroughly identify and extract ALL visible text within the image, including headlines, body copy, captions, calls to action, taglines, logos, and any other textual elements.
  2. **Presentation:** Present the extracted text in a clear, bulleted list format, maintaining the original order and structure to the extent possible.
  3. **Visual Analysis:**
     * **Placement:** Specify the location of each text element within the image (e.g., top left, centered, bottom right). Note any instances of overlapping text or elements that might hinder readability.
     * **Font Choices:** Describe the font style (serif, sans-serif, script, etc.), weight (bold, regular, light), size, and color of each distinct text element.
     * **Visual Relationships:** Explain how the text interacts with other visual elements (images, graphics, colors) and how it contributes to the overall message and hierarchy of information.

* **Video Analysis:**
  1. **Key Frame Identification:** Select the most representative frame(s) that showcase the primary text content.
  2. **Text Extraction:** Extract and present the text from these key frames in a clear, bulleted list format.
  3. **Temporal Analysis:** Briefly describe any significant textual changes or patterns that occur throughout the video.
  4. **Integration with Visuals and Audio:** Analyze how the text interacts with the video's visuals (scenes, characters, actions) and audio (dialogue, music, sound effects).

**Part 2: Textual Assessment**

Evaluate the extracted text based on the following criteria. For each aspect, provide a score from 1 (poor) to 5 (excellent) in increments of 0.5, a concise justification of the score highlighting strengths and weaknesses, and specific, actionable suggestions for enhancing the text's effectiveness. Structure your assessment in a table format with columns for Aspect, Score, Explanation, and Improvement.

| Aspect                     | Score | Explanation                                                                                                | Improvement                                                                                      |
|----------------------------|-------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Clarity and Conciseness    |       | Assess how easy it is to understand the text, considering sentence structure, vocabulary, and overall flow.| Suggest ways to simplify language, eliminate jargon, or shorten sentences.                       |
| Customer Focus             |       | Evaluate if the text addresses the customer's needs and uses language that resonates with them.            | Offer suggestions for incorporating the customer's perspective more effectively.                  |
| Engagement                 |       | Assess how compelling the text is, including storytelling, humor, and value proposition.                   | Propose methods to enhance engagement, such as using stronger verbs or improving formatting.      |
| Reading Effort             |       | Evaluate the ease of reading and understanding the text, considering vocabulary and sentence structure.   | Suggest using simpler structures and more accessible vocabulary.                                  |
| Purpose and Value          |       | Determine if the text's purpose and value proposition are clear and compelling.                            | Recommend clarifying the key message or benefits more directly.                                   |
| Motivation & Persuasion    |       | Analyze the text's persuasive power, including calls to action and social proof.                          | Suggest strengthening persuasive elements, such as adding stronger calls to action.               |
| Depth and Detail           |       | Evaluate if the text provides sufficient information and detail for the target audience.                   | Suggest adding or condensing information as necessary to meet audience needs.                     |
| Trustworthiness            |       | Assess the credibility of the text and its success in building trust with the audience.                    | Suggest ways to enhance trustworthiness, such as using more transparent language.                 |
| Memorability               |       | Evaluate if the text includes memorable elements such as catchy phrases or unique storytelling techniques. | Recommend incorporating memorable language or anecdotes to enhance retention.                     |
| Emotional Appeal           |       | Determine if the text evokes appropriate emotions aligned with the brand image and message.                | Suggest using language that evokes specific emotions to strengthen emotional impact.              |
| Uniqueness & Differentiation|       | Analyze if the text differentiates the brand from competitors effectively.                                  | Suggest ways to enhance uniqueness, such as developing a stronger brand voice.                    |
| Urgency and Curiosity      |       | Assess if the text creates a sense of urgency or curiosity, enticing the audience to learn more.           | Recommend methods to increase urgency, such as highlighting limited-time offers.                  |
| Benefit Orientation        |       | Evaluate if the text clearly articulates the benefits of the product/service to the target audience.       | Suggest making benefits more explicit and customer-centric.                                       |
| Target Audience Relevance  |       | Determine if the text's language, tone, and style are appropriate and appealing to the intended audience.  | Suggest adjustments to better align with the audience's interests and needs.                      |
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
Imagine you are a marketing consultant reviewing the headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the various headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Headline Extraction and Context**

**Image/Video:**
1. **Headline Identification:**
   * **Main Headline:** Clearly state the main headline extracted from the image or video.
   * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
   * **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.

**Part 2A: Main Headline Analysis**
"Analyze the provided image content alongside the main headline text to assess the main headline's effectiveness. Rate each criterion on a scale from 1 to 5 using increments of 0.5 (1 being poor, 5 being excellent), and provide an explanation for each score based on the synergy between the image and headline, and a recommendation on how it could be improved. Present your results in a table format with columns labeled: Criterion, Score, Explanation, Recommendation."

The criteria to assess are:
1. **Overall Effectiveness:** Summarize the overall effectiveness of the headline.
2. **Clarity:** How clearly does the headline convey the main point?
3. **Customer Focus:** Does the headline emphasize a customer-centric approach?
4. **Relevance:** How accurately does the headline reflect the content of the image?
5. **Keywords:** Are relevant keywords included naturally?
6. **Emotional Appeal:** Does the headline evoke curiosity or an emotional response, considering the image content?
7. **Uniqueness:** How original and creative is the headline?
8. **Urgency & Curiosity:** Does the headline create a sense of urgency or pique curiosity, considering the image?
9. **Benefit-Driven:** Does the headline convey a clear benefit or value proposition, aligned with the image content?
10. **Target Audience:** Is the headline tailored to resonate with the specific target audience, considering the image's visual cues?
11. **Length & Format:** Does the headline fall within an ideal length of 6-12 words?

**Part 2B: Image Headline Analysis**
"Analyze the provided image content alongside the image headline text to assess the image headline's effectiveness. Rate each criterion on a scale from 1 to 5 using increments of 0.5 (1 being poor, 5 being excellent), and provide an explanation for each score based on the synergy between the image and headline, and a recommendation on how it could be improved. Present your results in a table format with columns labeled: Criterion, Score, Explanation, Recommendation."

The criteria to assess are:
1. **Overall Effectiveness:** Summarize the overall effectiveness of the headline.
2. **Clarity:** How clearly does the headline convey the main point?
3. **Customer Focus:** Does the headline emphasize a customer-centric approach?
4. **Relevance:** How accurately does the headline reflect the content of the image?
5. **Keywords:** Are relevant keywords included naturally?
6. **Emotional Appeal:** Does the headline evoke curiosity or an emotional response, considering the image content?
7. **Uniqueness:** How original and creative is the headline?
8. **Urgency & Curiosity:** Does the headline create a sense of urgency or pique curiosity, considering the image?
9. **Benefit-Driven:** Does the headline convey a clear benefit or value proposition, aligned with the image content?
10. **Target Audience:** Is the headline tailored to resonate with the specific target audience, considering the image's visual cues?
11. **Length & Format:** Does the headline fall within an ideal length of 6-12 words?

**Part 2C: Supporting Headline Analysis**
"Analyze the provided image content alongside the supporting headline text to assess the supporting headline's effectiveness. Rate each criterion on a scale from 1 to 5 using increments of 0.5 (1 being poor, 5 being excellent), and provide an explanation for each score based on the synergy between the image and headline, and a recommendation on how it could be improved. Present your results in a table format with columns labeled: Criterion, Score, Explanation, Recommendation."

The criteria to assess are:
1. **Overall Effectiveness:** Summarize the overall effectiveness of the headline.
2. **Clarity:** How clearly does the headline convey the main point?
3. **Customer Focus:** Does the headline emphasize a customer-centric approach?
4. **Relevance:** How accurately does the headline reflect the content of the image?
5. **Keywords:** Are relevant keywords included naturally?
6. **Emotional Appeal:** Does the headline evoke curiosity or an emotional response, considering the image content?
7. **Uniqueness:** How original and creative is the headline?
8. **Urgency & Curiosity:** Does the headline create a sense of urgency or pique curiosity, considering the image?
9. **Benefit-Driven:** Does the headline convey a clear benefit or value proposition, aligned with the image content?
10. **Target Audience:** Is the headline tailored to resonate with the specific target audience, considering the image's visual cues?
11. **Length & Format:** Does the headline fall within an ideal length of 6-12 words?

**Part 3: Improved Headline Suggestions**
"Provide three improved headlines for EACH of the headline types that better align with the image content. Explain why you have selected these. Present your results in a table format with columns labeled: Headline Type (Main/Image/Supporting), Headline Recommendation, Explanation. This table must contain 9 rows."
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
**Part 1A: Main Headline Optimization Analysis**
"Analyze the provided image content alongside the main headline text to assess the headline's effectiveness. Evaluate each of the following criteria, provide an explanation based on the synergy between the image and the headline, and offer recommendations for improvement. Present your results in a table format with columns labeled: Criterion, Assessment, Explanation, Recommendation."

The criteria to assess are:
1. **Word count:** Number of words in the headline.
2. **Keyword Relevance:** Assessment of how well the headline incorporates relevant keywords or phrases.
3. **Common words:** Number of common words.
4. **Uncommon Words:** Number of uncommon words.
5. **Power Words:** Number of words with strong persuasive potential.
6. **Emotional words:** Number of words conveying emotion (e.g., positive, negative, neutral).
7. **Sentiment:** Overall sentiment: positive, negative, or neutral.
8. **Reading Grade Level:** Estimated grade level required to understand the headline.

**Part 1B: Image Headline Optimization Analysis**
"Analyze the provided image content alongside the image headline text to assess the headline's effectiveness. Evaluate each of the following criteria, provide an explanation based on the synergy between the image and the headline, and offer recommendations for improvement. Present your results in a table format with columns labeled: Criterion, Assessment, Explanation, Recommendation."

The criteria to assess are:
1. **Word count:** Number of words in the headline.
2. **Keyword Relevance:** Assessment of how well the headline incorporates relevant keywords or phrases.
3. **Common words:** Number of common words.
4. **Uncommon Words:** Number of uncommon words.
5. **Power Words:** Number of words with strong persuasive potential.
6. **Emotional words:** Number of words conveying emotion (e.g., positive, negative, neutral).
7. **Sentiment:** Overall sentiment: positive, negative, or neutral.
8. **Reading Grade Level:** Estimated grade level required to understand the headline.

**Part 1C: Supporting Headline Optimization Analysis**
"Analyze the provided image content alongside the supporting headline text to assess the headline's effectiveness. Evaluate each of the following criteria, provide an explanation based on the synergy between the image and the headline, and offer recommendations for improvement. Present your results in a table format with columns labeled: Criterion, Assessment, Explanation, Recommendation."

The criteria to assess are:
1. **Word count:** Number of words in the headline.
2. **Keyword Relevance:** Assessment of how well the headline incorporates relevant keywords or phrases.
3. **Common words:** Number of common words.
4. **Uncommon Words:** Number of uncommon words.
5. **Power Words:** Number of words with strong persuasive potential.
6. **Emotional words:** Number of words conveying emotion (e.g., positive, negative, neutral).
7. **Sentiment:** Overall sentiment: positive, negative, or neutral.
8. **Reading Grade Level:** Estimated grade level required to understand the headline.        
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
        
    def main_headline_analysis(uploaded_file, is_image=True):
        prompt = f"""
Imagine you are a marketing consultant reviewing the main headline text of a marketing asset ({'image' if is_image else 'video'}) for a client.
Your task is to assess the main headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Main Headline Context**
    **Image/Video:**
        - **Main Headline Identification:** Extract and clearly state the main headline from the image or video.

    **Part 2: Main Headline Analysis**
    Present the results in a well-formatted table for the main headline:
    | Criterion             | Assessment                   | Explanation                                                      | Recommendation                                       |
    |-----------------------|------------------------------|------------------------------------------------------------------|------------------------------------------------------|
    | Word Count            | [Automatic count] words      | The headline has [x] words, which is [appropriate/lengthy].     | Consider [reducing/increasing] the word count to [y].|
    | Keyword Relevance     | [High/Moderate/Low]          | The headline [includes/misses] relevant keywords such as [x].   | Incorporate [more/specific] keywords like [y].       |
    | Common Words          | [Number] common words        | Common words [enhance/reduce] readability and appeal.           | [Increase/reduce] the use of common words.           |
    | Uncommon Words        | [Number] uncommon words      | Uncommon words make the headline [stand out/confusing].         | Balance [common/uncommon] words for clarity.         |
    | Power Words           | [Number] power words         | Power words [create urgency/may overwhelm] the reader.          | Use power words [more sparingly/more effectively].   |
    | Emotional Words       | [Number] emotional words     | Emotional tone is [effective/overdone/subtle].                  | Adjust the emotional tone by [modifying x].          |
    | Sentiment             | [Positive/Negative/Neutral]  | The sentiment is [not aligning well/matching] with the image.   | Match the sentiment more closely with the image.     |
    | Reading Grade Level   | [Grade level] required       | The headline is [too complex/simple] for the target audience.   | Adapt the reading level to [simplify/complexify].    |
    **Part 3: Improved Headline Suggestions**
    Provide suggestions for improving the main headline considering the overall analysis.
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
    def image_headline_analysis(uploaded_file, is_image=True):
        prompt = f"""
Imagine you are a marketing consultant reviewing the image headline text of a marketing asset ({'image' if is_image else 'video'}) for a client.
Your task is to assess the image headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Image Headline Context**
    **Image/Video:**
        - **Image Headline Identification:** Extract and clearly state the separate headline from the image or video.

    **Part 2: Image Headline Analysis**
    Analyze and format the results:
    | Criterion             | Assessment                   | Explanation                                                      | Recommendation                                       |
    |-----------------------|------------------------------|------------------------------------------------------------------|------------------------------------------------------|
    | Word Count            | [Automatic count] words      | The headline length is [appropriate/lengthy] for visibility.     | Adjust the word count to [increase/decrease] clarity.|
    | Keyword Relevance     | [High/Moderate/Low]          | Headline's keywords [align/do not align] with visual content.    | Enhance keyword alignment for better SEO.            |
    | Common Words          | [Number] common words        | Common words [aid/hinder] immediate comprehension.               | Optimize common word usage for [audience/type].      |
    | Uncommon Words        | [Number] uncommon words      | Uncommon words add [uniqueness/confusion].                       | Find a balance in word rarity for better engagement.  |
    | Power Words           | [Number] power words         | Uses power words to [effectively/too aggressively] engage.       | Adjust power word usage for subtlety.                |
    | Emotional Words       | [Number] emotional words     | Emotional words [evoke strong/a weak] response.                  | Modify emotional words to better suit the tone.      |
    | Sentiment             | [Positive/Negative/Neutral]  | Sentiment [supports/contradicts] the visual theme.               | Align the sentiment more with the visual message.    |
    | Reading Grade Level   | [Grade level] required       | Reading level is [ideal/not ideal] for the target demographic.   | Tailor the complexity to better fit the audience.     |
    **Part 3: Recommendations**
    Suggest three improved headlines based on the analysis.
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
    def supporting_headline_analysis(uploaded_file, is_image=True):
        prompt = f"""
    Review anyImagine you are a marketing consultant reviewing the supporting headline text of a marketing asset ({'image' if is_image else 'video'}) for a client.
    Your task is to assess the supporting headline's effectiveness based on various linguistic and marketing criteria. supporting headlines in the provided image or video frame as a marketing consultant.
    **Part 1: Supporting Headline Context**
    **Image/Video:**
        - **Supporting Headline Identification:** Identify and state any supporting headlines.

    **Part 2: Supporting Headline Analysis**
    Format the results as follows:
    | Criterion             | Assessment                   | Explanation                                                      | Recommendation                                       |
    |-----------------------|------------------------------|------------------------------------------------------------------|------------------------------------------------------|
    | Word Count            | [Automatic count] words      | The supporting headline's length is [optimal/too long/short].    | Aim for a word count of [x] for better engagement.   |
    | Keyword Relevance     | [High/Moderate/Low]          | Keywords used are [not sufficiently/sufficiently] relevant.      | Incorporate more relevant keywords like [y].         |
    | Common Words          | [Number] common words        | Utilization of common words [enhances/detracts from] impact.     | Adjust common word usage to improve clarity.         |
    | Uncommon Words        | [Number] uncommon words      | Uncommon words help [distinguish/muddle] the message.            | Use uncommon words to [highlight/clarify] message.   |
    | Power Words           | [Number] power words         | Power words [effectively/ineffectively] persuade the audience.   | Refine the use of power words for better impact.     |
    | Emotional Words       | [Number] emotional words     | Emotional expression is [strong/weak], affecting impact.         | Enhance/reduce emotional wording for desired effect. |
    | Sentiment             | [Positive/Negative/Neutral]  | Sentiment of the headline [aligns/conflicts] with main content.  | Adjust sentiment to [complement/contrast] main tone. |
    | Reading Grade Level   | [Grade level] required       | The complexity suits [or does not suit] the intended audience.   | Modify to [simplify/complexify] reading level.       |
    **Part 3: Revised Headline Suggestions**
    Offer alternative headlines that enhance effectiveness based on the detailed analysis.
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
Based on the following targeting elements for Facebook, please describe 4 persona types
that are most likely to respond to the add. Please present these in a table (Persona Type,
Description). Once you have identified these, create 4 personas (including names) who
would be likely to purchase this product, and describe how you would expect them to react
to it detailing the characteristics. Present each persona with a table (Persona Type,
Description, Analysis) of the characteristics and analysis. Please include each of the
characteristic that can be selected in the Facebook targeting, and what you would select.

Location: Target users based on countries, states, cities, or even specific addresses and zip
codes.
Age: Select the age range of the audience.
Gender: Target ads specifically to men, women, or all genders.
Languages: Target users based on the languages they speak.
Interests: Based on user activities, liked pages, and closely related topics. This includes
interests in entertainment, fitness, hobbies, and more.

Behaviors: Includes user behavior based on device usage, travel patterns, purchase
behavior, and more.
Purchase Behavior: Target users who have made purchases in specific categories.
Device Usage: Target based on the devices used to access Facebook, like mobiles, tablets,
or desktops.
Connections to Your Pages, Apps, or Events: Target users who have already interacted with
your business on Facebook or exclude them to find new audiences.
Target users based on important life events like anniversaries, birthdays, recently moved,
newly engaged, or having a baby.
Education Level: Target users based on their educational background.
Education Fields of Study: Target users based on their educational background.
Job Title: Target professionals based on their job information.
Job Title Industries: Target professionals based on their job information.
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
Based on the following targeting elements for Linkedin, please describe 4 persona types that
are most likely to respond to the add. Please present these in a table (Persona Type,
Description). Once you have identified these, create 4 personas (including names) who
would be likely to purchase this product, and describe how you would expect them to react
to it detailing the characteristics. Present each persona with a table (Persona Type,
Description, Analysis) of the characteristics and analysis. Please include each of the
characteristic that can be selected in the Linkedin targeting, and what you would select.

Location: Country, city, or region.
Age: Though LinkedIn does not directly allow age and gender targeting, these can be
inferred through other demographic details.
Gender: Though LinkedIn does not directly allow age and gender targeting, these can be
inferred through other demographic details.
Company Industry: Reach professionals in particular industries.
Company Size: Target companies based on the number of employees.
Job Functions: Target users with specific job functions within companies.
Job Seniority: From entry-level to senior executives and managers.
Job Titles: Specific job titles, reaching users with particular roles.
Years of Experience: Reach users based on how long they’ve been in the professional
workforce.
Schools: Alumni of specific educational institutions.
Degrees: Users who hold specific degrees.
Fields of Study: Users who studied specific subjects.
Skills: Users who have listed specific skills on their profiles.
Member Groups: Target members of LinkedIn groups related to professional interests.
Interests: Based on content users interact with or their listed interests.
Traits: Includes aspects like member traits, which can reflect user activities and behaviors on
LinkedIn.
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
Based on the following targeting elements for X, please describe 4 persona types that are
most likely to respond to the ad. Please present these in a table (Persona Type,
Description). Once you have identified these, create 4 personas (including names) who
would be likely to purchase this product, and describe how you would expect them to react
to it detailing the characteristics. Present each persona with a table (Persona Type,
Description, Analysis) of the characteristics and analysis. Please include each of the
characteristics that can be selected in the X targeting, and what you would select.

Location: Target users by country, region, or metro area. More granular targeting, such as
city or postal code is also available.
Gender: You can select audiences based on gender.
Language: Target users based on the language they speak.
Interests: Target users based on their interests, which are inferred from their activities and
the topics they engage with on X.
Events: Target ads around specific events, both global and local, that generate significant
engagement on the platform.
Behaviors: Target based on user behaviors and actions, such as what they tweet or engage
with.
Keywords: Target users based on keywords in their tweets or tweets they engage with. This
can be particularly useful for capturing intent and interest in real-time.
Topics: Engage users who are part of conversations around predefined or custom topics.
Device: Target users based on the devices or operating systems they use to access X.

Carrier: Target users based on their mobile carrier, which can be useful for mobile-specific
campaigns.
Geography: Targeting based on user location can be fine-tuned to match the cultural context
and regional norms. 
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
        """Analyzes an image or video using a custom prompt."""

        try:
            if is_image:
                # Handle single image (ensure RGB format)
                image = Image.open(uploaded_file)
                image = convert_to_rgb(image)

                # Generate response using the image
                response = model.generate_content([custom_prompt, image]) 
            else:
                # Handle video file (extract frames)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                frames = extract_frames(tmp_path)
                if frames is None or not frames:
                    raise ValueError("No frames extracted from the video. Please check the video format.")

                responses = []
                for frame in frames:
                    # Ensure frame is in RGB format
                    frame = convert_to_rgb(frame)

                    # Generate response using the frame
                    response = model.generate_content([custom_prompt, frame])

                    if response and response.candidates and len(response.candidates[0].content.parts) > 0:
                        responses.append(response.candidates[0].content.parts[0].text.strip())
                    else:
                        responses.append("No valid response for this frame.")
                
                os.remove(tmp_path)  # Clean up the temporary video file

                return "\n\n".join(responses)  # Combine individual frame responses
        
            # Process the response for both image and video
            if response and response.candidates and len(response.candidates[0].content.parts) > 0:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                raise ValueError("Model did not provide a valid response or the response structure was unexpected.")

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"An error occurred while processing the media: {e}")

        return None  # Return None to signal an error occurred

        return None  # Return None on error
    def compare_images(image1, image2):
        prompt = """
    Provide a focused analysis comparing two images, emphasizing their observable visual elements and marketing attributes and overall effectiveness together. Your analysis should be factual, based on visible content only, and avoid inferential or speculative details. Discuss:
    
    - **Visual Similarities**: Identify and describe specific visual elements that are clearly similar in both images (e.g., color schemes, object types, layout).
    - **Visual Differences**: Detail the visual differences that are evident, such as color variations, the presence or absence of certain elements, and different composition styles.
    - **Marketing Messages**: Compare any explicit marketing messages or texts that are present in both images. If text is not present, focus only on the overt themes portrayed by the visuals.
    - **Comparative Strengths and Weaknesses**: Assess the strengths and weaknesses of each image strictly based on the visible content and its potential impact in a marketing context.

    Conclude with a brief evaluation on which image might perform better in a marketing scenario, based solely on the observed elements.

    At the end, Summarize the key points of your comparison in a structured table. Ensure that each entry in the table is directly supported by visible content in the images:

    | Aspect              | Image 1 Details              | Image 1 Improvements        | Image 2 Details              | Image 2 Improvements        | Comparative Insight                    |
    |---------------------|------------------------------|-----------------------------|------------------------------|-----------------------------|----------------------------------------|
    | Visual Appeal       | Describe the visual appeal based on colors, layout, and imagery. | Suggest improvements based on marketing effectiveness. | Describe the visual appeal based on colors, layout, and imagery. | Suggest improvements based on marketing effectiveness. | Which image has a stronger visual appeal and why?          |
    | Marketing Message   | State the explicit marketing message if present. | How could the message be made more clear or impactful? | State the explicit marketing message if present. | How could the message be made more clear or impactful? | Which image communicates its message more effectively?     |
    | Overall Impact      | Detail the immediate impact or the emotional appeal. | Suggest how to enhance the impact. | Detail the immediate impact or the emotional appeal. | Suggest how to enhance the impact. | Overall, which image creates a stronger impact and why?    |
    
    """
        try:
            response = model.generate_content([prompt, image1, image2])
            if response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                st.error("Model did not provide a valid response.")
                return None
        except Exception as e:
            st.error(f"Failed to process the images: {e}")
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic", "Detailed", "Headlines", "Persona","Others"])

        with tab1:
            basic_analysis = st.button("Basic Analysis")
            flash_analysis_button = st.button("Flash Analysis")

        with tab2:
            overall_analysis_button = st.button("Overall Marketing Analysis")
            text_analysis_button = st.button("Text Analysis")

        with tab3:
            headline_analysis_button = st.button("Headline Analysis")
            detailed_headline_analysis_button = st.button("Headline Optimization Report")

        with tab4:
            meta_profile_button = st.button("Facebook targeting")
            linkedin_profile_button = st.button("LinkedIn targeting")
            x_profile_button = st.button("X (formerly Twitter) targeting")
            
        with tab5:
            main_headline_analysis_button = st.button("Main Headline Analysis")
            image_headline_analysis_button = st.button("Image Headline Analysis")
            supporting_headline_analysis_button = st.button("Supporting Headline Analysis")
            main_headline_text_analysis_button = st.button("Main Headline Text Analysis")
            image_headline_text_analysis_button = st.button("Image Headline Text Analysis")
            supporting_headline_text_analysis_button = st.button("Supporting Headline Text Analysis")
            
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
        key="general_media_uploader"  # Unique key for the general media uploader
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

            if overall_analysis_button:
                with st.spinner("Performing Overall Marketing Analysis..."):
                    overall_analysis_result = overall_analysis(uploaded_file, is_image)
                    if overall_analysis_result:
                        st.write("## Overall Marketing Analysis Results:")
                        st.markdown(overall_analysis_result)
            if text_analysis_button:
                with st.spinner("Performing text analysis..."):
                    text_result = text_analysis(uploaded_file, is_image)
                    if text_result:
                        st.write("## Text Analysis Results:")
                        st.markdown(text_result)

            if headline_analysis_button:
                with st.spinner("Performing headline analysis..."):
                    headline_result = headline_analysis(uploaded_file, is_image)
                    if headline_result:
                        st.write("## Headline Analysis Results:")
                        st.markdown(headline_result)

            if main_headline_analysis_button:
                with st.spinner("Performing Main Headline Analysis..."):
                    main_headline_result = main_headline_detailed_analysis(uploaded_file, is_image)
                    if main_headline_result:
                        st.write("## Main Headline Analysis Results:")
                        st.markdown(main_headline_result)
            if image_headline_analysis_button:
                with st.spinner("Performing Image Headline Analysis..."):
                    image_detailed_headline_result = image_headline_detailed_analysis(uploaded_file, is_image)
                    if image_detailed_headline_result:
                        st.write("## Image Headline Analysis Results:")
                        st.markdown(image_detailed_headline_result)

            if supporting_headline_analysis_button:
                with st.spinner("Performing Supporting Headline Analysis..."):
                    supporting_detailed_headline_result = supporting_headline_detailed_analysis(uploaded_file, is_image)
                    if supporting_detailed_headline_result:
                        st.write("## Supporting Headline Analysis Report Results:")
                        st.markdown(supporting_detailed_headline_result)

            if detailed_headline_analysis_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    detailed_headline_result = headline_detailed_analysis(uploaded_file, is_image)
                    if detailed_headline_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(detailed_headline_result)
            if main_headline_text_analysis_button:
                with st.spinner("Performing Image Headline Analysis..."):
                    main_headline_analysis_result = main_headline_analysis(uploaded_file, is_image)
                    if main_headline_analysis_result:
                        st.write("## Image Headline Analysis Results:")
                        st.markdown(main_headline_analysis_result)                    
            if image_headline_text_analysis_button:
                with st.spinner("Performing Image Headline Analysis..."):
                    image_headline_analysis_result = image_headline_analysis(uploaded_file, is_image)
                    if image_headline_analysis_result:
                        st.write("## Image Headline Analysis Results:")
                        st.markdown(image_headline_analysis_result)
            if supporting_headline_text_analysis_button:
                with st.spinner("Performing Image Headline Analysis..."):
                    supporting_headline_analysis_result = supporting_headline_analysis(uploaded_file, is_image)
                    if supporting_headline_analysis_result:
                        st.write("## Image Headline Analysis Results:")
                        st.markdown(supporting_headline_analysis_result)
            if flash_analysis_button:
                with st.spinner("Performing Flash analysis..."):
                    flash_result = flash_analysis(uploaded_file, is_image)
                    if flash_result:
                        st.write("## Flash Analysis Results:")
                        st.markdown(flash_result)
            if custom_prompt_button and uploaded_files:
                with st.spinner("Performing custom prompt analysis..."):
                    is_image = uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]
                    custom_result = custom_prompt_analysis(uploaded_files[0], custom_prompt, is_image) 
                    if custom_result:
                        st.write("## Custom Prompt Analysis Results:")
                        st.markdown(custom_result)


            if meta_profile_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    meta_profile_result = meta_profile(uploaded_file, is_image)
                    if meta_profile_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(meta_profile_result)
                        
            if linkedin_profile_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    linked_profile_result = linkedin_profile(uploaded_file, is_image)
                    if linked_profile_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(linked_profile_result)
                    
            if x_profile_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    x_profile_result = x_profile(uploaded_file, is_image)
                    if x_profile_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(x_profile_result)
                        
            # File Uploader for Image Comparison
            uploaded_files_comparison = st.file_uploader(
                "Upload Two Marketing Images for Comparison:",
                accept_multiple_files=True,
                type=["png", "jpg", "jpeg"],
                help="Upload exactly two images for comparison."
            )

            # Display Uploaded Images for Comparison (if any)
            if uploaded_files_comparison and len(uploaded_files_comparison) == 2:
                image1, image2 = [convert_to_rgb(Image.open(file)) for file in uploaded_files_comparison]
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image1, caption="Image 1", use_column_width=True)
                with col2:
                    st.image(image2, caption="Image 2", use_column_width=True)

                # --- Image Comparison Options ---
                with st.expander("Image Comparison Options"):
                    # Button to trigger standard comparison
                    if st.button("Compare Images (Standard)", key="standard_compare_button"):
                        with st.spinner("Comparing images..."):
                            comparison_result = compare_images(image1, image2)  # Use the same compare_images function
                            if comparison_result:
                                st.write("## Image Comparison Results:")
                                st.markdown(comparison_result)

                    # Custom Prompt for Comparison
                    st.markdown("---")
                    custom_prompt = st.text_area(
                        "Custom Prompt for Comparison (Optional):",
                        value="""Provide a detailed comparison of these images, focusing on [your specific areas of interest]. Explain the similarities, differences, and potential impact on the target audience.""",
                        height=150,  # Adjust height as needed
                    )
                    if st.button("Compare with Custom Prompt", key="custom_compare_button"):
                        with st.spinner("Comparing images with custom prompt..."):
                            response = model.generate_content([custom_prompt, image1, image2])
                            if response.candidates and len(response.candidates[0].content.parts) > 0:
                                st.write("## Custom Comparison Results:")
                                st.markdown(response.candidates[0].content.parts[0].text.strip())
                            else:
                                st.error("Model did not provide a valid response.")
