import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import io
import google.generativeai as genai
import cv2
import tempfile
import re

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
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            pil_image = Image.open(io.BytesIO(buffer))
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
            "Analyze the media (image or video frame) for various marketing aspects, ensuring consistent results for each aspects. "
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
        prompt = """
        Imagine you are a marketing consultant reviewing a visual asset (image or video) and its headline(s) for a client. Your goal is to provide a comprehensive analysis of the headline's effectiveness across various key criteria.
        1. Content Analysis:
           - If analyzing an image, examine the visual elements such as key objects, colors, composition, and style, along with the underlying messages or themes conveyed.
           - If analyzing a video, focus on the most representative frame and consider the overall visual and auditory elements that contribute to the message.
        
        2. Headline Identification:
           - Clearly identify the main headline, Image Headline, and any supporting headlines present in the asset.
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
           - Display the main headline's evaluation in a table format with columns: Criterion, Score, Explanation, and Improvements. Ensure every cell in the table is filled, and mainly not any of the Improvements column should left empty or N\A, note that.
        
        5. Supporting Headline Evaluation (Optional):
           - If applicable, briefly assess any image headline or supporting headlines and note if they require further analysis. Consider creating a separate table if a more in-depth analysis is needed.
        
        6. Total Score:
           - Calculate and display the total score for the main headline based on the evaluations.
        
        7. Improved Headlines:
           - Provide three alternative headlines for the main headline and also the Image Headline that addresses any weaknesses identified. Ensure these headlines are free of colons, diverse in structure and style, and aligned with the visual content and the target audience.
        
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
                if not frames:
                    st.error("No frames were extracted from the video. Please check the video format.")
                    return None
                response = model.generate_content([prompt, frames[0]])

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
Imagine you are a marketing consultant reviewing the headline text of a marketing asset (image or video) for a client. 
Your task is to assess the headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Headline Extraction and Context**
  **Image/Video:**
    1. **Headline Identification:**
        * **Main Headline:** Clearly state the main headline extracted from the image or video.
        * **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
**Part 2: Headline Analysis**
Analyze the extracted headline(s) and present the results in a well-formatted table:

| Criterion             | Main Headline Assessment                                    | Image Headline Assessment (if applicable)                                  | Explanation |
|-----------------------|-------------------------------------------------------------|----------------------------------------------------------------------------|-------------|
| Word Count            | _[Number of words in the headline]_                         | _[Number of words in the headline]_                                        | _Example: The headline has X words, which is [concise/lengthy/appropriate] for a [type of asset]._ |
| Keyword Relevance     | _[Assessment of how well the headline incorporates relevant keywords or phrases]_ | _[Assessment of how well the headline incorporates relevant keywords or phrases]_ | _Example: The headline includes [X number] of relevant keywords, such as "[keyword 1]", "[keyword 2]", etc. This may help with [SEO/search engine visibility/targeting a specific audience]._ |
| Common Words          | _[Number of common words]_                                  | _[Number of common words]_                                                 | _Example: The headline uses X common words, which may [enhance readability/make it less memorable/be appropriate for the target audience]._ |
| Uncommon Words        | _[Number of uncommon words]_                                | _[Number of uncommon words]_                                               | _Example: The headline includes X uncommon words, which can [make it more memorable/potentially confuse some readers/be appropriate for a niche audience]._ |
| Emotional Words       | _[Number of words conveying emotion (positive, negative, neutral)]_ | _[Number of words conveying emotion (positive, negative, neutral)]_        | _Example: The headline contains X emotional words, indicating a [positive/negative/neutral] tone. This may [elicit a strong response/be appropriate for the context/be overly dramatic]._ |
| Power Words           | _[Number of words with strong persuasive potential]_        | _[Number of words with strong persuasive potential]_                       | _Example: The headline includes X power words, which are designed to [grab attention/persuade/create urgency]. However, overuse can make the headline seem [pushy/manipulative/less authentic]._ |
| Sentiment             | _[Overall sentiment: positive, negative, or neutral]_       | _[Overall sentiment: positive, negative, or neutral]_                      | _Example: The headline has a [positive/negative/neutral] sentiment, which [aligns well with the message/could be adjusted for better impact]._ |
| Reading Grade Level   | _[Estimated grade level required to understand the headline]_ | _[Estimated grade level required to understand the headline]_              | _Example: The headline is written at a [grade level] reading level, making it [easy/challenging] for the target audience to comprehend. This is [appropriate/inappropriate] considering their [assumed education level/interests/background]._ |
| Clarity & Conciseness | _[Assessment of how clear and easy to understand the headline is]_ | _[Assessment of how clear and easy to understand the headline is]_         | _Example: The headline is [clear and concise/somewhat vague/difficult to understand]. It [effectively/ineffectively] communicates the main message. The use of [specific words/phrases] contributes to its [clarity/lack of clarity]._ |
| Impact & Engagement   | _[Assessment of how well the headline grabs attention and evokes emotion]_ | _[Assessment of how well the headline grabs attention and evokes emotion]_  | _Example: The headline is [attention-grabbing/forgettable/mildly interesting]. It [does/does not] effectively create a sense of urgency or curiosity. The use of [emotional language/power words/humor] [successfully/unsuccessfully] evokes a response in the reader._ |
| Relevance             | _[Assessment of how well the headline aligns with the visual content and target audience]_ | _[Assessment of how well the headline aligns with the visual content and target audience]_ | _Example: The headline is [highly relevant/somewhat relevant/irrelevant] to the visual content and the target audience. It [successfully/partially/unsuccessfully] connects with their [interests/needs/pain points] by using [language/tone/style] that is [appropriate/inappropriate] for them._ |
| Overall Effectiveness | _[Overall rating of the headline's effectiveness based on all criteria, 1 (poor) to 5 (excellent)]_ | _[Overall rating of the headline's effectiveness based on all criteria, 1 (poor) to 5 (excellent)]_ | _Example: Considering all factors, the headline receives an overall score of X out of 5. It is [highly effective/moderately effective/ineffective] due to its [strengths/weaknesses] in [specific criteria]._ |

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

    # Streamlit UI setup
    st.title('Marketing Media Analysis AI Assistant')
    with st.sidebar:
        st.header("Options")
        basic_analysis = st.button('Basic Analysis')
        combined_analysis_V6 = st.button('Combined Detailed Marketing Analysis V6')
        text_analysis_button = st.button('Text Analysis')
        headline_analysis_button = st.button('Headline Analysis')
        detailed_headline_analysis_button = st.button('Headline Optimization Report')
        flash_analysis_button = st.button('Flash Analysis')
        custom_prompt = st.text_area("Enter your custom prompt here:")
        custom_prompt_button = st.button('Send Custom Prompt')

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

            # Basic analysis
            if basic_analysis:
                with st.spinner("Performing basic analysis..."):
                    uploaded_file.seek(0)
                    basic_analysis_result = analyze_media(uploaded_file, is_image)
                    if basic_analysis_result:
                        st.write("## Basic Analysis Results:")
                        st.json(basic_analysis_result)

            # Combined marketing analysis V6
            if combined_analysis_V6:
                with st.spinner("Performing combined marketing analysis V6..."):
                    uploaded_file.seek(0)
                    detailed_result_V6 = combined_marketing_analysis_V6(uploaded_file, is_image)
                    if detailed_result_V6:
                        st.write("## Combined Marketing Analysis V6 Results:")
                        st.markdown(detailed_result_V6)

            if text_analysis_button:
                with st.spinner("Performing text analysis..."):
                    uploaded_file.seek(0)
                    text_result = text_analysis(uploaded_file, is_image)
                    if text_result:
                        st.write("## Text Analysis Results:")
                        st.markdown(text_result)
                        
            # Headline analysis
            if headline_analysis_button:
                with st.spinner("Performing headline analysis..."):
                    uploaded_file.seek(0)
                    headline_result = headline_analysis(uploaded_file, is_image)
                    if headline_result:
                        st.write("## Headline Analysis Results:")
                        st.markdown(headline_result)

            # Further Image Headline Analysis button, shows only if image headline is found
            if 'headline_result' in st.session_state and 'Image Headline' in st.session_state.headlines:
                if st.button('Further Headline Analysis (Image Headline)'):
                    with st.spinner("Performing further headline analysis..."):
                        image_headline = st.session_state.headlines['Image Headline']
                        further_prompt = f"""Imagine you are a marketing consultant reviewing a visual asset (image or video) and its headline(s) for a client. Your goal is to provide a comprehensive analysis of the headline's effectiveness across various key criteria.
                        1. Image Headline Evaluation (Image Headline Only: '{image_headline}'):
                           - Evaluate the Image headline against the following criteria, rating each on a scale from 1 to 5 (1 being poor, 5 being excellent), and provide a concise explanation for each score:
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
                           - Display the '{image_headline}' Image headline's evaluation in a table format with columns: Criterion, Score, Explanation, and Improvements. Ensure every cell in the table is filled, and mainly not any of the Improvements column should left empty or N\\A, note that.

                        5. Supporting Headline Evaluation (Optional):
                           - If applicable, briefly assess any other image headline or supporting headlines and note if they require further analysis. Consider creating a separate table if a more in-depth analysis is needed.

                        6. Total Score:
                           - Calculate and display the total score for the image headline : '{image_headline}' based on the evaluations.

                        7. Improved Headlines:
                           - Provide three alternative headlines for the image headline : '{image_headline}' , that addresses any weaknesses identified. Ensure these headlines are free of colons, diverse in structure and style, and aligned with the visual content and the target audience.

                        Note: If analyzing a video, mention any notable changes in headlines or messaging throughout the video.
                        """                       
                        further_response = model.generate_content([further_prompt, image])  # Assuming 'image' is correctly scoped
                        if further_response.candidates:
                            further_raw_response = further_response.candidates[0].content.parts[0].text.strip()
                            st.write("Further Headline Analysis Results (Image Headline):")
                            st.markdown(further_raw_response, unsafe_allow_html=True)
                        else:
                            st.error("Unexpected response structure from the model.")

            # Headline optimization report
            if detailed_headline_analysis_button:
                with st.spinner("Performing Headline Optimization Report analysis..."):
                    uploaded_file.seek(0)
                    detailed_headline_result = headline_detailed_analysis(uploaded_file, is_image)
                    if detailed_headline_result:
                        st.write("## Headline Optimization Report Results:")
                        st.markdown(detailed_headline_result)

            # Flash analysis
            if flash_analysis_button:
                with st.spinner("Performing Flash analysis..."):
                    uploaded_file.seek(0)
                    flash_result = flash_analysis(uploaded_file, is_image)
                    if flash_result:
                        st.write("## Flash Analysis Results:")
                        st.markdown(flash_result)

            # Custom prompt analysis
            if custom_prompt_button:
                with st.spinner("Performing custom prompt analysis..."):
                    uploaded_file.seek(0)
                    custom_result = custom_prompt_analysis(uploaded_file, custom_prompt, is_image)
                    if custom_result:
                        st.write("## Custom Prompt Analysis Results:")
                        st.markdown(custom_result)
