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
        "temperature": 0.1,
        "top_p": 0.8,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
      model_name="gemini-2.0-flash",
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
Analyze the provided image for marketing effectiveness. Provide detailed responses for the following initial questions:

**1. Asset Type:**
   - Clearly identify and describe the type of marketing asset (e.g., email, social media post, advertisement, flyer, brochure, landing page).

**2. Purpose:**
   - Clearly state the specific purpose of this marketing asset (e.g., selling a product, getting more signups, driving traffic to a webpage, increasing brand awareness, engaging with customers).
   - Provide a detailed explanation of *how* it aims to achieve this purpose.

**3. Asset Audience:**
   - Identify the target audience for this marketing asset (e.g., age group, gender, location, income level, education, interests, behaviors).
   - Describe the demographics, interests, and needs of this audience.

---

**Aspect-Based Evaluation:**

For each aspect listed below, provide a score from 1 to 5 in increments of 0.5 (where 1 is low and 5 is high).  Present your evaluation in a table format with the following columns: **Aspect, Score, Explanation, Improvement**.

After the table, provide a concise paragraph with **Overall Improvement Suggestions**.

**Evaluation Aspects:**

1. **Distinction:** Does the content use visually arresting pictures to grab user attention? Does it appeal to the primal brain both visually and with accompanying text?

2. **Attention Flow:** Analyze the order in which a user would consume the content.
   -  Identify and analyze the **headline's prominence and position**.
   -  Evaluate **additional text** for visibility and reading sequence.
   -  Assess the **positioning of images** relative to the text.
   -  Examine **interactive elements** (e.g., buttons).
   -  Describe the **content consumption order** (e.g., headline -> text -> image -> button).
   -  Determine if the content **prioritizes important information** and effectively draws and holds attention.

3. **Purpose and Value (3-Second Test):** Is the purpose and value proposition immediately clear (within 3 seconds)? Is the content primarily product-centric or customer-centric?

4. **Clarity of Design:** Evaluate the clarity of all design elements. Are the visuals and text easily understandable at a glance?

5. **Creativity Score:** Assess the design's creativity and originality. How effectively does it stand out and capture attention through innovative design elements?

6. **First Impression:** Analyze the initial impact of the design. Does it create a strong and positive first impression on the viewer?

7. **Headline Review (Comprehensive):** Evaluate the headline based on the following criteria:
    - Clarity
    - Conciseness
    - Customer Centricity
    - SEO Keyword Integration
    - Emotional Appeal
    - Uniqueness
    - Urgency
    - Benefit to the Reader
    - Audience Targeting
    - Length
    - Use of Numbers/Lists
    - Brand Consistency
    - Power Words

8. **Headline Keywords & Emotional Appeal (Focused):** Does the headline effectively incorporate relevant keywords and evoke an emotional response in the target audience?

9. **Visual Cues & Color Usage:** Analyze the use of visual cues and color.
    - Does the image effectively use visual cues (e.g., arrows, frames) to guide attention to key elements?
    - How effectively do color choices and contrast draw the viewer's attention and highlight important information?

10. **Engagement (UX):** Assess the overall user engagement level of the design and user experience. Is the UX design captivating, intuitive, and satisfying to interact with?

11. **Trust & Credibility:** Assess the trustworthiness conveyed by the content.
    - Is the content brand-centric or customer-centric (customer-centric content generally fosters higher trust)?
    - Evaluate the credibility, reliability, and intimacy projected by the visual and textual elements.

12. **Motivation & Authority/Social Proof:**  Assess the design's ability to motivate users to action. Does it align with user motivations? Does it effectively leverage authority or social proof elements to encourage action?

13. **Influence & Persuasion:** Analyze the design's overall influence. How effectively does the asset persuade viewers and guide them towards the desired action or conversion?

14. **Calls to Action (CTAs):** Analyze the CTAs.
    - Presence: Are CTAs clearly present?
    - Prominence: How prominent are they in the design?
    - Benefits: Are the benefits of clicking the CTA clear?
    - Language: Is the language used in CTAs compelling and action-oriented?

15. **Overall User Experience (Holistic):** Assess the overall user experience. How effectively does the design facilitate a smooth, enjoyable, and efficient interaction from the user's perspective?

16. **Memorability:** Evaluate the design's memorability. How likely is it to leave a lasting impression on the viewer after they have seen it?

17. **Effort (Text Clarity):** Evaluate the clarity and conciseness of the text. (Scale: 1 = Very Dense & Difficult to Understand, 5 = Clear & Easy to Understand).  Does it convey the message effectively without being overly verbose or dense?

18. **Tone Effectiveness:** Evaluate the effectiveness of the tone used in the asset in enhancing its overall impact and persuasiveness.

19. **Framing Effectiveness:** Evaluate the effectiveness of the message framing used in the asset in enhancing its overall impact and persuasiveness.

20. **Content Investment (Conciseness):**  Evaluate the amount of text content. Is the amount of content presented concise and easily digestible, avoiding lengthy paragraphs that busy users might ignore? (Consider that paragraph blocks are generally negative for content consumption).

---

**Table Format Example:**

| Aspect                     | Score | Explanation                                                                 | Improvement Suggestions                                                       |
|-----------------------------|-------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Distinction                | 3.5   | [Concise explanation of the aspect's performance in the image]             | [Specific, actionable suggestions for improvement]                           |
| Attention Flow             | 4     | [Concise explanation of the aspect's performance in the image]             | [Specific, actionable suggestions for improvement]                           |
| ... (and so on for all aspects) | ...   | ...                                                                       | ...                                                                           |

---

**Overall Improvement Suggestions:**

[Concise paragraph summarizing the key areas for overall improvement and offering strategic suggestions to enhance the marketing asset's effectiveness.]
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

    def Story_Telling_Analysis(uploaded_file, is_image=True):
        prompt = """
Storytelling significantly impacts creative content, enriching it and enhancing its effectiveness. Evaluate the provided image based on the following **7 Storytelling Principles** in static creative.

**For each principle, you will provide:**
* **A Score (1-5):** In increments of 0.5, where 1 is low and 5 is high.
* **An Evaluation:** A concise explanation of how the image performs against the principle.
* **Improvement Suggestions:** Specific, actionable suggestions to enhance the image based on the principle.

Present your analysis in a table format as described below. After the table, provide a summary of your overall recommendations.

---

**7 Storytelling Principles for Evaluation:**

**1. Emotional Engagement:**
    * **Impact:** Evokes emotions, making content relatable and memorable.
    * **Explanation:** Connects with the audience emotionally (empathy, joy, sadness, excitement), increasing impact.
    * **Example:** Family enjoying a product = togetherness, happiness.

**2. Attention and Interest:**
    * **Impact:** Captures and holds audience attention.
    * **Explanation:** Narrative element intrigues viewers, increasing engagement time.
    * **Example:** Before-and-after product transformation = change, improvement, interest.

**3. Memorability:**
    * **Impact:** Enhances recall and retention.
    * **Explanation:** Information in stories is easier to remember than standalone facts.
    * **Example:** Image series of product journey = brand story embedded.

**4. Brand Identity and Values:**
    * **Impact:** Conveys brand identity and values.
    * **Explanation:** Stories express mission, vision, core values, differentiating brand and building loyalty.
    * **Example:** Founders working passionately = dedication, authenticity.

**5. Simplification of Complex Messages:**
    * **Impact:** Simplifies complex messages.
    * **Explanation:** Complex information becomes understandable through storytelling.
    * **Example:** Infographic story on climate change impact = simplified complex issue.

**6. Connection and Trust:**
    * **Impact:** Builds connection and trust.
    * **Explanation:** Authentic stories foster trust and connection; relatable stories increase brand trust.
    * **Example:** Customer testimonials = credibility, trust.

**7. Call to Action (CTA) Effectiveness:**
    * **Impact:** Enhances CTA effectiveness.
    * **Explanation:** Compelling stories make CTAs more appealing and urgent within the narrative context.
    * **Example:** Story of goal achievement with product + "Join the success" CTA = persuasive.

---

**Response Table Format:**

| Element                  | Score (1-5) | Evaluation                                          | How it could be improved                                  |
|--------------------------|-------------|------------------------------------------------------|----------------------------------------------------------|
| Emotional Engagement     |             | [Your evaluation for Emotional Engagement]          | [Your improvement suggestions for Emotional Engagement] |
| Attention and Interest   |             | [Your evaluation for Attention and Interest]        | [Your improvement suggestions for Attention and Interest] |
| Memorability             |             | [Your evaluation for Memorability]                  | [Your improvement suggestions for Memorability]         |
| Brand Identity & Values  |             | [Your evaluation for Brand Identity & Values]       | [Your improvement suggestions for Brand Identity & Values]|
| Simplification           |             | [Your evaluation for Simplification]                | [Your improvement suggestions for Simplification]       |
| Connection and Trust     |             | [Your evaluation for Connection and Trust]          | [Your improvement suggestions for Connection and Trust] |
| CTA Effectiveness        |             | [Your evaluation for CTA Effectiveness]             | [Your improvement suggestions for CTA Effectiveness]    |

---

**Summary of Recommendations:**

[Provide a concise summary highlighting the key areas for overall improvement based on your table evaluations to enhance the image's storytelling effectiveness.]
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
                st.write("Story Telling Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
    def emotional_resonance(uploaded_file, is_image=True):
            prompt = """
If the content is non-English, first translate it into English. Then, using the model below, evaluate the content and suggest improvements.

Instructions:

Translation Check:

If the content is non-English, translate it to English.
Evaluation Process:
For each key criterion, score the element from 1 to 5 (in increments of 0.5). Present your findings in a table with the following columns: Element, Score, Evaluation, How It Could Be Improved.

Criteria for Evaluation:

Clarity of Emotional Appeal

Criteria: The content clearly conveys the intended emotion(s).
Evaluation: Determine if the emotional message is easily understood without ambiguity.
Relevance to Target Audience

Criteria: The emotional appeal is relevant to the target audience’s experiences, values, and interests.
Evaluation: Assess if the content connects with the audience’s personal or professional life.
Authenticity

Criteria: The emotional appeal feels genuine and credible.
Evaluation: Check if the content avoids exaggeration and resonates as sincere and trustworthy.
Visual and Verbal Consistency

Criteria: Visual elements (images, colors, design) and verbal elements (language, tone) consistently support the emotional appeal.
Evaluation: Ensure that all elements of the content align to reinforce the intended emotion.
Emotional Intensity

Criteria: The strength of the emotional response elicited is appropriate for the context.
Evaluation: Measure whether the content evokes a strong enough emotional reaction without being overwhelming or underwhelming.
Engagement

Criteria: The content encourages audience engagement (likes, shares, comments, etc.).
Evaluation: Assess whether the content explicitly encourages engagement and includes means for users to share, like, or comment.
Final Recommendations:

After the table, provide a summary of overall recommendations and suggestions for improvement.
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
                    st.write("Emotional Resonance Analysis Results:")
                    st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
                else:
                    st.error("Unexpected response structure from the model.")
                return None
            except Exception as e:
                st.error(f"Failed to read or process the media: {e}")
                return None
    def emotional_analysis(uploaded_file, is_image=True):
            prompt = """
Using the following list of emotional resonance principles, assess whether the marketing content applies each principle. Present your findings in a table with the columns: Name, Applies (None, Some, A Lot), Definition, How It Is Applied, How It Could Be Implemented.

Emotional Resonance Principles to Assess:

Empathy

Definition: The ability to understand and share the feelings of others.
Application: Craft messages that show an understanding of the audience's challenges and emotions.
Joy

Definition: A feeling of great pleasure and happiness.
Application: Create content that makes the audience feel happy, excited, or entertained.
Surprise

Definition: A feeling of astonishment or shock caused by something unexpected.
Application: Use unexpected elements in marketing to capture attention and engage the audience.
Trust

Definition: Confidence in the honesty, integrity, and reliability of someone or something.
Application: Build trust through transparent communication, endorsements, and reliable information.
Fear

Definition: An unpleasant emotion caused by the belief that someone or something is dangerous.
Application: Highlight potential risks or losses to motivate the audience to take action.
Sadness

Definition: A feeling of sorrow or unhappiness.
Application: Use stories or scenarios that evoke sympathy and compassion to drive support for a cause or product.
Anger

Definition: A strong feeling of displeasure or hostility.
Application: Address injustices or problems that provoke a sense of outrage, motivating the audience to seek solutions.
Anticipation

Definition: Excitement or anxiety about a future event.
Application: Create a sense of excitement and eagerness for upcoming products, events, or announcements.
Disgust

Definition: A strong feeling of aversion or repulsion.
Application: Highlight negative aspects of a competing product or undesirable conditions to steer the audience toward a better alternative.
Relief

Definition: A feeling of reassurance and relaxation following release from anxiety or distress.
Application: Position a product or service as a solution that alleviates worries or problems.
Love

Definition: A deep feeling of affection, attachment, or devotion.
Application: Create campaigns that evoke feelings of love and affection toward family, friends, or the brand itself.
Pride

Definition: A feeling of deep pleasure or satisfaction derived from one's own achievements.
Application: Celebrate customer achievements and successes, making them feel proud of their association with the brand.
Belonging

Definition: The feeling of being accepted and included.
Application: Create communities and foster a sense of belonging among customers.
Nostalgia

Definition: A sentimental longing for the past.
Application: Use themes and imagery that evoke fond memories and a sense of nostalgia.
Hope

Definition: A feeling of expectation and desire for a particular thing to happen.
Application: Inspire hope and optimism about the future through positive and uplifting messages.
Instructions:

Analyze each principle in relation to the marketing content.
Indicate whether the content applies each principle as None, Some, or A Lot.
Provide a concise explanation for how the principle is applied and suggestions for how it could be implemented or improved.
Ensure that your response is precise, follows the consistent scoring and table format, and focuses on the user's image analysis perspective.
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
                    st.write("Emotional Resonance Analysis Results:")
                    st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
                else:
                    st.error("Unexpected response structure from the model.")
                return None
            except Exception as e:
                st.error(f"Failed to read or process the media: {e}")
                return None
            
    def Emotional_Appraisal_Models(uploaded_file, is_image=True):
            prompt = """
Translation Check:
Firstly, translate any non-English text to English.
Evaluation Task:

Using the emotional appraisal models listed below, evaluate the content.
For each model, provide an evaluation of the content and suggest possible improvements based on the model’s components.
Emotional Appraisal Models:

a. Lazarus’ Cognitive-Motivational-Relational Theory

Overview:
Richard Lazarus proposed that emotions result from cognitive appraisals of events, considering both personal relevance and coping potential.
Components:
Primary Appraisal: Evaluate the significance of an event for personal well-being (e.g., Is this event beneficial or harmful?).
Secondary Appraisal: Assess one’s ability to cope with the event (e.g., Do I have the resources to deal with this?).
Core Relational Themes: Identify patterns of appraisal that lead to specific emotions (e.g., loss leads to sadness, threat leads to fear).
b. Scherer's Component Process Model (CPM)

Overview:
Klaus Scherer's model posits that emotions result from a sequence of appraisals along several dimensions.
Components:
Novelty: Is the event new or unexpected?
Pleasantness: Is the event pleasant or unpleasant?
Goal Significance: Does the event help or hinder the attainment of goals?
Coping Potential: Can the individual cope with or manage the event?
Norm Compatibility: Does the event conform to social and personal norms?
c. Smith and Ellsworth’s Appraisal Model

Overview:
This model identifies several dimensions of appraisal that influence emotional responses.
Components:
Attention: The degree to which the event draws attention.
Certainty: The certainty or predictability of the event.
Control/Coping: The degree of control over the event and ability to cope.
Pleasantness: The pleasantness or unpleasantness of the event.
Perceived Obstacle: The extent to which the event is seen as an obstacle to goals.
Responsibility: Attribution of responsibility (self, others, or circumstances).
Anticipated Effort: The amount of effort required to manage the event.
d. Roseman’s Appraisal Theory

Overview:
Focuses on how appraisals in terms of motivational congruence and agency influence emotions.
Components:
Motivational State: Whether the event aligns or conflicts with one’s goals.
Situational State: Whether the event is caused by the environment or by the individual.
Probability: The likelihood of the event occurring.
Agency: Attribution of responsibility (self, other, or circumstance).
Power/Control: The degree of control over the event.
e. Weiner’s Attributional Theory of Emotion

Overview:
Focuses on how attributions about the causes of events influence emotional reactions.
Components:
Locus: Whether the cause is internal or external.
Stability: Whether the cause is stable or unstable over time.
Controllability: Whether the cause is controllable or uncontrollable by the individual.
f. Frijda’s Laws of Emotion

Overview:
Proposes several “laws” describing regularities in the relationship between appraisals and emotional responses.
Components:
Law of Situational Meaning: Emotions arise in response to the meaning structures of situations.
Law of Concern: Emotions arise when events are relevant to one’s concerns.
Law of Apparent Reality: Emotions are elicited by events appraised as real.
Law of Change: Emotions are triggered by changes in circumstances.
Law of Habituation: Continuous exposure to a stimulus reduces its emotional impact.
Law of Comparative Feeling: Emotional intensity depends on comparisons with other events.
Law of Hedonic Asymmetry: Pleasure is more transient than pain.
Law of Conservation of Emotional Momentum: Emotions persist until triggering conditions change.
g. Ellsworth’s Model of Appraisal Dimensions

Overview:
Emphasizes the importance of cultural and contextual factors in emotional appraisal.
Components:
Certainty: How certain one is about the event.
Attention: The extent to which the event captures attention.
Control: The degree of control over the event.
Pleasantness: Whether the event is perceived as pleasant or unpleasant.
Responsibility: Attribution of responsibility for the event.
Legitimacy: Whether the event is perceived as fair or unfair.
Practical Applications in Marketing:

Resonates with Core Concerns: Address primary and secondary appraisals of the target audience.
Triggers Relevant Emotions: Design messages that align with specific appraisal dimensions to evoke desired emotional responses.
Enhances Perceived Control: Empower consumers by highlighting how products or services help manage or cope with challenges.
Builds Trust and Credibility: Ensure messages are consistent, predictable, and aligned with social norms.
Instructions:

Evaluate the content using each of the emotional appraisal models outlined above.
For each model, provide:
A clear evaluation of how the content aligns (or does not align) with the model’s components.
Specific suggestions for improvements based on the evaluation.
Ensure your response is organized and structured clearly, maintaining consistent formatting and focusing on the user’s image analysis perspective.
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
                    st.write("Emotional Appraisal Mode Analysis Results:")
                    st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
                else:
                    st.error("Unexpected response structure from the model.")
                return None
            except Exception as e:
                st.error(f"Failed to read or process the media: {e}")
                return None    
    def behavioural_principles(uploaded_file, is_image=True):
        prompt = """
Using the following Behavioral Science principles, assess whether the marketing content applies each principle. Present your evaluation in a table with the following columns:

Applies the Principle (None, Some, A Lot)
Principle (Description)
Explanation
How it could be implemented
Behavioral Science Principles to Assess:

Anchoring

Description: The tendency to rely heavily on the first piece of information encountered (the "anchor") when making decisions.
Example: Displaying a higher original price next to a discounted price to make the discount seem more substantial.
Social Proof

Description: People tend to follow the actions of others, assuming that those actions are correct.
Example: Showing customer reviews and testimonials to build trust and encourage purchases.
Scarcity

Description: Items or opportunities become more desirable when they are perceived to be scarce or limited.
Example: Using phrases like "limited time offer" or "only a few left in stock" to create urgency.
Reciprocity

Description: People feel obligated to return favors or kindnesses received from others.
Example: Offering a free sample or trial to encourage future purchases.
Loss Aversion

Description: People prefer to avoid losses rather than acquire equivalent gains.
Example: Emphasizing what customers stand to lose if they don't take action, such as missing out on a sale.
Commitment and Consistency

Description: Once people commit to something, they are more likely to follow through to maintain consistency.
Example: Getting customers to make a small commitment first, like signing up for a newsletter, before asking for a larger commitment.
Authority

Description: People are more likely to trust and follow the advice of an authority figure.
Example: Featuring endorsements from experts or industry leaders.
Framing

Description: The way information is presented can influence decision-making.
Example: Highlighting the benefits of a product rather than the features, or framing a price as "only $1 a day" instead of "$30 a month."
Endowment Effect

Description: People value things more highly if they own them.
Example: Allowing customers to try a product at home before making a purchase decision.
Priming

Description: Exposure to certain stimuli can influence subsequent behavior and decisions.
Example: Using images and words that evoke positive emotions to enhance the appeal of a product.
Decoy Effect

Description: Adding a third option can make one of the original two options more attractive.
Example: Introducing a higher-priced premium option to make the mid-tier option seem like better value.
Default Effect

Description: People tend to go with the default option presented to them.
Example: Setting a popular product or service as the default selection on a website.
Availability Heuristic

Description: People judge the likelihood of events based on how easily examples come to mind.
Example: Highlighting popular or recent customer success stories to create a perception of common positive outcomes.
Cognitive Dissonance

Description: The discomfort experienced when holding conflicting beliefs, leading to a change in attitude or behavior to reduce discomfort.
Example: Reinforcing the positive aspects of a purchase to reduce buyer's remorse.
Emotional Appeal

Description: Emotions can significantly influence decision-making.
Example: Using storytelling and emotional imagery to create a connection with the audience.
Bandwagon Effect

Description: People are more likely to do something if they see others doing it.
Example: Showcasing the popularity of a product through sales numbers or social media mentions.
Frequency Illusion (Baader-Meinhof Phenomenon)

Description: Once people notice something, they start seeing it everywhere.
Example: Repeatedly exposing customers to a brand or product through various channels to increase recognition.
In-group Favoritism

Description: People prefer products or services associated with groups they identify with.
Example: Creating marketing campaigns that resonate with specific demographics or communities.
Hyperbolic Discounting

Description: People prefer smaller, immediate rewards over larger, delayed rewards.
Example: Offering instant discounts or rewards for immediate purchases.
Paradox of Choice

Description: Having too many options can lead to decision paralysis.
Example: Simplifying choices by offering curated selections or recommended products.
Instructions:

Evaluate each principle against the marketing content.
For each principle, indicate in the table whether the content applies the principle as None, Some, or A Lot.
In the table, include:
Principle (Description): Use the provided description.
Explanation: Explain how the principle is or is not applied in the content.
How it could be implemented: Suggest ways to implement or enhance the application of the principle.
Ensure that your final response is precise, follows the consistent table format, and focuses on the user's image analysis perspective.
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
                st.write("Behavioural Principles Result::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None

    def nlp_principles_analysis(uploaded_file, is_image=True):
        prompt = """
Using the following Neuro-Linguistic Programming (NLP) techniques, assess whether the marketing content applies each principle. Present your findings in a table with the following columns:

Applies the Principle (None, Some, A Lot)
Principle (Description)
Explanation
How it could be implemented
NLP Techniques to Assess:

Representational Systems
Example: If your target audience prefers visual information, ensure the content includes vivid images and visually appealing graphics.

Anchoring
Example: Use consistent colors and logos to create positive associations with your brand every time the audience sees them.

Meta-Modeling
Example: Clarify ambiguous statements like "Our product is the best" by specifying "Our product is rated #1 for quality by Consumer Reports."

Milton Model
Example: Use phrases like "You may find yourself feeling more relaxed when using our product" to embed suggestions subtly.

Chunking
Example: Provide both high-level benefits (chunking up) and detailed features (chunking down) of your product to cater to different audience needs.

Pacing and Leading
Example: Start with a relatable problem (pacing) like "Do you struggle with time management?" and lead to your solution: "Our planner can help you stay organized and efficient."

Swish Pattern
Example: Replace negative images (e.g., cluttered desk) with positive images (e.g., clean, organized workspace) in your content.

Submodalities
Example: Use bright, bold colors for calls to action to evoke excitement and urgency.

Perceptual Positions
Example: Present content from the user's perspective ("You will benefit from..."), from others' perspectives ("Others will admire your..."), and from an observer's perspective ("Imagine the positive impact...").

Well-Formed Outcomes
Example: Clearly state the desired outcome: "Increase your productivity by 20% with our planner in just one month."

Rapport Building
Example: Use language that resonates with your audience’s values and experiences: "We understand how hectic life can be, and we’re here to help."

Calibration
Example: Monitor engagement metrics like click-through rates and adjust content accordingly to better meet audience needs.

Reframing
Example: Turn a negative situation into a positive opportunity: "Stuck in traffic? Use this time to listen to our educational podcasts and learn something new."

Logical Levels
Example: Ensure your content addresses different levels, from the environment ("Work anywhere") to identity ("Be a proactive leader").

Timeline Therapy
Example: Highlight past successes, current benefits, and future potential: "Our product has helped thousands, it’s helping people right now, and it can help you too."

Meta Programs
Example: Tailor content to different motivational patterns, such as "towards" goals ("Achieve your dreams with our help") or "away from" problems ("Avoid stress with our solution").

Strategy Elicitation
Example: Show step-by-step how to use your product to achieve desired results, aligning with the audience's decision-making strategies.

Sensory Acuity
Example: Use descriptive language that appeals to the senses: "Feel the soft texture, see the vibrant colors, and hear the clear sound."

Pattern Interrupts
Example: Include unexpected elements like surprising statistics or bold images to capture attention and break habitual thought patterns.

Belief Change Techniques
Example: Challenge limiting beliefs with testimonials or case studies that show successful outcomes, shifting beliefs towards the positive.

Instructions:

Evaluate each NLP technique against the marketing content.
Indicate in the table whether the content applies the technique as None, Some, or A Lot.
Provide a brief explanation of your evaluation.
Suggest specific ways to implement or enhance the technique in the content.
Ensure that your final response is precise, uses the consistent table format, and focuses on the user's image analysis perspective.
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
                st.write("NLP Principles Result::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
            
    def text_analysis(uploaded_file, is_image=True):
        prompt = """
As a UX design and marketing analysis consultant, you are tasked with reviewing the text content of a marketing asset (image or video, excluding the headline) for a client. Your goal is to provide a comprehensive analysis of the text's effectiveness and offer actionable recommendations for improvement. **Important:** All analysis and recommendations must be provided in English, regardless of the language used in the original marketing asset.
---
### **Part 1: Text Extraction and Contextualization**

**A. Image Analysis**

1. **Text Extraction:**
   - Identify and extract **ALL** visible text from the image, including headlines, body copy, captions, calls to action, taglines, logos, and any other textual elements.
   - Translate any non-English text into English.

2. **Presentation:**
   - Present the extracted text in a clear, bulleted list format, preserving the original order and structure as much as possible.

3. **Visual Analysis:**
   - **Placement:** Specify the location of each text element (e.g., top left, centered, bottom right). Note any overlapping text or elements that may hinder readability.
   - **Font Choices:** Describe the font style (e.g., serif, sans-serif, script), weight (bold, regular, light), size, and color for each distinct text element.
   - **Visual Relationships:** Explain how the text interacts with other visual elements (e.g., images, graphics, colors) and how it contributes to the overall message and hierarchy of information.

**B. Video Analysis**

1. **Key Frame Identification:**
   - Select the key frame(s) that best showcase the primary text content.

2. **Text Extraction:**
   - Extract and present the text from these key frames in a clear, bulleted list format.

3. **Temporal Analysis:**
   - Briefly describe any significant textual changes or patterns observed throughout the video.

4. **Integration with Visuals and Audio:**
   - Analyze how the text interacts with the video's visuals (e.g., scenes, characters, actions) and audio (e.g., dialogue, music, sound effects).

---

### **Part 2: Textual Assessment**

Evaluate the extracted text based on the criteria below. For each aspect, provide:

- A **score** from **1 (poor)** to **5 (excellent)** in increments of **0.5**.
- A concise **justification** of the score, highlighting strengths and weaknesses.
- Specific, **actionable suggestions** for enhancing the text's effectiveness.

Present your assessment in a table with the following columns:

| Aspect                       | Score | Explanation                                                                                                | Improvement                                                                                      |
|------------------------------|-------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Clarity and Conciseness      |       | Assess how easy it is to understand the text, considering sentence structure, vocabulary, and flow.        | Suggest ways to simplify language, eliminate jargon, or shorten sentences.                       |
| Customer Focus               |       | Evaluate if the text addresses customer needs and resonates with the target audience.                      | Offer suggestions to better incorporate the customer's perspective.                             |
| Engagement                   |       | Assess how compelling the text is, including storytelling, humor, and value proposition.                   | Propose methods to enhance engagement (e.g., using stronger verbs, improved formatting).          |
| Reading Effort               |       | Evaluate the ease of reading and comprehension, considering vocabulary and sentence structure.             | Suggest using simpler sentence structures and more accessible vocabulary.                       |
| Purpose and Value            |       | Determine if the text's purpose and value proposition are clear and compelling.                          | Recommend clarifying the key message or benefits more directly.                                 |
| Motivation & Persuasion      |       | Analyze the persuasive power of the text, including calls to action and social proof.                      | Suggest strengthening persuasive elements, such as stronger calls to action.                     |
| Depth and Detail             |       | Evaluate whether the text provides sufficient information and detail for the target audience.              | Suggest adding or condensing information as necessary.                                          |
| Trustworthiness              |       | Assess the credibility of the text and its ability to build trust with the audience.                       | Propose ways to enhance trustworthiness (e.g., using transparent language).                      |
| Memorability                 |       | Evaluate if the text includes memorable elements such as catchy phrases or unique storytelling techniques.  | Recommend incorporating memorable language or anecdotes for better retention.                   |
| Emotional Appeal             |       | Determine if the text evokes appropriate emotions that align with the brand image and message.             | Suggest using language that evokes specific emotions to strengthen the emotional impact.         |
| Uniqueness & Differentiation |       | Analyze whether the text differentiates the brand from competitors effectively.                          | Recommend ways to enhance uniqueness, such as developing a stronger brand voice.                 |
| Urgency and Curiosity        |       | Assess if the text creates a sense of urgency or curiosity, motivating the audience to learn more.         | Propose methods to increase urgency (e.g., highlighting limited-time offers).                    |
| Benefit Orientation          |       | Evaluate if the text clearly articulates the benefits of the product/service to the target audience.       | Suggest making the benefits more explicit and customer-centric.                                 |
| Target Audience Relevance    |       | Determine if the text's language, tone, and style are appropriate and appealing to the intended audience.  | Recommend adjustments to better align with the audience's interests and needs.                   |

---

**Final Instructions:**

- Ensure your final response is entirely in English.
- Follow the table format precisely and provide clear, actionable recommendations.
- Focus on the user's image analysis perspective throughout your analysis.

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

            if response.candidates and response.candidates[0].content.parts:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Text Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("No candidates returned from the model or the response structure is unexpected.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media. Error details: {e}")
            st.error(f"Response from model: {response}")  # Log the response from the model if possible
    def Text_Analysis_2(uploaded_file, is_image=True):
        prompt = """
If the content is non-English, translate it into English. Please evaluate the image based on the following principles:
---
### **1. Textual Analysis**

- **Readability Analysis:**  
  Use tools like the Flesch-Kincaid readability tests to determine how easy the content is to read. This ensures that the language is appropriate for the target audience.

- **Lexical Diversity:**  
  Analyze the variety of words used in the content. High lexical diversity can indicate rich, engaging language, while lower diversity might result in simpler, clearer text.

---

### **2. Semantic Analysis**

- **Keyword Analysis:**  
  Evaluate the frequency and placement of key terms related to the brand or product. Ensure that the most important keywords are prominently featured and well-integrated.

- **Topic Modeling:**  
  Use techniques such as Latent Dirichlet Allocation (LDA) to identify the main topics covered in the content. This helps verify that the content aligns with the intended message and themes.

---

### **3. Sentiment Analysis**

- **Polarity Assessment:**  
  Utilize NLP tools to analyze the sentiment of the content, categorizing it as positive, negative, or neutral. This helps ensure the tone matches the intended emotional impact.

- **Emotion Detection:**  
  Apply advanced NLP tools to detect specific emotions (e.g., joy, anger, sadness) conveyed by the content, beyond simple sentiment.

---

### **4. Structural Analysis**

- **Narrative Structure:**  
  Examine the structure of the content to ensure it follows a logical flow (e.g., introduction, problem statement, solution, and conclusion).

- **Visual Composition Analysis:**  
  For visual marketing content, evaluate the layout, use of colors, fonts, and imagery. Ensure these elements align with branding guidelines and are aesthetically pleasing.

---

### **5. Linguistic Style Matching**

- **Consistency with Brand Voice:**  
  Analyze whether the content maintains consistency with the established brand voice and style guidelines (tone, style, and terminology).

- **Grammar and Syntax Analysis:**  
  Use grammar-checking tools to ensure the content is free from grammatical errors and awkward phrasing.

---

### **6. Cohesion and Coherence Analysis**

- **Cohesion Metrics:**  
  Measure how well different parts of the text link together. Tools like Coh-Metrix can provide insights into the coherence of the content.

- **Logical Flow:**  
  Evaluate the logical progression of ideas to ensure the content flows smoothly and makes sense from start to finish.

---

### **7. Visual and Multimodal Analysis**

- **Image and Text Alignment:**  
  Analyze the relationship between text and images in the content. Ensure that images support and enhance the message conveyed by the text.

- **Aesthetic Quality:**  
  Evaluate the aesthetic elements of the visual content, considering aspects such as balance, symmetry, color harmony, and typography.

---

### **8. Compliance and Ethical Analysis**
- **Regulatory Compliance:**  
  Verify that the content complies with advertising regulations and industry standards.
- **Ethical Considerations:**  
  Analyze the content for any potential ethical issues, such as misleading claims, cultural insensitivity, or inappropriate content.
---
**Final Instructions:**
- Ensure all responses are provided in English.
- Focus your evaluation on the image analysis perspective.
- Present your analysis in a clear, consistent format.
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
                st.write("Text Analysis 2 Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
    def Text_Analysis_2_table(uploaded_file, is_image=True):
        prompt = """
If the content is non-English, translate it to English. Please evaluate the image against the following principles in a table. For each element and sub-element, provide:

- A **score** (from 1 to 5, in increments of 0.5)
- An **analysis**
- **Recommendations**

Present your evaluation in a table with the following columns:

- **Element (and Sub-Element)**
- **Score**
- **Analysis**
- **Recommendations**

---

### **Principles to Evaluate:**

1. **Textual Analysis**
   - **Readability Analysis:**  
     Use tools like the Flesch-Kincaid readability tests to determine how easy the content is to read. This ensures that the language is appropriate for the target audience.
   - **Lexical Diversity:**  
     Analyze the variety of words used. High lexical diversity can indicate rich, engaging language, while lower diversity might result in simpler, clearer content.

2. **Semantic Analysis**
   - **Keyword Analysis:**  
     Evaluate the frequency and placement of key terms related to the brand or product. Ensure that the most important keywords are prominently featured and well-integrated.
   - **Topic Modeling:**  
     Use techniques like Latent Dirichlet Allocation (LDA) to identify the main topics covered. This helps determine if the content aligns with the intended message and themes.

3. **Sentiment Analysis**
   - **Polarity Assessment:**  
     Use NLP tools to analyze the sentiment of the content, categorizing it as positive, negative, or neutral. This ensures that the tone matches the intended emotional impact.
   - **Emotion Detection:**  
     Use advanced NLP tools to detect specific emotions (e.g., joy, anger, sadness) conveyed by the content.

4. **Structural Analysis**
   - **Narrative Structure:**  
     Examine the content’s structure to ensure it follows a logical flow (e.g., introduction, problem statement, solution, conclusion).
   - **Visual Composition Analysis:**  
     For visual marketing content, analyze the layout, use of colors, fonts, and imagery. Ensure these elements are aligned with branding guidelines and are aesthetically pleasing.

5. **Linguistic Style Matching**
   - **Consistency with Brand Voice:**  
     Analyze whether the content maintains consistency with the established brand voice and style guidelines (tone, style, and terminology).
   - **Grammar and Syntax Analysis:**  
     Use grammar-checking tools to ensure the content is free from grammatical errors and awkward phrasing.

6. **Cohesion and Coherence Analysis**
   - **Cohesion Metrics:**  
     Measure how well different parts of the text link together. Tools like Coh-Metrix can provide insights into the coherence of the content.
   - **Logical Flow:**  
     Evaluate the logical progression of ideas to ensure the content flows smoothly from start to finish.

7. **Visual and Multimodal Analysis**
   - **Image and Text Alignment:**  
     Analyze the relationship between text and images. Ensure that images support and enhance the message conveyed by the text.
   - **Aesthetic Quality:**  
     Evaluate the aesthetic elements (balance, symmetry, color harmony, typography) of the visual content.

8. **Compliance and Ethical Analysis**
   - **Regulatory Compliance:**  
     Ensure the content complies with advertising regulations and industry standards.
   - **Ethical Considerations:**  
     Analyze the content for potential ethical issues (e.g., misleading claims, cultural insensitivity, or inappropriate content).
---

**Final Instructions:**

- Ensure that all responses are provided in English.
- Focus on the user's image analysis perspective.
- Provide clear, detailed responses with consistent scoring and formatting.

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
                st.write("ext Analysis 2 - table Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None        
    def headline_analysis(uploaded_file, is_image=True):
        prompt = f"""
Imagine you are a marketing consultant reviewing the headline text of a marketing asset (either an image or a video) for a client. Your task is to assess the effectiveness of the headlines based on various linguistic and marketing criteria.

---

### **Part 1: Headline Extraction and Context**

**Image/Video:**

1. **Headline Identification:**
   - **Main Headline:** Clearly state the main headline extracted from the image or video.
   - **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
   - **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.

---
### **Part 2: Headline Analysis**

For each headline type (Main, Image, and Supporting), perform an analysis by rating each criterion on a scale from **1 to 5** in increments of **0.5** (1 = poor, 5 = excellent). Provide an explanation for each score based on the synergy between the image and the headline, along with a recommendation for improvement.

Present your results in a table with the following columns: **Criterion**, **Score**, **Explanation**, **Recommendation**.

---
#### **Part 2A: Main Headline Analysis**
**Criteria:**
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

---
#### **Part 2B: Image Headline Analysis**

Analyze the image headline text using the same criteria as in Part 2A:
1. **Overall Effectiveness**
2. **Clarity**
3. **Customer Focus**
4. **Relevance**
5. **Keywords**
6. **Emotional Appeal**
7. **Uniqueness**
8. **Urgency & Curiosity**
9. **Benefit-Driven**
10. **Target Audience**
11. **Length & Format**
---
#### **Part 2C: Supporting Headline Analysis**

Analyze the supporting headline text using the same criteria:
1. **Overall Effectiveness**
2. **Clarity**
3. **Customer Focus**
4. **Relevance**
5. **Keywords**
6. **Emotional Appeal**
7. **Uniqueness**
8. **Urgency & Curiosity**
9. **Benefit-Driven**
10. **Target Audience**
11. **Length & Format**
---

### **Part 3: Improved Headline Suggestions**

Provide three improved headlines for **each** headline type (Main, Image, and Supporting) that better align with the image content. For each suggestion, explain why you have selected it.

Present your results in a table with the following columns:
- **Headline Type (Main/Image/Supporting)**
- **Headline Recommendation**
- **Explanation**

**Note:** This table must contain 9 rows in total (3 rows per headline type).
---

**Final Instructions:**

- Ensure your analysis and recommendations are provided entirely in English.
- Maintain a consistent scoring and table format throughout your response.
- Focus on the synergy between the image content and the headline(s) in your analysis.
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
        prompt = f"""
You are a marketing consultant tasked with evaluating the effectiveness of headline text from a marketing asset. Your analysis will focus on the synergy between the image content and the headline text. The evaluation is divided into three parts: Main Headline Optimization Analysis, Image Headline Optimization Analysis, and Supporting Headline Optimization Analysis.

For each part, analyze the provided image content alongside the respective headline text. Evaluate each of the criteria listed below, provide an explanation for your assessment based on the synergy between the image and the headline, and offer recommendations for improvement. Present your results in a table with the following columns: **Criterion**, **Assessment**, **Explanation**, **Recommendation**.

---

### **Part 1A: Main Headline Optimization Analysis**

Evaluate the main headline text using the following criteria:

1. **Word count:**  
   - Number of words in the headline.
2. **Keyword Relevance:**  
   - How well the headline incorporates relevant keywords or phrases.
3. **Common words:**  
   - Number of common words used.
4. **Uncommon Words:**  
   - Number of uncommon words used.
5. **Power Words:**  
   - Number of words with strong persuasive potential.
6. **Emotional words:**  
   - Number of words conveying emotion (e.g., positive, negative, neutral).
7. **Sentiment:**  
   - Overall sentiment of the headline (positive, negative, or neutral).
8. **Reading Grade Level:**  
   - Estimated grade level required to understand the headline.

---

### **Part 1B: Image Headline Optimization Analysis**

Evaluate the image headline text (if applicable) using the same criteria as in Part 1A:

1. **Word count:**  
   - Number of words in the headline.
2. **Keyword Relevance:**  
   - How well the headline incorporates relevant keywords or phrases.
3. **Common words:**  
   - Number of common words used.
4. **Uncommon Words:**  
   - Number of uncommon words used.
5. **Power Words:**  
   - Number of words with strong persuasive potential.
6. **Emotional words:**  
   - Number of words conveying emotion (e.g., positive, negative, neutral).
7. **Sentiment:**  
   - Overall sentiment of the headline (positive, negative, or neutral).
8. **Reading Grade Level:**  
   - Estimated grade level required to understand the headline.

---

### **Part 1C: Supporting Headline Optimization Analysis**

Evaluate the supporting headline text (if applicable) using the same criteria as above:

1. **Word count:**  
   - Number of words in the headline.
2. **Keyword Relevance:**  
   - How well the headline incorporates relevant keywords or phrases.
3. **Common words:**  
   - Number of common words used.
4. **Uncommon Words:**  
   - Number of uncommon words used.
5. **Power Words:**  
   - Number of words with strong persuasive potential.
6. **Emotional words:**  
   - Number of words conveying emotion (e.g., positive, negative, neutral).
7. **Sentiment:**  
   - Overall sentiment of the headline (positive, negative, or neutral).
8. **Reading Grade Level:**  
   - Estimated grade level required to understand the headline.
---
**Final Instructions:**

- Ensure your analysis is based on the synergy between the image content and the corresponding headline text.
- Present your findings in clear tables with columns labeled: **Criterion**, **Assessment**, **Explanation**, **Recommendation**.
- Maintain a consistent and detailed format throughout your response.
    
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
Imagine you are a marketing consultant reviewing the main headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the main headline's effectiveness based on various linguistic and marketing criteria.
---
### **Part 1: Headline Extraction and Context**
**Image/Video:**
1. **Headline Identification:**
   - **Main Headline:** Clearly state the main headline extracted from the image or video.
   - **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, clearly state it here.
   - **Supporting Headline (if applicable):** If there is a supporting headline present, clearly state it here.
---
### **Part 2: Headline Analysis**
Analyze the extracted Main Headline and present your evaluation in a well-formatted table. Use the following table format and criteria:
**Headline being analyzed:** [Main Headline]

| Criterion               | Score (1-5) | Explanation                                          | Main Headline Improvement                         |
|-------------------------|-------------|------------------------------------------------------|---------------------------------------------------|
| **Clarity**             | _[1-5]_    | _[Explanation for clarity of the main headline]_     | _[Suggested improvement or reason it's effective]_|
| **Customer Focus**      | _[1-5]_    | _[Explanation for customer focus of the main headline]_ | _[Suggested improvement or reason it's effective]_|
| **Relevance**           | _[1-5]_    | _[Explanation for relevance of the main headline]_   | _[Suggested improvement or reason it's effective]_|
| **Emotional Appeal**    | _[1-5]_    | _[Explanation for emotional appeal of the main headline]_ | _[Suggested improvement or reason it's effective]_|
| **Uniqueness**          | _[1-5]_    | _[Explanation for uniqueness of the main headline]_  | _[Suggested improvement or reason it's effective]_|
| **Urgency & Curiosity** | _[1-5]_    | _[Explanation for urgency & curiosity of the main headline]_ | _[Suggested improvement or reason it's effective]_|
| **Benefit-Driven**      | _[1-5]_    | _[Explanation for benefit-driven nature of the main headline]_ | _[Suggested improvement or reason it's effective]_|
| **Target Audience**     | _[1-5]_    | _[Explanation for target audience focus of the main headline]_ | _[Suggested improvement or reason it's effective]_|
| **Length & Format**     | _[1-5]_    | _[Explanation for length & format of the main headline]_ | _[Suggested improvement or reason it's effective]_|
| **Overall Effectiveness**| _[1-5]_   | _[Explanation for overall effectiveness of the main headline]_ | _[Suggested improvement or reason it's effective]_|

**Total Score:** _[Sum of all scores]_
---
### **Part 3: Improved Headline Suggestions**

Provide three alternative headlines for the main headline that better align with the image content. For each option, include a brief explanation for your suggestion.

- **Option 1:** [Headline] - [Explanation]
- **Option 2:** [Headline] - [Explanation]
- **Option 3:** [Headline] - [Explanation]
---

**Final Instructions:**

- Ensure that all responses are provided entirely in English.
- Maintain the specified table format and consistent scoring (from 1 to 5, in increments of 0.5).
- Focus your analysis on the synergy between the image content and the headline.
- Provide clear, actionable recommendations.
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
Imagine you are a marketing consultant reviewing the image headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the image headline's effectiveness based on various linguistic and marketing criteria.

---

### **Part 1: Headline Extraction and Context**

**Image/Video:**

1. **Headline Identification:**
   - **Main Headline:** Clearly state the main headline extracted from the image or video.
   - **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, state it here.
   - **Supporting Headline (if applicable):** If there is a supporting headline present, state it here.

---

### **Part 2: Headline Analysis**

Analyze the extracted **Image Headline** and present your evaluation in a well-formatted table. The headline being analyzed is: **[Image Headline]**.

Your table should include the following columns:
- **Criterion**
- **Score** (from 1 to 5, in increments of 0.5)
- **Explanation**
- **Image Headline Improvement**

The criteria to assess are:

1. **Clarity:**  
   Explain how clearly the image headline conveys its message.
2. **Customer Focus:**  
   Assess whether the headline emphasizes a customer-centric approach.
3. **Relevance:**  
   Evaluate how accurately the headline reflects the content of the image.
4. **Emotional Appeal:**  
   Determine if the headline evokes an emotional response or curiosity.
5. **Uniqueness:**  
   Analyze how original and creative the headline is.
6. **Urgency & Curiosity:**  
   Evaluate whether the headline creates a sense of urgency or piques curiosity.
7. **Benefit-Driven:**  
   Assess if the headline clearly conveys a benefit or value proposition.
8. **Target Audience:**  
   Determine if the headline is tailored to resonate with the specific target audience.
9. **Length & Format:**  
   Evaluate whether the headline’s length and format are appropriate (ideally 6-12 words).
10. **Overall Effectiveness:**  
    Summarize the overall effectiveness of the headline.

**Total Score:** _[Sum of all scores]_

---

### **Part 3: Improved Headline Suggestions**

Provide three alternative headlines for the image headline, along with a brief explanation for each suggestion. Present your results in a bulleted list:

- **Option 1:** [Headline] - [Explanation]
- **Option 2:** [Headline] - [Explanation]
- **Option 3:** [Headline] - [Explanation]

---

**Final Instructions:**

- Ensure all responses are provided in English.
- Maintain the specified table format and consistent scoring.
- Focus on the synergy between the image content and the headline.
- Provide clear, actionable recommendations for improvement.
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
Imagine you are a marketing consultant reviewing the supporting headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the supporting headline's effectiveness based on various linguistic and marketing criteria.

---

### **Part 1: Headline Extraction and Context**

**Image/Video:**

1. **Headline Identification:**
   - **Main Headline:** Clearly state the main headline extracted from the image or video.
   - **Image Headline (if applicable):** If the image contains a distinct headline separate from the main headline, state it here.
   - **Supporting Headline (if applicable):** If there is a supporting headline, state it here.

---

### **Part 2: Headline Analysis**

Analyze the extracted **Supporting Headline** and present your evaluation in a well-formatted table. The headline being analyzed is: **[Supporting Headline]**.

Your table should include the following columns:
- **Criterion**
- **Score** (from 1 to 5, in increments of 0.5)
- **Explanation**
- **Supporting Headline Improvement**

Evaluate each of the following criteria:

| Criterion               | Score    | Explanation                                                  | Supporting Headline Improvement                         |
|-------------------------|----------|--------------------------------------------------------------|---------------------------------------------------------|
| **Clarity**             | _[1-5]_ | _[Explanation for clarity of the supporting headline]_       | _[Suggested improvement or reason it's effective]_      |
| **Customer Focus**      | _[1-5]_ | _[Explanation for customer focus of the supporting headline]_  | _[Suggested improvement or reason it's effective]_      |
| **Relevance**           | _[1-5]_ | _[Explanation for relevance of the supporting headline]_     | _[Suggested improvement or reason it's effective]_      |
| **Emotional Appeal**    | _[1-5]_ | _[Explanation for emotional appeal of the supporting headline]_ | _[Suggested improvement or reason it's effective]_      |
| **Uniqueness**          | _[1-5]_ | _[Explanation for uniqueness of the supporting headline]_    | _[Suggested improvement or reason it's effective]_      |
| **Urgency & Curiosity** | _[1-5]_ | _[Explanation for urgency & curiosity of the supporting headline]_ | _[Suggested improvement or reason it's effective]_      |
| **Benefit-Driven**      | _[1-5]_ | _[Explanation for benefit-driven nature of the supporting headline]_ | _[Suggested improvement or reason it's effective]_      |
| **Target Audience**     | _[1-5]_ | _[Explanation for target audience focus of the supporting headline]_ | _[Suggested improvement or reason it's effective]_      |
| **Length & Format**     | _[1-5]_ | _[Explanation for length & format of the supporting headline]_ | _[Suggested improvement or reason it's effective]_      |
| **Overall Effectiveness**| _[1-5]_ | _[Explanation for overall effectiveness of the supporting headline]_ | _[Suggested improvement or reason it's effective]_      |

**Total Score:** _[Sum of all scores]_

---

### **Part 3: Improved Headline Suggestions**

Provide three alternative headlines for the supporting headline, along with a brief explanation for each option. Present your suggestions as follows:

- **Option 1:** [Headline] - [Explanation]
- **Option 2:** [Headline] - [Explanation]
- **Option 3:** [Headline] - [Explanation]

---

**Final Instructions:**

- Ensure all responses are in English.
- Maintain the specified table format and consistent scoring (from 1 to 5, in increments of 0.5).
- Focus on the synergy between the image content and the supporting headline.
- Provide clear, actionable recommendations.
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
Imagine you are a marketing consultant reviewing the main headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the main headline's effectiveness based on various linguistic and marketing criteria.

---

### **Part 1: Main Headline Context**

**Image/Video:**
- **Main Headline Identification:**  
  Extract and clearly state the main headline from the image or video.

---

### **Part 2: Main Headline Analysis**

Present your evaluation in a well-formatted table. Use the table below as your template:

**Headline being analyzed:** [Main Headline]

| Criterion             | Assessment                          | Explanation                                                       | Recommendation                                            |
|-----------------------|-------------------------------------|-------------------------------------------------------------------|-----------------------------------------------------------|
| **Word Count**        | [Automatic count] words             | The headline has [x] words, which is [appropriate/lengthy].       | Consider [reducing/increasing] the word count to [y].      |
| **Keyword Relevance** | [High/Moderate/Low]                 | The headline [includes/misses] relevant keywords such as [x].     | Incorporate [more/specific] keywords like [y].           |
| **Common Words**      | [Number] common words               | Common words [enhance/reduce] readability and appeal.             | [Increase/reduce] the use of common words.                |
| **Uncommon Words**    | [Number] uncommon words             | Uncommon words make the headline [stand out/confusing].           | Balance [common/uncommon] words for clarity.              |
| **Power Words**       | [Number] power words                | Power words [create urgency/may overwhelm] the reader.            | Use power words [more sparingly/more effectively].        |
| **Emotional Words**   | [Number] emotional words            | Emotional tone is [effective/overdone/subtle].                    | Adjust the emotional tone by [modifying x].               |
| **Sentiment**         | [Positive/Negative/Neutral]         | The sentiment is [not aligning well/matching] with the image.      | Match the sentiment more closely with the image.          |
| **Reading Grade Level** | [Grade level] required              | The headline is [too complex/simple] for the target audience.       | Adapt the reading level to [simplify/complexify].         |

**Total Score:** _[Sum of all scores]_
---
### **Part 3: Improved Headline Suggestions**

Based on your overall analysis, provide suggestions for improving the main headline. Include three alternative headline options along with a brief explanation for each:

- **Option 1:** [Headline] – [Explanation]
- **Option 2:** [Headline] – [Explanation]
- **Option 3:** [Headline] – [Explanation]
---
**Final Instructions:**

- Ensure all responses are in English.
- Maintain the specified table format and consistent scoring.
- Focus on the synergy between the image content and the main headline.
- Provide clear, actionable recommendations for improvement.
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
Imagine you are a marketing consultant reviewing the image headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the image headline's effectiveness based on various linguistic and marketing criteria.

---

### **Part 1: Image Headline Context**

**Image/Video:**

- **Image Headline Identification:**  
  Extract and clearly state the separate headline from the image or video.

---

### **Part 2: Image Headline Analysis**

Analyze the extracted image headline and present your evaluation in the table below. Use the following table format:

| Criterion             | Assessment                      | Explanation                                                      | Recommendation                                       |
|-----------------------|---------------------------------|------------------------------------------------------------------|------------------------------------------------------|
| **Word Count**        | [Automatic count] words         | The headline length is [appropriate/lengthy] for visibility.       | Adjust the word count to [increase/decrease] clarity.|
| **Keyword Relevance** | [High/Moderate/Low]             | Headline's keywords [align/do not align] with visual content.      | Enhance keyword alignment for better SEO.            |
| **Common Words**      | [Number] common words           | Common words [aid/hinder] immediate comprehension.                 | Optimize common word usage for [audience/type].      |
| **Uncommon Words**    | [Number] uncommon words         | Uncommon words add [uniqueness/confusion].                         | Find a balance in word rarity for better engagement.  |
| **Power Words**       | [Number] power words            | Uses power words to [effectively/too aggressively] engage.         | Adjust power word usage for subtlety.                |
| **Emotional Words**   | [Number] emotional words        | Emotional words [evoke strong/a weak] response.                    | Modify emotional words to better suit the tone.      |
| **Sentiment**         | [Positive/Negative/Neutral]     | Sentiment [supports/contradicts] the visual theme.                 | Align the sentiment more with the visual message.    |
| **Reading Grade Level** | [Grade level] required         | Reading level is [ideal/not ideal] for the target demographic.     | Tailor the complexity to better fit the audience.     |

**Total Score:** _[Sum of all scores]_

---

### **Part 3: Recommendations**

Based on your analysis, suggest three improved headlines for the image headline. Provide each alternative with a brief explanation:

- **Option 1:** [Headline] – [Explanation]
- **Option 2:** [Headline] – [Explanation]
- **Option 3:** [Headline] – [Explanation]

---

**Final Instructions:**

- Ensure all responses are provided in English.
- Follow the specified table format and maintain consistent scoring.
- Focus on the synergy between the image content and the image headline.
- Provide clear, actionable recommendations for improvement.
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
Imagine you are a marketing consultant reviewing the supporting headline text of a marketing asset ({'image' if is_image else 'video'}) for a client. Your task is to assess the supporting headline's effectiveness based on various linguistic and marketing criteria.

**Part 1: Supporting Headline Context**

**Image/Video:**
- **Supporting Headline Identification:**  
  Identify and state any supporting headlines present in the provided image or video frame.

**Part 2: Supporting Headline Analysis**

Format your analysis in a table with the following columns: **Criterion**, **Assessment**, **Explanation**, and **Recommendation**. The table should include the following rows:

| Criterion               | Assessment                   | Explanation                                                      | Recommendation                                       |
|-------------------------|------------------------------|------------------------------------------------------------------|------------------------------------------------------|
| **Word Count**          | [Automatic count] words      | The supporting headline's length is [optimal/too long/short].    | Aim for a word count of [x] for better engagement.   |
| **Keyword Relevance**   | [High/Moderate/Low]          | Keywords used are [not sufficiently/sufficiently] relevant.      | Incorporate more relevant keywords like [y].         |
| **Common Words**        | [Number] common words        | Utilization of common words [enhances/detracts from] impact.       | Adjust common word usage to improve clarity.         |
| **Uncommon Words**      | [Number] uncommon words      | Uncommon words help [distinguish/muddle] the message.              | Use uncommon words to [highlight/clarify] message.     |
| **Power Words**         | [Number] Power words         | Power words [effectively/ineffectively] persuade the audience.     | Refine the use of power words for better impact.     |
| **Emotional Words**     | [Number] emotional words     | Emotional expression is [strong/weak], affecting impact.           | Enhance/reduce emotional wording for desired effect. |
| **Sentiment**           | [Positive/Negative/Neutral]  | Sentiment of the headline [aligns/conflicts] with main content.    | Adjust sentiment to [complement/contrast] main tone.   |
| **Reading Grade Level** | [Grade level] required       | The complexity suits [or does not suit] the intended audience.     | Modify to [simplify/complexify] reading level.         |

**Part 3: Revised Headline Suggestions**

Based on your analysis, offer three alternative headlines that enhance the supporting headline's effectiveness. Provide each suggestion with a brief explanation:

- **Option 1:** [Headline] – [Explanation]
- **Option 2:** [Headline] – [Explanation]
- **Option 3:** [Headline] – [Explanation]

Ensure your response is entirely in English, and focus on the synergy between the image content and the supporting headline. Provide clear and actionable recommendations.
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
Imagine you are a marketing consultant tasked with creating Facebook targeting personas for a product. Based on the following targeting elements for Facebook, please perform the following tasks:

1. **Describe 4 Persona Types:**
   - Identify 4 persona types that are most likely to respond to the ad.
   - Present these in a table with two columns: **Persona Type** and **Description**.

2. **Create 4 Detailed Personas:**
   - For each persona type, create a persona (include a name) who would likely purchase this product.
   - Describe how you expect them to react to the product, detailing their characteristics.
   - For each persona, present a table with three columns: **Persona Type**, **Description**, and **Analysis**.
   - In your analysis, include each of the characteristics available in Facebook targeting and specify what you would select. These characteristics include:
     - **Location:** Countries, states, cities, or specific addresses/zip codes.
     - **Age:** The age range of the audience.
     - **Gender:** Men, women, or all genders.
     - **Languages:** The languages they speak.
     - **Interests:** Based on activities, liked pages, topics (entertainment, fitness, hobbies, etc.).
     - **Behaviors:** User behavior such as device usage, travel patterns, purchase behavior, etc.
     - **Purchase Behavior:** Users who have made purchases in specific categories.
     - **Device Usage:** Devices used to access Facebook (mobiles, tablets, desktops).
     - **Connections:** Interaction with your Pages, Apps, or Events.
     - **Life Events:** Important events like anniversaries, birthdays, recently moved, newly engaged, or having a baby.
     - **Education Level:** Educational background.
     - **Education Fields of Study:** Specific areas of study.
     - **Job Title:** Professional information.
     - **Job Title Industries:** Industries related to their job information.

Please ensure that all responses are presented in English and are structured in a clear, tabular format as specified.
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
Imagine you are a marketing consultant tasked with creating LinkedIn targeting personas for a product. Based on the following LinkedIn targeting elements, perform the following tasks:

1. **Describe 4 Persona Types:**
   - Identify 4 persona types that are most likely to respond to the ad.
   - Present these in a table with the columns: **Persona Type** and **Description**.

2. **Create 4 Detailed Personas:**
   - For each persona type, create a persona (include a name) who would likely purchase this product.
   - Describe how you expect them to react to the product by detailing their characteristics.
   - For each persona, present a table with the columns: **Persona Type**, **Description**, and **Analysis**.
   - In your analysis, include each targeting characteristic available on LinkedIn along with your selection. The targeting elements to consider are:
     - **Location:** Country, city, or region.
     - **Age:** (Note: LinkedIn does not directly allow age and gender targeting, but these can be inferred through other demographic details.)
     - **Gender:** (Similarly inferred.)
     - **Company Industry:** Target professionals in specific industries.
     - **Company Size:** Based on the number of employees.
     - **Job Functions:** Specific job functions within companies.
     - **Job Seniority:** From entry-level to senior executives and managers.
     - **Job Titles:** Specific roles within companies.
     - **Years of Experience:** How long users have been in the professional workforce.
     - **Schools:** Alumni of specific educational institutions.
     - **Degrees:** Users who hold specific degrees.
     - **Fields of Study:** Subjects studied.
     - **Skills:** Listed skills on their profiles.
     - **Member Groups:** Membership in LinkedIn groups related to professional interests.
     - **Interests:** Content interactions or listed interests.
     - **Traits:** Member traits reflecting user activities and behaviors on LinkedIn.

Ensure your final response is entirely in English and that all tables follow the specified format.
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
Imagine you are a marketing consultant tasked with creating targeting personas for platform X. Based on the following targeting elements for X, please complete the following tasks:

1. **Describe 4 Persona Types:**
   - Identify 4 persona types most likely to respond to the ad.
   - Present these in a table with two columns: **Persona Type** and **Description**.

2. **Create 4 Detailed Personas:**
   - For each persona type, create a detailed persona (including a name) who would likely purchase this product.
   - Describe how you expect each persona to react to the product by detailing their characteristics.
   - For each persona, present your findings in a table with three columns: **Persona Type**, **Description**, and **Analysis**.
   - In your analysis, include each targeting characteristic available on X and specify what you would select. The targeting elements include:
     - **Location:** Target users by country, region, or metro area; more granular options such as city or postal code are available.
     - **Gender:** Select audiences based on gender.
     - **Language:** Target users based on the language they speak.
     - **Interests:** Target users based on their interests, inferred from their activities and engagement topics on X.
     - **Events:** Target ads around specific events—both global and local—that generate significant engagement.
     - **Behaviors:** Target based on user behaviors and actions, such as what they tweet or engage with.
     - **Keywords:** Target users based on keywords in their tweets or tweets they engage with; useful for capturing intent in real time.
     - **Topics:** Engage users involved in conversations around predefined or custom topics.
     - **Device:** Target users based on the devices or operating systems they use to access X.
     - **Carrier:** Target users based on their mobile carrier, which is useful for mobile-specific campaigns.
     - **Geography:** Fine-tune targeting based on user location to match cultural contexts and regional norms.

Please ensure that all responses are provided in English and follow the specified table formats. The final output should be clear, precise, and consistent, focusing on the synergy between the targeting elements and the personas.
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
        
    def Personality_Trait_Assessment(uploaded_file, is_image=True):
       prompt = f"""
If the content is non-English, first translate it to English. Then, evaluate the content against the following personality trait models. For each personality trait, provide a score from 1 to 5 (in increments of 0.5) based on how well the content is likely to resonate with that personality type. Please include columns for Criterion, Score, Analysis, and Recommendation. At the end, provide an overall summary and overall recommendations.

**Main Personality Trait Models:**

1. **The Big Five Personality Traits (OCEAN/CANOE)**
   - Traits:
     - Openness to Experience: Imagination, creativity, curiosity, preference for novelty.
     - Conscientiousness: Organization, dependability, discipline, goal-directed behavior.
     - Extraversion: Sociability, assertiveness, excitement-seeking, positive emotionality.
     - Agreeableness: Compassion, cooperation, trust, kindness.
     - Neuroticism: Emotional instability, anxiety, moodiness, sadness.

2. **Eysenck’s Three-Factor Model (PEN Model)**
   - Traits:
     - Psychoticism: Aggressiveness, impulsivity, lack of empathy.
     - Extraversion: Sociability, liveliness, activity.
     - Neuroticism: Emotional instability, anxiety, moodiness.

3. **HEXACO Model**
   - Traits:
     - Honesty-Humility: Sincerity, fairness, modesty, low greed.
     - Emotionality: Similar to Neuroticism with sentimentality and dependence.
     - Extraversion: Sociability, assertiveness, enthusiasm.
     - Agreeableness: Patience, forgiveness, cooperation.
     - Conscientiousness: Organization, diligence, reliability.
     - Openness to Experience: Aesthetic appreciation, inquisitiveness, creativity.

4. **Cattell’s 16 Personality Factors (16PF)**
   - Traits: Includes warmth, reasoning, emotional stability, dominance, liveliness, rule-consciousness, social boldness, sensitivity, vigilance, and others.

5. **Myers-Briggs Type Indicator (MBTI)**
   - Traits (Dichotomies):
     - Extraversion (E) vs. Introversion (I)
     - Sensing (S) vs. Intuition (N)
     - Thinking (T) vs. Feeling (F)
     - Judging (J) vs. Perceiving (P)

6. **The Dark Triad**
   - Traits:
     - Machiavellianism: Manipulativeness, deceitfulness, personal gain focus.
     - Narcissism: Excessive self-love, entitlement, need for admiration.
     - Psychopathy: Lack of empathy, impulsivity, antisocial behaviors.

7. **Cloninger’s Temperament and Character Inventory (TCI)**
   - Traits:
     - Novelty Seeking: Impulsiveness and a desire for excitement.
     - Harm Avoidance: Caution and risk aversion.
     - Reward Dependence: Sensitivity to social cues and approval.
     - Persistence: Perseverance despite challenges.

8. **Enneagram of Personality**
   - Types:
     - Reformer (Type 1): Perfectionistic, principled, self-controlled.
     - Helper (Type 2): Caring, generous, people-pleasing.
     - Achiever (Type 3): Success-oriented, adaptable, driven.
     - Individualist (Type 4): Sensitive, expressive, introspective.
     - Investigator (Type 5): Analytical, innovative, private.
     - Loyalist (Type 6): Committed, responsible, anxious.
     - Enthusiast (Type 7): Spontaneous, versatile, distractible.
     - Challenger (Type 8): Confident, assertive, confrontational.
     - Peacemaker (Type 9): Easygoing, agreeable, complacent.

9. **DISC Personality Model**
   - Traits:
     - Dominance (D): Assertive, results-oriented, driven.
     - Influence (I): Sociable, enthusiastic, persuasive.
     - Steadiness (S): Cooperative, patient, supportive.
     - Conscientiousness (C): Analytical, detail-oriented, systematic.

10. **Keirsey Temperament Sorter**
    - Temperaments:
      - Artisan: Spontaneous, adaptable, action-oriented.
      - Guardian: Dependable, detail-focused, community-minded.
      - Idealist: Empathetic, enthusiastic, growth-oriented.
      - Rational: Strategic, logical, problem-solving.

11. **Revised NEO Personality Inventory (NEO-PI-R)**
    - An extended assessment of the Big Five, providing detailed facets (e.g., anxiety, excitement-seeking, orderliness) within each factor.

12. **Jungian Archetypes**
    - Archetypes:
      - The Hero: Courage, strength, resilience.
      - The Caregiver: Nurturing, supportive, protective.
      - The Explorer: Adventure-seeking, discovery-oriented.
      - The Rebel: Challenging authority, seeking change.
      - The Lover: Valuing relationships, passion, and connection.
      - The Creator: Imaginative, innovative, artistic.

**Instructions:**

- Evaluate the content against each of the above personality trait models.
- For each trait or personality type, provide:
  - **Criterion:** The personality trait or model being evaluated.
  - **Score:** A rating from 1 to 5 (in increments of 0.5) based on how well the content is likely to resonate with that personality type.
  - **Analysis:** A brief explanation for the given score.
  - **Recommendation:** Specific suggestions for how the content could be improved to better engage that personality type.

- Present your results in a table with the following columns:
  | Criterion | Score | Analysis | Recommendation |
  
- After the table, provide an overall summary of your findings along with overall recommendations for enhancing the content’s appeal across these personality traits.

Ensure your response is entirely in English and that it focuses on the user's image analysis perspective, offering clear, actionable insights.
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
                st.write("Personality Trait Assessment Results::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
        
    def BMTI_Analysis(uploaded_file, is_image=True):
       prompt = f"""
If the content is non-English, translate it to English. Then, evaluate the content against the following MBTI personality types in a table. For each personality type, provide a score from 1 to 5 (in increments of 0.5) reflecting how well the personality would perceive and respond to the content. Include columns for Analysis and Recommendations. At the end, provide an overall summary and overall recommendations.

The Myers-Briggs Type Indicator (MBTI) consists of 16 personality types, each a combination of four dichotomies:

- Extraversion (E) vs. Introversion (I): Focus on the outer world vs. the inner world.
- Sensing (S) vs. Intuition (N): Focus on concrete details vs. abstract concepts.
- Thinking (T) vs. Feeling (F): Decision-making based on logic vs. emotions.
- Judging (J) vs. Perceiving (P): Preference for structure vs. spontaneity.

**The 16 MBTI Personality Types:**

1. **ISTJ - The Inspector**  
   Introverted, Sensing, Thinking, Judging  
   Practical, fact-minded, and responsible.

2. **ISFJ - The Protector**  
   Introverted, Sensing, Feeling, Judging  
   Kind, conscientious, and dedicated to serving others.

3. **INFJ - The Advocate**  
   Introverted, Intuitive, Feeling, Judging  
   Idealistic, insightful, and driven by personal values.

4. **INTJ - The Architect**  
   Introverted, Intuitive, Thinking, Judging  
   Strategic, logical, and determined innovators.

5. **ISTP - The Virtuoso**  
   Introverted, Sensing, Thinking, Perceiving  
   Bold, practical, and skilled at handling tools and situations.

6. **ISFP - The Adventurer**  
   Introverted, Sensing, Feeling, Perceiving  
   Flexible, charming, and live in the moment.

7. **INFP - The Mediator**  
   Introverted, Intuitive, Feeling, Perceiving  
   Idealistic, creative, and driven by core values.

8. **INTP - The Logician**  
   Introverted, Intuitive, Thinking, Perceiving  
   Analytical, curious, and enjoy exploring ideas and concepts.

9. **ESTP - The Entrepreneur**  
   Extraverted, Sensing, Thinking, Perceiving  
   Energetic, spontaneous, and enjoy living on the edge.

10. **ESFP - The Entertainer**  
    Extraverted, Sensing, Feeling, Perceiving  
    Fun-loving, sociable, and love the spotlight.

11. **ENFP - The Campaigner**  
    Extraverted, Intuitive, Feeling, Perceiving  
    Enthusiastic, imaginative, and enjoy exploring possibilities.

12. **ENTP - The Debater**  
    Extraverted, Intuitive, Thinking, Perceiving  
    Quick-witted, innovative, and love intellectual challenges.

13. **ESTJ - The Executive**  
    Extraverted, Sensing, Thinking, Judging  
    Organized, direct, and enjoy taking charge of situations.

14. **ESFJ - The Consul**  
    Extraverted, Sensing, Feeling, Judging  
    Caring, sociable, and value harmony in relationships.

15. **ENFJ - The Protagonist**  
    Extraverted, Intuitive, Feeling, Judging  
    Charismatic, inspiring, and love helping others reach their potential.

16. **ENTJ - The Commander**  
    Extraverted, Intuitive, Thinking, Judging  
    Bold, strategic, and love to lead.

**Instructions:**

- For each of the 16 MBTI personality types, provide:
  - **Score:** A rating from 1 to 5 (in increments of 0.5) reflecting how well the content is likely to resonate with that personality type.
  - **Analysis:** A brief explanation for the given score.
  - **Recommendation:** Specific suggestions for how the content could be improved to better engage that personality type.

- Present your findings in a table with the following columns:
  | Personality Type | Score | Analysis | Recommendation |

- After the table, include an overall summary and overall recommendations for enhancing the content's appeal across these personality types.

Ensure your entire response is in English and that your analysis is focused on the user's image analysis perspective.
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
                st.write("BMTI Analysis Results::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None        
                
        
    def Image_Analysis(uploaded_file, is_image=True):
        prompt = f"""
If the content is non-English, translate it to English.
For each aspect listed below, provide a score from 1 to 5 (in increments of 0.5, where 1 is low and 5 is high), along with an explanation and suggestions for improvement. Present your results in a table with the following columns: **Aspect**, **Score**, **Explanation**, and **Improvement**. After the table, provide an overall summary and overall recommendations.

Please evaluate the content based on the following aspects:

1. **Visual Appeal**
   - **Impact:** Attracts attention and conveys emotions.
   - **Analysis:** Assess the color scheme, composition, clarity, and overall aesthetic quality.
   - **Application:** Ensure the image is clear, visually appealing, and professionally designed.

2. **Relevance**
   - **Impact:** Resonates with the target audience.
   - **Analysis:** Determine if the image aligns with audience preferences, context, and brand values.
   - **Application:** Adjust the image to better match the audience’s interests and brand messaging.

3. **Emotional Impact**
   - **Impact:** Evokes the desired emotions.
   - **Analysis:** Analyze the emotional resonance and connection the image creates.
   - **Application:** Use storytelling and relatable scenarios to strengthen emotional engagement.

4. **Message Clarity**
   - **Impact:** Communicates the intended message effectively.
   - **Analysis:** Ensure the main subject is clear and that the image is free from clutter.
   - **Application:** Focus on the key message and simplify the design for better clarity.

5. **Engagement Potential**
   - **Impact:** Captures and retains audience attention.
   - **Analysis:** Evaluate the attention-grabbing elements and interaction potential.
   - **Application:** Incorporate compelling visuals and narratives to encourage interaction.

6. **Brand Recognition**
   - **Impact:** Enhances brand recall and association.
   - **Analysis:** Check for visible and well-integrated brand elements.
   - **Application:** Use consistent brand colors, logos, and style to reinforce identity.

7. **Cultural Sensitivity**
   - **Impact:** Respects and represents cultural norms and diversity.
   - **Analysis:** Assess the image for inclusivity and cultural appropriateness.
   - **Application:** Ensure the image is culturally sensitive and globally appealing.

8. **Technical Quality**
   - **Impact:** Maintains high resolution and professional editing.
   - **Analysis:** Evaluate resolution, lighting, and post-processing quality.
   - **Application:** Use high-resolution images with proper lighting and professional editing.

9. **Color**
   - **Impact:** Influences mood, perception, and attention.
   - **Analysis:** Analyze the psychological impact of the colors used.
   - **Application:** Use colors purposefully to evoke desired emotions and reinforce brand recognition.

10. **Typography**
    - **Impact:** Affects readability and engagement.
    - **Analysis:** Assess font choice, size, placement, and overall readability.
    - **Application:** Ensure typography complements the image and enhances readability.

11. **Symbolism**
    - **Impact:** Conveys complex ideas quickly.
    - **Analysis:** Examine the use of symbols and icons within the image.
    - **Application:** Use universally recognized symbols that align with the ad’s message.

12. **Contrast**
    - **Impact:** Highlights important elements and improves visibility.
    - **Analysis:** Evaluate the contrast between different elements in the image.
    - **Application:** Use contrast effectively to draw attention to key parts of the image.

13. **Layout Balance**
    - **Impact:** Ensures the image is visually balanced and aesthetically pleasing.
    - **Analysis:** Assess the distribution of elements to confirm even visual weight.
    - **Application:** Arrange elements to avoid clutter and achieve a harmonious balance.

14. **Hierarchy**
    - **Impact:** Guides the viewer’s eye to the most important elements first.
    - **Analysis:** Evaluate the visual hierarchy to ensure key elements stand out.
    - **Application:** Use size, color, and placement to direct attention to the primary message.

Present your analysis in a table with columns: **Aspect**, **Score**, **Explanation**, **Improvement**.
After the table, provide an overall summary and your overall recommendations for improving the content.
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
                st.write("Image Analysis::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
        
    def Image_Analysis_2(uploaded_file, is_image=True):
        prompt = f"""
If the content is non-English, translate it to English.
Please evaluate the image against the following principles. For each aspect, provide a score from 1 to 5 (in increments of 0.5, where 1 is low and 5 is high), an explanation, and suggestions for improvement. Present your results in a table with the columns: **Aspect**, **Score**, **Explanation**, and **Improvement**. After the table, include a concise overall summary with suggestions for overall improvement.

**Aspects to Evaluate:**

1. **Emotional Appeal**
   - Does the image evoke a strong emotional response?
   - What specific emotions are triggered (e.g., happiness, nostalgia, excitement, urgency)?
   - How might these emotions influence the viewer's perception of the brand or product?
   - How well does the image align with the intended emotional tone of the campaign?
   - Does the emotional tone match the target audience's expectations and values?

2. **Eye Attraction**
   - Does the image grab attention immediately?
   - Which elements (color, subject, composition) are most effective in drawing the viewer’s attention?
   - Is there anything in the image that distracts from the main focal point?
   - Is there a clear focal point that naturally draws the viewer's eye?
   - How effectively does the focal point communicate the key message or subject?

3. **Visual Appeal**
   - How aesthetically pleasing is the image overall?
   - Are the elements of balance, symmetry, and composition well-executed?
   - Does the image use unique or creative visual techniques to enhance its appeal?
   - Are the visual elements harmonious and balanced?
   - Do any elements feel out of place or clash with the overall design?

4. **Text Overlay (Clarity, Emotional Connection, Readability)**
   - Is the text overlay easily readable?
   - Is there sufficient contrast between the text and the background?
   - Are font size, style, and color appropriate for readability?
   - Does the text complement the image and enhance emotional connection?
   - Is the messaging clear, concise, and impactful?
   - Does the text align with the brand's identity and maintain consistency with its tone and voice?

5. **Contrast and Clarity**
   - Is there adequate contrast between different elements of the image?
   - How well do the foreground and background elements distinguish themselves?
   - Does the contrast help highlight the key message or subject?
   - Is the image clear and sharp?
   - Are all important details easily distinguishable?
   - Does the image suffer from any blurriness or pixelation?

6. **Visual Hierarchy**
   - Is there a clear visual hierarchy guiding the viewer’s eye?
   - Are the most important elements (e.g., brand name, product, call to action) placed prominently?
   - How effectively does the hierarchy direct attention from one element to the next?
   - Are key elements ordered by importance?
   - Does the visual flow reinforce the intended message?

7. **Negative Space**
   - Is negative space used effectively to balance the composition?
   - Does negative space help focus attention on the key elements?
   - Is there enough negative space to avoid clutter without making the image feel empty?
   - Does the use of negative space enhance overall clarity and readability?
   - How does negative space contribute to the visual hierarchy?

8. **Color Psychology**
   - Are the colors used appropriate for the message and target audience?
   - Do the colors evoke the intended emotional response (e.g., trust, excitement, calm)?
   - Are any colors off-putting or conflicting?
   - How well do the colors align with the brand’s color palette and identity?
   - Does the color scheme contribute to or detract from brand recognition?

9. **Depth and Texture**
   - Does the image have a sense of depth and texture?
   - Are shadows, gradients, or layering techniques used effectively to create a three-dimensional feel?
   - How does depth or texture contribute to realism and engagement?
   - Is the texture or depth distracting or enhancing?
   - Does it add value to the visual appeal or complicate the message?

10. **Brand Consistency**
    - Is the image consistent with the brand’s visual identity?
    - Are color schemes, fonts, and overall style in line with brand guidelines?
    - Does the image reinforce the brand’s core values and messaging?
    - Does the image maintain a coherent connection to previous branding efforts?
    - Is there any risk of confusing the audience with a departure from established brand aesthetics?

11. **Psychological Triggers**
    - Does the image effectively use psychological triggers (e.g., scarcity, social proof, authority) to encourage a desired action?
    - How well do these triggers align with the target audience’s motivations and behaviors?
    - Are the psychological triggers subtle or overt?
    - Does the image risk appearing manipulative, or is the influence balanced and respectful?

12. **Emotional Connection**
    - How strong is the emotional connection between the image and the target audience?
    - Does the image resonate with the audience’s values, desires, or pain points?
    - Is the connection likely to inspire action or loyalty?
    - Is the emotional connection authentic and genuine, or does it feel forced?

13. **Suitable Effect Techniques**
    - Are any special effects or filters used in the image?
    - Do these effects enhance the overall message and visual appeal?
    - Are the effects aligned with the brand’s identity and the image’s purpose?
    - Do these effects support the key message and theme, or do they distract from it?

14. **Key Message and Subject**
    - Is the key message of the image clear and easily understood at a glance?
    - Is the message prominent, or does it get lost among other elements?
    - How well does the image communicate its purpose or call to action?
    - Is the subject (product, service, idea) highlighted appropriately and does it stand out?
    - Is there a clear connection between the subject and the intended message?

**Instructions:**
- Present your evaluation in a table with the columns: **Aspect**, **Score**, **Explanation**, **Improvement**.
- After the table, provide an overall summary with suggestions for overall improvement.
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
                st.write("Image Analysis 2 ::")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None
        except Exception as e:
            st.error(f"Failed to read or process the media: {e}")
            return None
        
    def Image_Analysis_2_table(uploaded_file, is_image=True):
       prompt = f"""
If the content is non-English, translate it to English.
Please evaluate the image against the following principles. For each aspect, provide a score from 1 to 5 (in increments of 0.5, where 1 is low and 5 is high), along with a brief explanation and suggestions for improvement. Present your results in a table with the columns: **Aspect**, **Score**, **Explanation**, and **Improvement**. After the table, include an overall summary with recommendations for overall improvement.

**Aspects to Evaluate:**

1. **Emotional Appeal**
   - Does the image evoke a strong emotional response?
   - What specific emotions are triggered (e.g., happiness, nostalgia, excitement, urgency)?
   - How might these emotions influence the viewer's perception of the brand or product?
   - How well does the image align with the intended emotional tone of the campaign?
   - Does the emotional tone match the target audience's expectations and values?

2. **Eye Attraction**
   - Does the image grab attention immediately?
   - Which elements (color, subject, composition) effectively draw the viewer’s attention?
   - Is there any element that distracts from the main focal point?
   - Is there a clear focal point that naturally draws the viewer's eye?
   - How effectively does the focal point communicate the key message or subject?

3. **Visual Appeal**
   - How aesthetically pleasing is the image overall?
   - Are the elements of balance, symmetry, and composition well-executed?
   - Does the image use any unique or creative visual techniques that enhance its appeal?
   - Are the visual elements harmonious and balanced?
   - Do any elements feel out of place or clash with the overall design?

4. **Text Overlay (Clarity, Emotional Connection, Readability)**
   - Is the text overlay easily readable?
   - Is there sufficient contrast between the text and the background?
   - Are font size, style, and color appropriate for readability?
   - Does the text complement the image and enhance emotional connection?
   - Is the messaging clear, concise, and impactful?
   - Does the text align with the brand's identity and maintain consistency with its tone and voice?

5. **Contrast and Clarity**
   - Is there adequate contrast between different elements of the image?
   - How well do the foreground and background elements distinguish themselves?
   - Does the contrast help highlight the key message or subject?
   - Is the image clear and sharp?
   - Are all important details easily distinguishable?
   - Does the image suffer from any blurriness or pixelation?

6. **Visual Hierarchy**
   - Is there a clear visual hierarchy guiding the viewer’s eye?
   - Are the most important elements (e.g., brand name, product, call to action) placed prominently?
   - How effectively does the hierarchy direct attention from one element to the next?
   - Are key elements ordered by importance?
   - Does the visual flow reinforce the intended message?

7. **Negative Space**
   - Is negative space used effectively to balance the composition?
   - Does the negative space help focus attention on the key elements?
   - Is there enough negative space to avoid clutter without making the image feel empty?
   - Does the use of negative space enhance overall clarity of the message?
   - How does it contribute to the visual hierarchy and readability?

8. **Color Psychology**
   - Are the colors used in the image appropriate for the message and target audience?
   - Do the colors evoke the intended emotional response (e.g., trust, excitement, calm)?
   - Are any colors potentially off-putting or conflicting?
   - How well do the colors align with the brand’s color palette and identity?
   - Does the color scheme contribute to or detract from brand recognition?

9. **Depth and Texture**
   - Does the image have a sense of depth and texture?
   - Are shadows, gradients, or layering techniques used effectively to create a three-dimensional feel?
   - How does the depth or texture contribute to the realism and engagement of the image?
   - Is the texture or depth distracting or enhancing?
   - Does it add value to the visual appeal or complicate the message?

10. **Brand Consistency**
    - Is the image consistent with the brand’s visual identity?
    - Are color schemes, fonts, and overall style in line with brand guidelines?
    - Does the image reinforce the brand’s core values and messaging?
    - Does it maintain a coherent connection to previous branding efforts?
    - Is there any risk of confusing the audience with a departure from established brand aesthetics?

11. **Psychological Triggers**
    - Does the image use any psychological triggers effectively (e.g., scarcity, social proof, authority)?
    - How well do these triggers align with the target audience’s motivations and behaviors?
    - Are the psychological triggers subtle or overt?
    - Does the image risk appearing manipulative, or is the influence balanced and respectful?

12. **Emotional Connection**
    - How strong is the emotional connection between the image and the target audience?
    - Does the image resonate with the audience’s values, desires, or pain points?
    - Is the connection likely to inspire action or loyalty?
    - Is the emotional connection authentic and genuine, or does it feel forced?

13. **Suitable Effect Techniques**
    - Are any special effects or filters used in the image?
    - Do these effects enhance the overall message and visual appeal?
    - Are the effects aligned with the brand’s identity and the image’s purpose?
    - Do these effects support the key message and theme, or do they distract from it?

14. **Key Message and Subject**
    - Is the key message of the image clear and easily understood at a glance?
    - Is the message prominent, or does it get lost among other elements?
    - How well does the image communicate its purpose or call to action?
    - Is the subject of the image (product, service, idea) highlighted appropriately and does it stand out?
    - Is there a clear connection between the subject and the intended message?

**Instructions:**
- Present your evaluation in a table with the columns: **Aspect**, **Score**, **Explanation**, **Improvement**.
- After the table, provide an overall summary with suggestions for overall improvement.
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
                st.write("Image Analysis 2 Table ::")
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
   - For Images:
     - Describe the prominent visual elements (e.g., objects, people, animals, settings).
     - Note the dominant colors and their overall effect.
     - Mention any text present, including its content, font style, size, and placement.
     - Describe the composition and layout of the elements.
   - For Videos:
     - Describe the key scenes, actions, and characters.
     - Note the visual style, color palette, and editing techniques.
     - Mention any text overlays, captions, or speech (transcribe if possible).
     - Identify background music or sound effects, if present.

2. Cultural References and Symbolism:
   - Identify any cultural references, symbols, or visual metaphors that could be significant to the target audience.
   - Explain how these elements might be interpreted or resonate with the audience.

3. Marketing Implications:
   - Summarize the potential marketing implications based on the visual and textual elements.
   - Consider how the asset might appeal to different demographics or interests.
   - Mention any potential positive or negative associations the asset may evoke.

4. Additional Notes:
   - For Video Analysis: Focus on the most representative frame(s) for the initial description.
   - Note any significant changes or variations in visuals or text throughout the video.

Ensure your description is:
- **Objective:** Focus solely on factual details without subjective opinions.
- **Detailed:** Provide comprehensive information for the client to fully understand the asset's visual and textual content.
- **Marketing-Oriented:** Highlight elements that are relevant to marketing strategy and decision-making.
- **Consistent:** Maintain a uniform approach in your descriptions, regardless of repeated analyses of similar assets.
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
    def motivation(uploaded_file, is_image=True):
        prompt = f"""
If the content is non-English, translate it to English.
Your task is to evaluate the content based on Self-Determination Theory (SDT), focusing on how well the content satisfies the audience's psychological needs for autonomy, competence, and relatedness. For each aspect listed below, provide a score from 1 to 5 (in increments of 0.5, where 1 is low and 5 is high), along with a brief explanation and suggestions for improvement. Present your results in a table with the columns: **Aspect**, **Score**, **Explanation**, and **Improvement**. After the table, include an overall summary with suggestions for overall improvement.

**Aspects to Evaluate:**

1. **Autonomy (The need to feel in control and have choices)**
   - Does the content provide the audience with a sense of choice or control in their decision-making process?
   - Does it allow the consumer to feel that they are making their own decision rather than being pressured?
   - Are there options or customizable features that emphasize personal control over the purchase?
   - How well does the content empower the audience to make an informed decision?
   - Is the content transparent, providing clear and unbiased information that boosts confidence?
   - Does the content respect the audience’s intelligence by avoiding manipulative language or fear-based tactics?
   - Is the messaging personalized to acknowledge the audience’s unique preferences and needs?

2. **Competence (The need to feel effective and capable)**
   - Does the content make the audience feel capable of successfully using the product or service?
   - Is the product or service presented in a way that highlights ease of use, reducing potential uncertainty or inadequacy?
   - Are the benefits and usage instructions clearly explained to boost the audience's confidence?
   - Does the content illustrate how the product can enhance the consumer’s skills, knowledge, or overall effectiveness?
   - Are there testimonials, case studies, or examples that encourage an "I can do this too" sentiment?

3. **Relatedness (The need to feel connected to others or a community)**
   - Does the content create a sense of belonging or community?
   - Does it emphasize social proof, such as reviews or user-generated content, to demonstrate a network of satisfied customers?
   - Is there a clear alignment with a community or cause that makes the audience feel connected?
   - How well does the content build emotional resonance and human connection?
   - Does the content evoke feelings of trust, warmth, or empathy that foster a relationship with the brand?
   - Is the messaging aligned with the audience’s values, making them feel understood and part of a larger purpose?

4. **General SDT-Aligned Content Assessment**
   - Does the content create a balance between external incentives (e.g., promotions, discounts) and intrinsic motivation (e.g., personal values, empowerment)?
   - Does it avoid over-relying on extrinsic motivators in favor of long-term personal satisfaction?
   - Is the audience encouraged to consider how the product satisfies deeper, lasting needs (autonomy, competence, relatedness) rather than just surface-level wants?
   - Does the content help the audience see how the purchase will meaningfully improve their life and align with their internal motivations (self-improvement, connection, self-sufficiency)?

**Motivational Score Calculation:**
At the end of the table, calculate a **Motivational Score** based on:
   - 50% of the Autonomy Score
   - 30% of the Competence Score
   - 20% of the Relatedness Score

**Instructions:**
- Present your evaluation in a table with columns: **Aspect**, **Score**, **Explanation**, **Improvement**.
- After the table, provide a concise overall summary with recommendations for overall improvement.
- Ensure your entire response is in English and remains focused on the user's image analysis perspective.
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
                st.write("Motivation Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
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
# --- Streamlit App ---
st.title("Marketing Media Analysis AI Assistant with Gemini-2.0-flash")

# --- Sidebar ---
with st.sidebar:
    st.header("Analysis Options")
    tabs = st.tabs(["Basic", "Detailed", "Headlines", "Persona", "Others"])

    # Analysis buttons within each tab
    with tabs[0]:  # Basic
        basic_analysis = st.button("Basic Analysis")
        motivation_button = st.button("Motivation")
        flash_analysis_button = st.button("Flash Analysis")
        emotional_resonance_button=st.button("Emotional Resonance")
        emotional_analysis_button=st.button("Emotional Analysis")
        Emotional_Appraisal_Models_button=st.button("Emotional Appraisal Models")
        Image_Analysis_button=st.button("Image Analysis")
        Image_Analysis_2_button = st.button("Image Analysis 2")
        Image_Analysis_2_table_button = st.button("Image Analysis 2 table")                

    with tabs[1]:  # Detailed
        behavioural_principles_button = st.button("Behaviour Principles")
        nlp_principles_analysis_button = st.button("NLP Principles Analysis")
        overall_analysis_button = st.button("Overall Marketing Analysis")
        Story_Telling_Analysis_button = st.button("Story Telling Analysis")
        text_analysis_button = st.button("Text Analysis")
        text_analysis_2_button = st.button("Text Analysis 2")
        text_analysis_2_table_button = st.button("Text Analysis 2 - table")

    with tabs[2]:  # Headlines
        headline_analysis_button = st.button("Headline Analysis")
        detailed_headline_analysis_button = st.button("Headline Optimization Report")
        main_headline_analysis_button = st.button("Main Headline Analysis")
        image_headline_analysis_button = st.button("Image Headline Analysis")
        supporting_headline_analysis_button = st.button("Supporting Headline Analysis")

    with tabs[3]:  # Persona
        meta_profile_button = st.button("Facebook targeting")
        linkedin_profile_button = st.button("LinkedIn targeting")
        x_profile_button = st.button("X (formerly Twitter) targeting")
        personality_trait_assessment_button = st.button("Personality Trait Assessment")
        BMTI_Analysis_button = st.button("BMTI Analysis")
    with tabs[4]:  # Others
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
    key="general_media_uploader"  # Unique key for this uploader
)

# Display Uploaded Media (Responsive Design)
for uploaded_file in uploaded_files:
    is_image = uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]

    with st.container():  # Use container for better layout
        # Display the uploaded media
        if is_image:
            image = Image.open(uploaded_file)
            image = resize_image(image)  # Resize for display
            st.image(image, caption="Uploaded Image", use_column_width='auto')
        else:
            st.video(uploaded_file, format="video/mp4")

        # Analysis Results
        uploaded_file.seek(0)  # Reset file pointer for re-analysis

        # Check which analysis button was clicked and call the corresponding function
        if basic_analysis:
            with st.spinner("Performing basic analysis..."):
                result = analyze_media(uploaded_file, is_image)
                if result:
                    st.write("## Basic Analysis Results:")
                    st.markdown(result, unsafe_allow_html=True)
        if emotional_resonance_button:
            with st.spinner("Performing Emotional Resonance Analysis..."):
                result = emotional_resonance(uploaded_file, is_image)
                if result:
                    st.write("## Emotional Resonance Results:")
                    st.markdown(result)
        if emotional_analysis_button:
            with st.spinner("Performing Emotional Analysis..."):
                result = emotional_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Emotional Analysis Results:")
                    st.markdown(result)
        if Emotional_Appraisal_Models_button:
            with st.spinner("Performing Emotional Appraisal Models Analysis..."):
                result = Emotional_Appraisal_Models(uploaded_file, is_image)
                if result:
                    st.write("## Emotional Appraisal Models Analysis Results:")
                    st.markdown(result)
        elif flash_analysis_button:
            with st.spinner("Performing Flash analysis..."):
                result = flash_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Flash Analysis Results:")
                    st.markdown(result)  # Display results directly
        elif behavioural_principles_button:
            with st.spinner("Analyzing Behavioral Principles..."):
                result = behavioural_principles(uploaded_file, is_image)
                if result:
                    st.write("## Behavioral Principles Analysis Results:")
                    st.markdown(result, unsafe_allow_html=True)
        elif nlp_principles_analysis_button:
            with st.spinner("Analyzing NLP Principles..."):
                result = nlp_principles_analysis(uploaded_file, is_image)
                if result:
                    st.write("## NLP Principles Analysis Results:")
                    st.markdown(result, unsafe_allow_html=True)
        elif overall_analysis_button:
            with st.spinner("Performing overall marketing analysis..."):
                result = overall_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Overall Marketing Analysis Results:")
                    st.markdown(result)
        elif motivation_button:
            with st.spinner("Performing Motivation Analysis..."):
                result = motivation(uploaded_file, is_image)
                if result:
                    st.write("## Motivation Analysis Results:")
                    st.markdown(result)
        elif Story_Telling_Analysis_button:
            with st.spinner("Performing Story Telling Analysis..."):
                result = Story_Telling_Analysis(uploaded_file, is_image)
                if result:
                    st.write("## Overall Story Telling Analysis Results:")
                    st.markdown(result)
        elif text_analysis_button:
            with st.spinner("Performing text analysis..."):
                result = text_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Text Analysis Results:")
                    st.markdown(result)
        elif text_analysis_2_button:
            with st.spinner("Performing Text Analysis 2..."):
                result = Text_Analysis_2(uploaded_file, is_image)
                if result:
                    st.write("## Text Analysis 2 Results:")
                    st.markdown(result)
        elif text_analysis_2_table_button:
            with st.spinner("Performing Text Analysis 2 - Table Button..."):
                result = Text_Analysis_2_table(uploaded_file, is_image)
                if result:
                    st.write("## Text Analysis 2 - Table Results:")
                    st.markdown(result)
        elif Image_Analysis_2_button:
            with st.spinner("Performing Image Analysis 2..."):
                result = Image_Analysis_2(uploaded_file, is_image)
                if result:
                    st.write("## TImage Analysis 2 Results:")
                    st.markdown(result)
        elif Image_Analysis_2_table_button:
            with st.spinner("Performing Image Analysis 2 table..."):
                result = Image_Analysis_2_table(uploaded_file, is_image)
                if result:
                    st.write("## Image Analysis 2 table Results:")
                    st.markdown(result)                    
        elif headline_analysis_button:
            with st.spinner("Performing headline analysis..."):
                result = headline_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Headline Analysis Results:")
                    st.markdown(result)
        
        elif main_headline_analysis_button:
            with st.spinner("Performing Main Headline Analysis..."):
                result = main_headline_detailed_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Main Headline Analysis Results:")
                    st.markdown(result)

        elif image_headline_analysis_button:
            with st.spinner("Performing Image Headline Analysis..."):
                result = image_headline_detailed_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Image Headline Analysis Results:")
                    st.markdown(result)

        elif supporting_headline_analysis_button:
            with st.spinner("Performing Supporting Headline Analysis..."):
                result = supporting_headline_detailed_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Supporting Headline Analysis Report Results:")
                    st.markdown(result)

        elif detailed_headline_analysis_button:
            with st.spinner("Performing Headline Optimization Report analysis..."):
                result = headline_detailed_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Headline Optimization Report Results:")
                    st.markdown(result)
                    
        elif main_headline_text_analysis_button:
            with st.spinner("Performing Main Headline Text Analysis..."):
                result = main_headline_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Main Headline Text Analysis Results:")
                    st.markdown(result)

        elif image_headline_text_analysis_button:
            with st.spinner("Performing Image Headline Text Analysis..."):
                result = image_headline_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Image Headline Text Analysis Results:")
                    st.markdown(result)

        elif supporting_headline_text_analysis_button:
            with st.spinner("Performing Supporting Headline Text Analysis..."):
                result = supporting_headline_analysis(uploaded_file, is_image)
                if result:
                    st.write("## Supporting Headline Text Analysis Results:")
                    st.markdown(result)
        elif meta_profile_button:
            with st.spinner("Performing Meta Analysis..."):
                result = meta_profile(uploaded_file, is_image)
                if result:
                    st.write("## Meta Profile Analysis Results:")
                    st.markdown(result)
        elif linkedin_profile_button:
            with st.spinner("Performing Linkedin profile Analysis..."):
                result = linkedin_profile(uploaded_file, is_image)
                if result:
                    st.write("## Linkedin profile Analysis Results:")
                    st.markdown(result)
        elif x_profile_button:
            with st.spinner("Performing X (formerly Twitter) targeting Analysis..."):
                result = x_profile(uploaded_file, is_image)
                if result:
                    st.write("## X (formerly Twitter) targeting Analysis Results:")
                    st.markdown(result)
        elif personality_trait_assessment_button:
            with st.spinner("Performing Personality Trait Assessment Analysis..."):
                result = Personality_Trait_Assessment(uploaded_file, is_image)
                if result:
                    st.write("## Personality Trait Assessment Analysis Results:")
                    st.markdown(result)
        elif BMTI_Analysis_button:
            with st.spinner("Performing BMTI Analysis..."):
                result = BMTI_Analysis(uploaded_file, is_image)
                if result:
                    st.write("## BMTI Analysis Results:")
                    st.markdown(result)                    
        elif Image_Analysis_button:
            with st.spinner("Performing Image Analysis..."):
                result = Image_Analysis(uploaded_file, is_image)
                if result:
                    st.write("## Image Analysis Results:")
                    st.markdown(result)                     
        # Custom Prompt Analysis
        elif custom_prompt_button and custom_prompt:
            with st.spinner("Performing custom prompt analysis..."):
                result = custom_prompt_analysis(uploaded_file, custom_prompt, is_image)
                if result:
                    st.write("## Custom Prompt Analysis Results:")
                    st.markdown(result)
# Function to compare all images with a standard prompt or custom prompt
def compare_all_images(images, filenames, model, custom_prompt=None):
    # Define the prompt
    if custom_prompt is None:
        # Construct the prompt string with proper concatenation
        image_list_str = '\n'.join([f'- **Image {i+1}:** {filenames[i]}' for i in range(len(images))])
        table_rows = '\n'.join(
            [
                f'| {i+1} | Description: | Statement: | Impact: |\n|   | Suggestions: | Enhancements: | Improvements: |'
                for i in range(len(images))
            ]
        )
        
        prompt = (
            f"Analyze and compare the following {len(images)} marketing images. Focus on their visual elements, "
            f"marketing attributes, and overall effectiveness. The images to be analyzed are:\n\n"
            f"{image_list_str}\n\n"
            "Your analysis should be factual, based solely on visible content, and avoid inferential or speculative details. "
            "Please address the following points:\n\n"
            "1. **Visual Elements**:\n"
            "   - Identify and describe common visual elements across all images (e.g., color schemes, object types, layout, composition styles).\n"
            "   - Highlight unique elements that distinguish each image from the others.\n\n"
            "2. **Marketing Messages**:\n"
            "   - Examine any explicit and implicit marketing messages conveyed in the images.\n"
            "   - Discuss how these messages align with or diverge from the visual elements.\n\n"
            "3. **Comparative Analysis**:\n"
            "   - Assess the relative strengths and weaknesses of each image in a marketing context.\n"
            "   - Consider factors like visual appeal, clarity of message, and potential audience impact.\n\n"
            "4. **Overall Evaluation**:\n"
            "   - Provide a summary of the key findings.\n"
            "   - Highlight the most effective image(s) based on the analysis and justify your choice.\n\n"
            "Structure the results in a detailed table summarizing the key points for each image:\n\n"
            "| Img # | Visual Appeal | Marketing Message | Overall Impact |\n"
            "|-------|---------------|-------------------|----------------|\n"
            f"{table_rows}\n\n"
            "In your response, ensure each section is covered thoroughly with clear, concise points, and present any necessary "
            "improvements or recommendations for each image. Use factual observations to support your analysis."
        )
    else:
        # Use custom prompt if provided
        prompt = custom_prompt

    # Generate content using the model and prompt
    try:
        response = model.generate_content([prompt] + images)
        if response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            st.error("Model did not provide a valid response.")
            return None
    except Exception as e:
        st.error(f"Failed to process the images: {e}")
        return None

# Initialize the Streamlit app
st.title("Marketing Image Comparison AI Assistant with Gemini-2.0-flash")

# File Uploader for Multiple Images
uploaded_files = st.file_uploader(
    "Upload Marketing Images for Comparison (minimum 2, maximum 10):",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"],
    help="Select multiple images for comparison.",
)

# Display Uploaded Images in a Grid (if multiple)
if uploaded_files:
    st.write("## Uploaded Images:")
    st.image([convert_to_rgb(Image.open(file)) for file in uploaded_files], width=200, caption=[f"Image {i + 1}" for i in range(len(uploaded_files))])

# Image Comparison if there are at least 2 uploaded files
if uploaded_files and len(uploaded_files) >= 2:
    # Image Comparison Options
    with st.expander("Image Comparison Options"):

        # Standard Comparison (All Images Together)
        if st.button("Compare All Images Together (Standard)", key="all_images_compare_button"):
            with st.spinner("Comparing all images..."):
                image_list = [convert_to_rgb(Image.open(file)) for file in uploaded_files]
                filenames = [file.name for file in uploaded_files]
                results = compare_all_images(image_list, filenames, model)
                if results:
                    st.write("## Image Comparison Results:")
                    st.markdown(results)

        # Custom Prompt Comparison (All Images Together)
        custom_prompt = st.text_area(
            "Custom Prompt for Comparison (Optional):",
            value="""Provide a detailed comparison of these images, focusing on [your specific areas of interest]. Explain the similarities, differences, and potential impact on the target audience.""",
            height=150
        )

        if st.button("Compare with Custom Prompt", key="all_images_custom_compare_button"):
            with st.spinner("Comparing all images with custom prompt..."):
                image_list = [convert_to_rgb(Image.open(file)) for file in uploaded_files]
                filenames = [file.name for file in uploaded_files]
                results = compare_all_images(image_list, filenames, model, custom_prompt)
                if results:
                    st.write("## Custom Image Comparison Results:")
                    st.markdown(results)

elif uploaded_files and len(uploaded_files) < 2:
    st.warning("Please upload at least two images for comparison.")
