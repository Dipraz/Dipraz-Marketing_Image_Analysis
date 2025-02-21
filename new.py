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
        "max_output_tokens": 30000,
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
### **Objective:**  
Evaluate and analyze the provided image to assess its effectiveness as a marketing asset in capturing audience attention, conveying value, and driving engagement or action. This analysis will be structured, comprehensive, and data-driven, examining visual, textual, and psychological elements to determine their impact on the target audience. The goal is to provide clear, actionable insights for optimizing the asset’s performance, ensuring maximum clarity, engagement, and conversion potential. The response will be fully completed without truncation, delivering a thorough assessment that guides strategic marketing improvements.
---
### **Evaluation Method:**  

✅ **Step 1: Identify Key Asset Details**  
1. **Asset Type** – Classify the marketing asset (e.g., social media post, advertisement, email, flyer, landing page).  
2. **Purpose** – Clearly state the asset's objective (e.g., product promotion, lead generation, brand awareness, customer engagement).  
3. **Target Audience** – Define the demographic, interests, and needs of the intended audience (e.g., age, gender, location, income, behaviors).  

✅ **Step 2: Structured Image Evaluation**  
Each aspect below should be **rated from 1 to 5** (increments of 0.5) with a **detailed explanation and improvement recommendations**. The results must be presented in a **structured table format**.  

| Aspect                     | Score (1-5) | Explanation | Suggested Improvement |  
|----------------------------|------------|-------------|------------------------|  
| **Distinction** | _[Score]_ | Does the image stand out and grab attention? Does it work with/without text? | _[Suggested improvement]_ |  
| **Attention** | _[Score]_ | Is the **content order effective** (headline → text → CTA)? | _[Suggested improvement]_ |  
| **Purpose & Value Clarity** | _[Score]_ | Is the **purpose clear within 3 seconds**? Is it **product/customer-focused**? | _[Suggested improvement]_ |  
| **Clarity** | _[Score]_ | Are the visuals and text **easy to understand**? | _[Suggested improvement]_ |  
| **Creativity** | _[Score]_ | Does the **design use innovative elements** to capture attention? | _[Suggested improvement]_ |  
| **First Impressions** | _[Score]_ | Does the **design create a strong initial impact**? | _[Suggested improvement]_ |  
| **Headline Review** | _[Score]_ | Is the **headline clear, concise, customer-focused, emotionally appealing**? | _[Suggested improvement]_ |  
| **Keyword & Emotional Appeal** | _[Score]_ | Does the headline use **SEO-friendly and emotional keywords**? | _[Suggested improvement]_ |  
| **Visual Cues & Color Usage** | _[Score]_ | Are **colors, contrast, and elements used effectively** to guide attention? | _[Suggested improvement]_ |  
| **Engagement Potential** | _[Score]_ | Does the UX **encourage interaction and retention**? | _[Suggested improvement]_ |  
| **Trustworthiness** | _[Score]_ | Does the content feel **credible, reliable, and customer-centric**? | _[Suggested improvement]_ |  
| **Motivation** | _[Score]_ | Does it align with **user motivators, authority, or social proof**? | _[Suggested improvement]_ |  
| **Influence** | _[Score]_ | Does it effectively **persuade users to take action**? | _[Suggested improvement]_ |  
| **Calls to Action (CTA)** | _[Score]_ | Are the **CTA buttons clear, visible, and compelling**? | _[Suggested improvement]_ |  
| **User Experience (UX)** | _[Score]_ | Does the **design facilitate an easy and enjoyable experience**? | _[Suggested improvement]_ |  
| **Memorability** | _[Score]_ | Does the **design leave a lasting impression**? | _[Suggested improvement]_ |  
| **Text Readability & Effort** | _[Score]_ | Is the text **clear, concise, and skimmable**? | _[Suggested improvement]_ |  
| **Tone Effectiveness** | _[Score]_ | Does the **tone align with brand messaging and audience expectations**? | _[Suggested improvement]_ |  
| **Message Framing** | _[Score]_ | Is the message **strategically framed** to maximize impact? | _[Suggested improvement]_ |  
| **Content Investment** | _[Score]_ | Does it **avoid information overload** while keeping the message strong? | _[Suggested improvement]_ |  

✅ **Step 3: Final Recommendations & Summary**  
- **Overall Score:** _[Weighted Average of All Scores]_  
- **Effectiveness Summary:** _[Does the asset successfully drive engagement and action?]_  
- **Top 3 Areas for Improvement:** _[List key focus areas]_  
- **Final Optimization Strategy:** _[Specific actions to enhance performance]_ 
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
Analyze the provided image for storytelling effectiveness in marketing. Ensure **each aspect is fully evaluated and completed**, without omission or truncation.

### **Part 1: Storytelling Impact Breakdown**
Provide a detailed assessment of how storytelling enhances **static creative content** across the following seven principles:

1. **Emotional Engagement**  
   - **Impact**: How well does the content **evoke emotions** (e.g., happiness, nostalgia, excitement, urgency)?  
   - **Evaluation**: Does the image **connect with the audience on an emotional level**?  
   - **Example**: A family enjoying a product conveys warmth and togetherness.  
   - **Improvement**: How can emotions be **intensified or refined**?  

2. **Attention and Interest**  
   - **Impact**: Does the storytelling element **capture and hold attention**?  
   - **Evaluation**: How **intriguing and engaging** is the content?  
   - **Example**: A before-and-after transformation keeps the viewer invested.  
   - **Improvement**: How can the narrative **increase retention**?  

3. **Memorability**  
   - **Impact**: Does the story help **reinforce brand recall**?  
   - **Evaluation**: Is the storytelling element **easy to remember**?  
   - **Example**: A visual journey of the product's impact on a customer.  
   - **Improvement**: How can the **message be made even more memorable**?  

4. **Brand Identity and Values**  
   - **Impact**: Does the content **reflect the brand’s core values**?  
   - **Evaluation**: How well does the story **align with brand positioning**?  
   - **Example**: Showing the founders working on their first product to highlight authenticity.  
   - **Improvement**: How can the brand’s **storytelling be strengthened**?  

5. **Simplification of Complex Messages**  
   - **Impact**: Does storytelling **make complex ideas easier to understand**?  
   - **Evaluation**: Is the **message clear and accessible**?  
   - **Example**: A simple infographic telling a sustainability story.  
   - **Improvement**: How can the content be **simplified further**?  

6. **Connection and Trust**  
   - **Impact**: Does the storytelling element **establish trust** with the audience?  
   - **Evaluation**: How **authentic and relatable** is the content?  
   - **Example**: Real customer testimonials featuring their journey with the product.  
   - **Improvement**: How can **trust-building elements** be enhanced?  

7. **Call-to-Action (CTA) Effectiveness**  
   - **Impact**: How well does storytelling enhance **CTA performance**?  
   - **Evaluation**: Does the **narrative create urgency or appeal**?  
   - **Example**: A success journey leading to a CTA like **“Join the success”**.  
   - **Improvement**: How can the **CTA be made more compelling**?  

---

### **Part 2: Storytelling Effectiveness Scoring**
Provide a **score from 1 to 5** (in increments of **0.5**) for **each storytelling principle**, along with a detailed **evaluation** and **improvement suggestions**. Format the response in the table below:

| **Storytelling Principle**   | **Score (1-5)** | **Evaluation**                                         | **Improvement**                                      |
|-----------------------------|-----------------|-------------------------------------------------------|------------------------------------------------------|
| **1. Emotional Engagement**  | _1-5_           | How well does the image evoke emotions?                | How can the emotional impact be enhanced?            |
| **2. Attention & Interest**  | _1-5_           | Does the narrative hold attention effectively?        | How can attention retention be improved?            |
| **3. Memorability**          | _1-5_           | Does the story make the brand or product memorable?   | How can memorability be increased?                  |
| **4. Brand Identity & Values** | _1-5_         | Does the image reflect and reinforce brand identity?  | How can storytelling better align with branding?    |
| **5. Simplification of Messages** | _1-5_     | Does the content make complex ideas more digestible? | How can the story be clearer and more engaging?     |
| **6. Connection & Trust**    | _1-5_           | Does the content foster audience trust?               | How can credibility and authenticity be enhanced?   |
| **7. CTA Effectiveness**     | _1-5_           | How persuasive is the CTA in the context of storytelling? | How can the CTA be optimized for stronger impact?   |

---

### **Part 3: Summary and Final Recommendations**
After the table, provide a **final summary** that includes:

1. **Overall Storytelling Effectiveness Score**  
   - Calculate an average score based on all **seven principles**.  

2. **Key Strengths**  
   - Summarize **what is working well** in the content.  

3. **Key Areas for Improvement**  
   - Highlight **specific weaknesses** and how they can be addressed.  

4. **Final Recommendations for Optimization**  
   - Provide **3-5 actionable recommendations** to **enhance storytelling impact**.  

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
If the content is in a non-English language, **translate it into English first**. Then, evaluate the content’s emotional resonance based on the following criteria. 

For each element, **assign a score from 1 to 5** (in increments of **0.5**), provide an **evaluation**, and suggest **specific improvements**. Present the response in a **structured table** with the following columns: **Element, Score, Evaluation, and Improvement Suggestions**. 

At the end, include a **summary** with **overall recommendations** for enhancing emotional appeal.

---

### **Part 1: Emotional Resonance Evaluation**
Evaluate the content using the following **six key criteria**:

#### **1. Clarity of Emotional Appeal**
- **Criteria:** The content clearly conveys the intended emotion(s).  
- **Evaluation:** Is the emotional message easily understood and free from ambiguity?  
- **Example:** A heartfelt message about community support should **clearly** evoke warmth and connection.  
- **Improvement:** How can the emotional message be made clearer?  

#### **2. Relevance to Target Audience**
- **Criteria:** The emotional appeal is **aligned with the audience’s values, interests, and experiences**.  
- **Evaluation:** Does the content **resonate with the personal or professional life** of the audience?  
- **Example:** A financial planning ad should use **emotionally relevant examples** (e.g., “secure your family’s future”).  
- **Improvement:** How can the emotional connection be **deepened for the audience**?  

#### **3. Authenticity**
- **Criteria:** The emotional appeal **feels natural, credible, and sincere**.  
- **Evaluation:** Does the content avoid **exaggeration or forced emotions**?  
- **Example:** A **genuine** testimonial from a real customer builds **trust** better than a scripted message.  
- **Improvement:** How can **authenticity be enhanced**?  

#### **4. Visual and Verbal Consistency**
- **Criteria:** The **visual (images, colors, design) and verbal (language, tone) elements** reinforce the intended emotion.  
- **Evaluation:** Does every element **align to support the emotional appeal**?  
- **Example:** A **calm, reassuring** medical advertisement should **not** use bright red alarming visuals.  
- **Improvement:** How can **visuals and text** be more emotionally aligned?  

#### **5. Emotional Intensity**
- **Criteria:** The emotional response is **appropriate in strength** (not too weak or overwhelming).  
- **Evaluation:** Does the content **evoke emotions at the right intensity for the message**?  
- **Example:** A **charity donation appeal** should be **deeply moving but not manipulative**.  
- **Improvement:** Should the intensity be **dialed up or down**?  

#### **6. Engagement Potential**
- **Criteria:** The content encourages **user engagement** (likes, shares, comments, actions).  
- **Evaluation:** Does the content **invite interaction** and **provide clear engagement opportunities**?  
- **Example:** A **social media post with a compelling question** drives higher engagement.  
- **Improvement:** How can the content be **more engaging and shareable**?  

---

### **Part 2: Structured Table Format**
Provide the evaluation results in **this structured table format**:

| **Element**                    | **Score (1-5)** | **Evaluation**                                         | **Improvement Suggestions**                          |
|--------------------------------|---------------|-------------------------------------------------------|------------------------------------------------------|
| **Clarity of Emotional Appeal** | _1-5_         | How clear is the emotional message?                   | How can clarity be improved?                        |
| **Relevance to Target Audience** | _1-5_        | Does the content emotionally resonate with users?     | How can it be made more relevant?                   |
| **Authenticity**               | _1-5_         | Does the content feel sincere and credible?           | How can it feel more natural and genuine?           |
| **Visual & Verbal Consistency** | _1-5_         | Are visuals and text aligned with the emotion?        | How can they better reinforce the emotional appeal? |
| **Emotional Intensity**        | _1-5_         | Is the emotional response appropriate in strength?    | Should it be amplified or toned down?               |
| **Engagement Potential**        | _1-5_         | Does the content encourage interaction?               | How can engagement be improved?                     |

---

### **Part 3: Final Summary and Recommendations**
At the end, provide a **concise summary** that includes:

1. **Overall Emotional Resonance Score**  
   - Calculate an **average score** across all **six elements**.  

2. **Key Strengths**  
   - Summarize **what works well** in the content.  

3. **Areas for Improvement**  
   - Highlight **specific weaknesses** and how they can be addressed.  

4. **Final Recommendations**  
   - Provide **3-5 clear, actionable suggestions** to **enhance emotional impact**.  
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
If the content is non-English, first translate it into English. Then, evaluate the marketing content based on its emotional resonance, determining whether and how each emotional principle is applied. 

For each emotional resonance type, **present the information in a structured table** with the following columns:  
- **Name** (Emotion being assessed)  
- **Applies** (None, Some, A Lot)  
- **Definition** (Brief explanation of the emotional concept)  
- **How It Is Applied** (How the marketing content currently uses this emotion)  
- **How It Could Be Implemented** (Suggestions for improving or strengthening emotional impact)

---
### **Emotional Resonance Evaluation Criteria**
Assess the **following 15 emotional principles** in the marketing content:

#### **1. Empathy**
- **Definition:** The ability to understand and share the feelings of others.  
- **Evaluation:** Does the content demonstrate understanding of the audience’s emotions and challenges?  
- **Implementation Example:** Craft messages that align with the audience’s struggles and aspirations.  

#### **2. Joy**
- **Definition:** A feeling of great pleasure and happiness.  
- **Evaluation:** Does the content generate excitement, happiness, or amusement?  
- **Implementation Example:** Use vibrant visuals and uplifting messages that make the audience feel positive.  

#### **3. Surprise**
- **Definition:** A feeling of astonishment caused by something unexpected.  
- **Evaluation:** Does the content feature an element of surprise or unpredictability?  
- **Implementation Example:** Introduce unexpected imagery, wording, or promotions to capture attention.  

#### **4. Trust**
- **Definition:** Confidence in the honesty, integrity, and reliability of a brand.  
- **Evaluation:** Does the content establish credibility through transparency and authenticity?  
- **Implementation Example:** Incorporate testimonials, verified facts, and brand consistency to enhance trust.  

#### **5. Fear**
- **Definition:** An unpleasant emotion triggered by perceived threats or risks.  
- **Evaluation:** Does the content highlight dangers, risks, or consequences to drive action?  
- **Implementation Example:** Use fear-based messaging carefully to encourage protective actions (e.g., cybersecurity threats).  

#### **6. Sadness**
- **Definition:** A feeling of sorrow or empathy.  
- **Evaluation:** Does the content evoke sympathy or compassion to engage the audience?  
- **Implementation Example:** Use human-centered storytelling to highlight real-world problems and solutions.  

#### **7. Anger**
- **Definition:** A strong feeling of displeasure or frustration.  
- **Evaluation:** Does the content address injustices or problems that provoke action?  
- **Implementation Example:** Highlight unfair situations and offer solutions to mobilize the audience.  

#### **8. Anticipation**
- **Definition:** Excitement or anxiety about an upcoming event.  
- **Evaluation:** Does the content create excitement for future offerings?  
- **Implementation Example:** Use countdowns, teasers, and sneak peeks to generate anticipation.  

#### **9. Disgust**
- **Definition:** A strong aversion to something unpleasant.  
- **Evaluation:** Does the content leverage repulsion to steer the audience away from undesired alternatives?  
- **Implementation Example:** Contrast low-quality competitors with the benefits of your product.  

#### **10. Relief**
- **Definition:** A sense of reassurance and release from stress.  
- **Evaluation:** Does the content position the product/service as a problem-solver?  
- **Implementation Example:** Use messaging that alleviates concerns (e.g., “Never worry about X again!”).  

#### **11. Love**
- **Definition:** A deep feeling of affection, attachment, or devotion.  
- **Evaluation:** Does the content evoke love for family, relationships, or the brand itself?  
- **Implementation Example:** Show genuine connections between customers and the brand.  

#### **12. Pride**
- **Definition:** A sense of accomplishment or satisfaction.  
- **Evaluation:** Does the content celebrate the audience’s achievements or status?  
- **Implementation Example:** Highlight how the brand contributes to personal success.  

#### **13. Belonging**
- **Definition:** A sense of acceptance and inclusion.  
- **Evaluation:** Does the content create a sense of community?  
- **Implementation Example:** Encourage audience participation and user-generated content.  

#### **14. Nostalgia**
- **Definition:** Sentimental longing for the past.  
- **Evaluation:** Does the content tap into positive memories?  
- **Implementation Example:** Use retro visuals and references to evoke familiarity.  

#### **15. Hope**
- **Definition:** Expectation and desire for a positive outcome.  
- **Evaluation:** Does the content inspire optimism and motivation?  
- **Implementation Example:** Use uplifting messaging about a better future.  

---

### **Structured Table Format**
Provide the evaluation results in the following table format:

| **Name**      | **Applies (None, Some, A Lot)** | **Definition**                         | **How It Is Applied**                          | **How It Could Be Implemented**              |
|--------------|--------------------------------|----------------------------------------|-----------------------------------------------|----------------------------------------------|
| **Empathy**  | _None/Some/A Lot_             | The ability to understand and share emotions. | How the content shows empathy (or lacks it). | Ways to enhance empathetic storytelling.   |
| **Joy**      | _None/Some/A Lot_             | A feeling of happiness and excitement. | Does the content evoke joy?                   | Use uplifting visuals and positive messaging. |
| **Surprise** | _None/Some/A Lot_             | Unexpected elements that grab attention. | Does the content include a surprise factor?  | Add unexpected twists in visuals/text.      |
| **Trust**    | _None/Some/A Lot_             | Confidence in the brand's reliability. | How trust is built (or not).                  | Strengthen credibility through testimonials. |
| **Fear**     | _None/Some/A Lot_             | Highlighting risks to drive action.    | Is fear-based messaging used?                 | Ensure fear messaging remains ethical.      |
| **Sadness**  | _None/Some/A Lot_             | Evoking sympathy and compassion.       | Does the content trigger emotional concern?  | Use real stories to create emotional depth. |
| **Anger**    | _None/Some/A Lot_             | Addressing injustice to spark action.  | How the content presents injustice/issues.   | Focus on solution-driven messaging.         |
| **Anticipation** | _None/Some/A Lot_         | Excitement for upcoming events.        | Is excitement effectively generated?         | Add countdowns and sneak peeks.            |
| **Disgust**  | _None/Some/A Lot_             | Strong aversion to a negative element. | Is disgust used to contrast alternatives?    | Ensure it's used ethically.                 |
| **Relief**   | _None/Some/A Lot_             | Reassurance and stress alleviation.    | How relief is positioned in the content.     | Use calming visuals and messaging.         |
| **Love**     | _None/Some/A Lot_             | Deep emotional affection.              | Is love/emotional attachment conveyed?      | Highlight relationships and brand affinity. |
| **Pride**    | _None/Some/A Lot_             | Satisfaction from achievements.        | Does the content celebrate achievements?     | Showcase user success stories.              |
| **Belonging** | _None/Some/A Lot_           | Feeling of being accepted.             | Does the content create community?          | Foster inclusivity in brand messaging.     |
| **Nostalgia** | _None/Some/A Lot_           | Sentimental longing for the past.      | Are nostalgic elements included?            | Use retro visuals and references.         |
| **Hope**     | _None/Some/A Lot_             | Inspiring optimism for the future.     | Is hope conveyed through the content?       | Use uplifting, forward-thinking messaging. |

---

### **Final Summary and Recommendations**
At the end, provide:
1. **Overall Emotional Resonance Score** (average of applied emotions).  
2. **Key Strengths** (which emotions were most effective).  
3. **Areas for Improvement** (which emotions could be strengthened).  
4. **3-5 Actionable Recommendations** to enhance emotional impact.  
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
If the content is non-English, first translate it into English. Then, analyze the marketing content using the following emotional appraisal models. Provide a structured evaluation for each model, including key findings and suggested improvements. 

For each **emotional appraisal model**, present the results in a **structured table** with the following columns:
- **Model Name** (The emotional appraisal model being applied)
- **Component** (Specific evaluation criteria within the model)
- **Assessment** (How the marketing content aligns with this component)
- **Emotional Response** (The predicted emotional reaction from the target audience)
- **Improvement Suggestions** (Ways to enhance the emotional impact based on the model)

---

### **Emotional Appraisal Models for Evaluation**

#### **1. Lazarus’ Cognitive-Motivational-Relational Theory**
- **Overview:** Emotions result from cognitive evaluations (appraisals) of events that consider personal relevance and coping potential.
- **Key Components:**
  - **Primary Appraisal:** How significant is this event to personal well-being? (Beneficial vs. Harmful)
  - **Secondary Appraisal:** Does the individual have the resources to handle this event?
  - **Core Relational Themes:** Specific appraisals that lead to emotions (e.g., loss → sadness, threat → fear).
- **Marketing Application:** Evaluate if the content aligns with consumers’ concerns and perceived ability to act.

#### **2. Scherer’s Component Process Model (CPM)**
- **Overview:** Emotions result from a sequence of appraisals across multiple dimensions.
- **Key Components:**
  - **Novelty:** Is the event new or unexpected?
  - **Pleasantness:** Is the content enjoyable?
  - **Goal Significance:** Does it help or hinder user goals?
  - **Coping Potential:** Can the audience easily understand and act on the message?
  - **Norm Compatibility:** Does it align with societal norms?
- **Marketing Application:** Assess if the content leverages novelty, ease of engagement, and alignment with audience goals.

#### **3. Smith and Ellsworth’s Appraisal Model**
- **Overview:** Emotional responses are shaped by how people evaluate situations.
- **Key Components:**
  - **Attention:** Does the content draw and sustain focus?
  - **Certainty:** How predictable is the outcome?
  - **Control/Coping:** Can the audience influence the situation?
  - **Pleasantness:** Does the content evoke positive or negative feelings?
  - **Perceived Obstacle:** Does it introduce friction or barriers?
  - **Responsibility:** Who is responsible for the event (brand, user, external factors)?
  - **Anticipated Effort:** How much effort does the audience need to engage?
- **Marketing Application:** Determine if the content is **easy to consume, emotionally appealing, and action-driven**.

#### **4. Roseman’s Appraisal Theory**
- **Overview:** Focuses on whether an event aligns with personal motivations and who/what caused it.
- **Key Components:**
  - **Motivational State:** Does the content align with user desires?
  - **Situational State:** Is it controlled by the brand, audience, or external factors?
  - **Probability:** How likely is the promised outcome?
  - **Agency:** Who is responsible for the message's credibility?
  - **Power/Control:** Does the content make users feel in control?
- **Marketing Application:** Assess whether the messaging empowers the audience or shifts responsibility.

#### **5. Weiner’s Attributional Theory of Emotion**
- **Overview:** Focuses on how people attribute causes to emotions.
- **Key Components:**
  - **Locus:** Is the cause internal (user-driven) or external (brand-driven)?
  - **Stability:** Is the cause stable or temporary?
  - **Controllability:** Can the audience influence the outcome?
- **Marketing Application:** Ensure the content **reinforces consumer confidence and control** over outcomes.

#### **6. Frijda’s Laws of Emotion**
- **Overview:** Describes patterns in how people react emotionally to events.
- **Key Laws:**
  - **Law of Situational Meaning:** Emotions stem from event meanings.
  - **Law of Concern:** Emotional reactions are tied to personal relevance.
  - **Law of Apparent Reality:** Perceived realness determines intensity.
  - **Law of Change:** New stimuli evoke stronger emotions.
  - **Law of Habituation:** Repeated exposure dulls response.
  - **Law of Comparative Feeling:** Emotions are relative to past experiences.
  - **Law of Hedonic Asymmetry:** Negative emotions last longer than positive ones.
  - **Law of Emotional Momentum:** Emotions persist unless altered.
- **Marketing Application:** Assess if the content **creates sustained emotional impact** or fades quickly.

#### **7. Ellsworth’s Model of Appraisal Dimensions**
- **Overview:** Extends appraisal theory to **culture and context**.
- **Key Components:**
  - **Certainty:** Does the content instill confidence in outcomes?
  - **Attention:** Does it capture and sustain focus?
  - **Control:** Does the audience feel empowered?
  - **Pleasantness:** Is the content emotionally appealing?
  - **Responsibility:** Who is accountable for outcomes?
  - **Legitimacy:** Is the message perceived as fair?
- **Marketing Application:** Ensure content is **engaging, ethical, and culturally aligned**.

---

### **Structured Table Format**
For each model, provide insights in the following structured table:

| **Model Name**   | **Component**            | **Assessment**                                     | **Emotional Response**                     | **Improvement Suggestions**                  |
|-----------------|-------------------------|---------------------------------------------------|-------------------------------------------|---------------------------------------------|
| **Lazarus' Theory** | Primary Appraisal       | Does the audience perceive the event as a benefit or threat? | Is the emotion aligned with personal concerns? | Adjust messaging to emphasize benefits.    |
| **CPM**         | Novelty                   | Is the content fresh and attention-grabbing?     | Curiosity, excitement, or indifference   | Introduce unexpected elements.             |
| **Smith & Ellsworth** | Attention             | Does it hold audience focus effectively?         | Engagement or distraction                 | Optimize layout and emphasis.               |
| **Roseman’s Theory** | Motivational State    | Does the content align with user goals?          | Motivation or frustration                 | Align messaging with key audience needs.    |
| **Weiner’s Attribution** | Locus           | Who is seen as responsible for the message?      | Trust or skepticism                      | Clarify brand accountability.               |
| **Frijda’s Laws** | Law of Change           | Is the content dynamic and evolving?             | Emotional persistence or disengagement    | Maintain novelty through variation.         |
| **Ellsworth’s Model** | Legitimacy          | Does the audience perceive the message as fair?  | Trust or resistance                      | Avoid exaggerated claims, ensure honesty.   |

---

### **Final Summary and Recommendations**
At the end of the analysis, provide:
1. **Overall Emotional Impact Score** (average alignment with emotional models).  
2. **Key Strengths** (which emotional triggers were most effective).  
3. **Areas for Improvement** (which emotional triggers could be strengthened).  
4. **3-5 Actionable Strategies** to improve **emotional depth and marketing effectiveness**.
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
Evaluate the provided marketing content using the following **Behavioral Science principles** to determine its effectiveness. 

---

### **Instructions:**
1. **Assessment Table Format:**  
   - For each principle, assess whether the content **applies it fully, partially, or not at all**.
   - Present the analysis in a **structured table** with the following columns:
     - **Applies the Principle** (None, Some, A Lot)
     - **Principle (Description)**
     - **Explanation** (How the content applies—or does not apply—the principle)
     - **How It Could Be Implemented** (Suggestions to strengthen the application of the principle)

2. **Comprehensive Evaluation:**  
   - Ensure **each principle** is reviewed **systematically** for its application within the content.  
   - Identify **missing opportunities** and **recommend specific improvements**.

---

### **Behavioral Science Principles for Assessment**

#### **1. Anchoring**  
- **Definition:** People rely on the first piece of information they encounter when making decisions.  
- **Example:** Displaying a **higher original price** next to a discounted price to make the discount **seem more significant**.

#### **2. Social Proof**  
- **Definition:** People assume that the actions of others indicate the correct choice.  
- **Example:** Showcasing **customer reviews, testimonials, or influencer endorsements** to build trust.

#### **3. Scarcity**  
- **Definition:** Perceived scarcity increases desirability.  
- **Example:** Using **"limited time offer"** or **"only a few left in stock"** to create urgency.

#### **4. Reciprocity**  
- **Definition:** People feel obligated to return favors.  
- **Example:** Offering a **free sample, trial, or exclusive content** to encourage future engagement.

#### **5. Loss Aversion**  
- **Definition:** People prefer **avoiding losses** over gaining equivalent rewards.  
- **Example:** Emphasizing **what customers will lose** if they do not act now.

#### **6. Commitment and Consistency**  
- **Definition:** Once people commit, they are likely to follow through.  
- **Example:** Encouraging **small commitments** (like signing up for a newsletter) before a larger one.

#### **7. Authority**  
- **Definition:** People trust figures of **expertise and authority**.  
- **Example:** Featuring **endorsements from industry experts** or statistics.

#### **8. Framing**  
- **Definition:** The way information is presented influences decisions.  
- **Example:** Presenting a price as **"Only $1 per day"** instead of **"$30 per month"**.

#### **9. Endowment Effect**  
- **Definition:** People **overvalue** things they feel ownership over.  
- **Example:** Allowing customers to **try a product before buying**.

#### **10. Priming**  
- **Definition:** Prior exposure influences subsequent decisions.  
- **Example:** Using **positive language and visuals** to **shape perception**.

#### **11. Decoy Effect**  
- **Definition:** Introducing a third **less attractive** option makes another option **more desirable**.  
- **Example:** Adding a **premium pricing tier** to make the **mid-tier option** more attractive.

#### **12. Default Effect**  
- **Definition:** People stick with the **default option** presented.  
- **Example:** Making a **popular or high-margin product the default selection**.

#### **13. Availability Heuristic**  
- **Definition:** People judge likelihood based on **how easily examples come to mind**.  
- **Example:** Highlighting **recent success stories** to reinforce credibility.

#### **14. Cognitive Dissonance**  
- **Definition:** People experience discomfort when holding conflicting beliefs, leading to **attitude changes**.  
- **Example:** Reinforcing **positive aspects of a purchase** to **reduce buyer's remorse**.

#### **15. Emotional Appeal**  
- **Definition:** **Emotions** significantly influence decision-making.  
- **Example:** Using **storytelling, emotional imagery, and relatable messaging**.

#### **16. Bandwagon Effect**  
- **Definition:** People are **more likely** to do something if **others are doing it**.  
- **Example:** Highlighting **high sales numbers, viral trends, or social media mentions**.

#### **17. Frequency Illusion (Baader-Meinhof Phenomenon)**  
- **Definition:** Once people **notice something**, they **see it more frequently**.  
- **Example:** Increasing **brand exposure through multiple channels**.

#### **18. In-Group Favoritism**  
- **Definition:** People prefer **products that align with their identity groups**.  
- **Example:** **Targeted messaging** toward specific **demographics, cultures, or communities**.

#### **19. Hyperbolic Discounting**  
- **Definition:** People prefer **immediate rewards** over **delayed benefits**.  
- **Example:** Offering **instant discounts or rewards for immediate action**.

#### **20. Paradox of Choice**  
- **Definition:** Too many choices **overwhelm** consumers, leading to **decision paralysis**.  
- **Example:** **Simplifying choices** with **curated selections or recommended options**.

---

### **Structured Table Format**
For each principle, the response should be structured as follows:

| **Applies the Principle** | **Principle (Description)** | **Explanation** | **How It Could Be Implemented** |
|-----------------|---------------------------------|-----------------------------------------|----------------------------------------------|
| **None / Some / A Lot** | **Anchoring** (The first info encountered affects decisions) | The content does/does not emphasize a strong initial anchor (e.g., pricing, comparison) | Introduce price anchoring or reference points |
| **None / Some / A Lot** | **Social Proof** (People follow the crowd) | There are / are not customer reviews, testimonials, or influencer endorsements | Increase visible user-generated content |
| **None / Some / A Lot** | **Scarcity** (Limited availability increases desire) | The content does/does not create urgency with time-limited offers or stock scarcity | Add countdown timers or scarcity messaging |
| **None / Some / A Lot** | **Reciprocity** (Giving first increases the chance of return) | There is / is not a free sample, trial, or incentive | Introduce lead magnets or free trials |
| **None / Some / A Lot** | **Loss Aversion** (Avoiding loss is more motivating than gaining) | The content does/does not emphasize potential losses | Frame messaging around “Don’t miss out” scenarios |
| **None / Some / A Lot** | **Emotional Appeal** (Feelings drive decisions) | The content does/does not incorporate strong emotional storytelling | Strengthen visuals, relatable scenarios |

---

### **Final Summary & Recommendations**
After the **principle-by-principle** evaluation, provide:
1. **Overall Behavioral Science Score** (How well the content applies psychological persuasion principles).
2. **Key Strengths** (Which behavioral principles were effectively applied).
3. **Areas for Improvement** (Which principles were underutilized or missing).
4. **3-5 Actionable Strategies** to **enhance engagement and persuasion**.
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
**Objective:**  
Evaluate the **marketing content** using **Neuro-Linguistic Programming (NLP) techniques** to determine its effectiveness in persuasion and engagement.  
---
### **Instructions:**  
1. **Assessment Table Format:**  
   - Analyze whether the **marketing content** applies each NLP principle **fully, partially, or not at all**.  
   - Present the findings in a **structured table** with the following columns:  
     - **Applies the Principle** (None, Some, A Lot)  
     - **Principle (Description)**  
     - **Explanation** (How the content applies—or does not apply—the principle)  
     - **How It Could Be Implemented** (Recommendations to strengthen the application)  

2. **Comprehensive Evaluation:**  
   - Review **each NLP principle systematically** to identify **strengths, weaknesses, and opportunities for improvement**.  
   - Offer **practical and specific** recommendations for **enhancing engagement and persuasion**.  
---
### **Neuro-Linguistic Programming (NLP) Techniques for Assessment**  

#### **1. Representational Systems**  
- **Definition:** People process information visually, auditorily, or kinesthetically.  
- **Example:** If the target audience prefers visual information, **use strong imagery, colors, and graphics**.  

#### **2. Anchoring**  
- **Definition:** Associating emotions or experiences with specific stimuli.  
- **Example:** **Consistently using brand colors, logos, or music** to create **familiarity and trust**.  

#### **3. Meta-Modeling**  
- **Definition:** Challenging vague statements to enhance clarity and precision.  
- **Example:** Instead of **"Best product available"**, say **"Rated #1 by Consumer Reports for durability and performance."**  

#### **4. Milton Model**  
- **Definition:** Using ambiguous and persuasive language to engage subconscious decision-making.  
- **Example:** **"You may start to feel more confident using this product..."** (open-ended suggestion).  

#### **5. Chunking**  
- **Definition:** Adjusting the level of detail based on audience preference.  
- **Example:** **Providing high-level benefits first**, then offering **detailed product specifications** for those seeking more information.  

#### **6. Pacing and Leading**  
- **Definition:** Establishing rapport by aligning with the audience’s current mindset before introducing new ideas.  
- **Example:** **"Do you struggle with productivity? Our planner can help you stay on track effortlessly."**  

#### **7. Swish Pattern**  
- **Definition:** Replacing negative associations with positive ones.  
- **Example:** **Using imagery of an untidy desk transitioning to an organized workspace** to **imply transformation and improvement**.  

#### **8. Submodalities**  
- **Definition:** Adjusting sensory details to create stronger emotional impact.  
- **Example:** **Using high-contrast colors and larger fonts for key messages** to ensure visibility and urgency.  

#### **9. Perceptual Positions**  
- **Definition:** Viewing content from different perspectives (self, others, observer).  
- **Example:** **"Imagine how much time you will save…"** (self), **"Your colleagues will admire your organization"** (others), **"Experts recommend this approach"** (observer).  

#### **10. Well-Formed Outcomes**  
- **Definition:** Ensuring the marketing message clearly defines **a goal, path, and achievable result**.  
- **Example:** **"Increase your productivity by 20% within 30 days using our planner."**  

#### **11. Rapport Building**  
- **Definition:** Aligning language, values, and emotions with the audience.  
- **Example:** **Using language that resonates with the audience’s lifestyle or professional challenges.**  

#### **12. Calibration**  
- **Definition:** Adjusting content based on **user engagement metrics and feedback**.  
- **Example:** **Monitoring engagement rates and refining messaging for better clarity and impact.**  

#### **13. Reframing**  
- **Definition:** Changing perception by presenting a new perspective.  
- **Example:** **"Traffic isn’t wasted time—it’s an opportunity to learn through our podcast!"**  

#### **14. Logical Levels**  
- **Definition:** Addressing audience needs at different levels:  
  - **Environment** – Where they interact with the product (**"Work from anywhere"**).  
  - **Behavior** – How they use it (**"Stay focused and organized"**).  
  - **Identity** – What it means about them (**"Be a highly effective professional"**).  

#### **15. Timeline Therapy**  
- **Definition:** Presenting past, present, and future outcomes to shape perception.  
- **Example:** **"Thousands have already improved their workflow, and now it's your turn!"**  

#### **16. Meta Programs**  
- **Definition:** Understanding **whether the audience is motivated toward goals or away from problems**.  
- **Example:** **"Achieve your dream productivity levels"** (toward) vs. **"Eliminate distractions and wasted time"** (away from).  

#### **17. Strategy Elicitation**  
- **Definition:** Mapping **step-by-step user decisions** to align with their thinking process.  
- **Example:** **"Step 1: Sign up. Step 2: Select your goals. Step 3: Achieve success."**  

#### **18. Sensory Acuity**  
- **Definition:** Enhancing sensory appeal to **increase emotional engagement**.  
- **Example:** **"Feel the soft leather, see the rich color, and hear the precision of our product."**  

#### **19. Pattern Interrupts**  
- **Definition:** Using unexpected elements to **capture attention and break habitual thinking patterns**.  
- **Example:** **"Most planners don’t work—here’s why ours does."**  

#### **20. Belief Change Techniques**  
- **Definition:** Overcoming audience skepticism by **challenging limiting beliefs**.  
- **Example:** **Showcasing case studies or testimonials to counter doubts.**  

---

### **Structured Table Format**
The response should be formatted as follows:

| **Applies the Principle** | **Principle (Description)** | **Explanation** | **How It Could Be Implemented** |
|-----------------|---------------------------------|-----------------------------------------|----------------------------------------------|
| **None / Some / A Lot** | **Anchoring** (Creating strong brand associations) | The content does/does not reinforce brand recognition through colors, logos, or repeated messaging. | Introduce visual consistency in branding. |
| **None / Some / A Lot** | **Meta-Modeling** (Enhancing clarity) | The content does/does not use vague claims like "Best on the market" without evidence. | Add specific data or credibility indicators. |
| **None / Some / A Lot** | **Pacing and Leading** (Aligning with user experiences) | The content does/does not connect with audience struggles before presenting a solution. | Start by identifying common audience pain points. |
| **None / Some / A Lot** | **Belief Change Techniques** (Challenging skepticism) | The content does/does not address potential objections with proof. | Include customer testimonials or expert validation. |

---
### **Final Summary & Recommendations**
After analyzing the content through **NLP principles**, provide:
1. **Overall NLP Effectiveness Score** (How well the content applies persuasion techniques).  
2. **Key Strengths** (Which NLP techniques are effectively used).  
3. **Areas for Improvement** (Which NLP techniques are missing or underutilized).  
4. **3-5 Actionable Strategies** for **enhancing engagement and impact**.
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
### **Objective:**  
Analyze the **text content** of a marketing asset (**image or video, excluding the headline**) to assess its effectiveness in clarity, engagement, persuasion, and relevance. Provide actionable recommendations for improvement.
---
### **Instructions:**  

#### **Part 1: Text Extraction and Contextualization**  

- **Image Analysis:**  
  1. **Text Extraction:** Identify and extract **ALL visible text** in the image, including:  
     - Body copy  
     - Captions  
     - Calls to action  
     - Taglines  
     - Logo text  
     - Any other textual elements  
  2. **Translation:** If the text is **not in English**, first **translate it accurately into English**.  
  3. **Presentation:** Display the extracted text in **a structured, bulleted list** maintaining the **original order and structure** as closely as possible.  
  4. **Visual Analysis:**  
     - **Placement:** Specify where each text element is located (e.g., **top left, center, bottom right**). Identify **overlapping text** or **visual clutter** that may impact readability.  
     - **Font Choices:** Describe the **font type** (serif, sans-serif, script), **weight** (bold, regular, light), **size**, and **color** of each text element.  
     - **Text & Visual Interactions:** Explain how the **text integrates with other visual elements** (images, graphics, colors) and whether it **supports** or **detracts from** the overall message.  

- **Video Analysis:**  
  1. **Key Frame Identification:** Select the **most representative frame(s)** that showcase **primary text content**.  
  2. **Text Extraction:** Extract and present the **visible text** from these frames in a **structured, bulleted list**.  
  3. **Temporal Analysis:** Summarize how **text elements change throughout the video**, noting key **timing patterns** or **animated effects** that influence visibility.  
  4. **Integration with Visuals & Audio:** Analyze how the **text interacts with visuals (scenes, characters, actions)** and **audio elements (dialogue, music, sound effects)**.  

---

### **Part 2: Textual Assessment**  

**Evaluation Criteria:**  
- **For each aspect**, assign a **score from 1 (poor) to 5 (excellent)** in **0.5 increments**.  
- Provide a **concise justification** highlighting **strengths, weaknesses, and opportunities for improvement**.  
- Offer **clear, specific recommendations** to **enhance readability, engagement, and conversion potential**.  
- Structure the analysis in **a table format** with the following columns:  

| **Aspect**                 | **Score** | **Explanation** (Strengths & Weaknesses) | **Improvement** (Specific Actionable Steps) |
|----------------------------|----------|-----------------------------------------|---------------------------------------------|
| **Clarity & Conciseness**  | _1-5_    | How easy is the text to understand? Are sentences clear and well-structured? | Simplify language, remove jargon, shorten sentences for better clarity. |
| **Customer Focus**         | _1-5_    | Does the text **address audience needs** and **resonate with their preferences**? | Adjust messaging to make it **more customer-centric**, emphasizing **pain points and solutions**. |
| **Engagement**             | _1-5_    | How **compelling** is the text? Does it use storytelling, humor, or persuasive techniques? | Use **stronger verbs, storytelling, or a more dynamic tone** to captivate the reader. |
| **Reading Effort**         | _1-5_    | Is the text **easy to read and process** for the average user? | Use **simpler vocabulary and sentence structures** for better accessibility. |
| **Purpose & Value**        | _1-5_    | Does the text **clearly communicate** its **purpose and value** to the audience? | Make the **value proposition** more **explicit and compelling**. |
| **Motivation & Persuasion**| _1-5_    | How effectively does the text **persuade users**? Are there **strong CTAs or social proof**? | Strengthen **calls to action**, use **authority indicators** (reviews, statistics). |
| **Depth & Detail**         | _1-5_    | Is the text **too vague** or does it provide **enough information** for decision-making? | Adjust text length by **adding missing details** or **removing unnecessary content**. |
| **Trustworthiness**        | _1-5_    | Does the text build **credibility** through language, tone, and factual accuracy? | Use **transparent, informative language** to **reinforce authenticity**. |
| **Memorability**           | _1-5_    | Is the text **unique, catchy, or distinctive**? Does it include **memorable elements**? | Incorporate **catchy phrases, unique brand messaging**, or **a storytelling approach**. |
| **Emotional Appeal**       | _1-5_    | Does the text **evoke the right emotions** aligned with the brand’s message? | Refine **word choice and tone** to evoke stronger **emotional connections**. |
| **Uniqueness & Differentiation** | _1-5_ | Does the text make the brand stand out from competitors? | Enhance **brand personality**, avoid **generic phrasing**. |
| **Urgency & Curiosity**    | _1-5_    | Does the text create **a sense of urgency or intrigue**? | Use **time-sensitive phrases** or **engaging hooks** to prompt action. |
| **Benefit Orientation**    | _1-5_    | Does the text focus on **benefits** rather than just **features**? | Make benefits **more explicit and customer-focused**. |
| **Target Audience Relevance** | _1-5_ | Does the text use the **right language, tone, and style** for the intended audience? | Adjust phrasing to **match audience preferences and expectations**. |

---

### **Final Summary & Recommendations**  

At the end of the evaluation, **provide a well-structured summary** with:  
1. **Overall Effectiveness Score** (based on the average of all individual scores).  
2. **Key Strengths:** A **summary of what works well** in the text.  
3. **Areas for Improvement:** The **top 3-5 areas** where the text could be enhanced.  
4. **Action Plan:** A **concise, prioritized list of recommendations** to optimize readability, engagement, and persuasion.
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
### **Objective:**  
Analyze the provided **image content** for **textual, structural, visual, and compliance-related aspects** to ensure its marketing effectiveness. Evaluate and provide actionable recommendations for improvement.
---

### **Instructions:**  

#### **Step 1: Translation (if applicable)**  
- If the content is **not in English**, first **translate it accurately into English** before proceeding with the analysis.
---

### **Step 2: Evaluation Criteria**  
- **Each aspect** should be assigned a **score from 1 (poor) to 5 (excellent)** in **0.5 increments**.  
- Provide a **concise explanation** of strengths and weaknesses.  
- Offer **specific, actionable recommendations** to enhance **clarity, engagement, and effectiveness**.  
- **Results should be structured in a table format** with the following columns:  

| **Aspect**                     | **Score** | **Explanation** (Strengths & Weaknesses) | **Improvement** (Specific Actionable Steps) |
|----------------------------|----------|-----------------------------------------|---------------------------------------------|

---

### **Step 3: Detailed Image Analysis**  

#### **1. Textual Analysis**  
- **Readability Analysis:**  
  - Measure readability using the **Flesch-Kincaid readability test** to determine how **easy or difficult** the content is to read.  
  - Ensure that the language is appropriate for the **target audience's education and literacy level**.  

- **Lexical Diversity:**  
  - Evaluate the **variety of words used** in the content.  
  - **Higher diversity** can indicate **richness in language** but may reduce clarity.  
  - **Lower diversity** may be **simpler and clearer** but risks sounding repetitive.  

#### **2. Semantic Analysis**  
- **Keyword Analysis:**  
  - Identify **key terms related to the brand or product** and analyze their **frequency, prominence, and integration**.  
  - Ensure **strategic placement** of keywords for **SEO and brand messaging clarity**.  

- **Topic Modeling:**  
  - Use techniques like **Latent Dirichlet Allocation (LDA)** to identify the **main themes of the content**.  
  - Assess if these topics align with the **intended marketing message and brand strategy**.  

#### **3. Sentiment Analysis**  
- **Polarity Assessment:**  
  - Use **Natural Language Processing (NLP)** to categorize the sentiment as **positive, neutral, or negative**.  
  - Ensure the **tone aligns with the brand’s intended emotional impact**.  

- **Emotion Detection:**  
  - Beyond polarity, analyze for **specific emotions** conveyed (e.g., **joy, anger, excitement, sadness**).  
  - Assess if the content **evokes the desired emotions in the target audience**.  

#### **4. Structural Analysis**  
- **Narrative Structure:**  
  - Does the text follow a logical sequence (e.g., **introduction → problem statement → solution → CTA**)?
  - Ensure the **message is easy to follow and compelling**.  

- **Visual Composition Analysis:**  
  - Analyze the **layout, typography, color scheme, and imagery**.  
  - Ensure alignment with **brand guidelines** and **aesthetic best practices**.  

#### **5. Linguistic Style Matching**  
- **Consistency with Brand Voice:**  
  - Does the content match the **brand’s tone, style, and personality**?  
  - Ensure **terminology and phrasing are aligned with prior brand communications**.  

- **Grammar & Syntax Analysis:**  
  - Check for **grammatical errors, awkward phrasing, and sentence structure**.  
  - Ensure the text is **polished and professional**.  

#### **6. Cohesion & Coherence Analysis**  
- **Cohesion Metrics:**  
  - Evaluate how **well different parts of the text link together** for **logical consistency**.  
  - Use tools like **Coh-Metrix** to assess readability and content cohesion.  

- **Logical Flow:**  
  - Ensure the **progression of ideas** makes **sense** and **guides the reader effectively**.  

#### **7. Visual & Multimodal Analysis**  
- **Image & Text Alignment:**  
  - Assess how **text elements interact with visuals** (e.g., are they **complementary or distracting**?).  
  - Ensure **proper contrast, positioning, and readability**.  

- **Aesthetic Quality:**  
  - Evaluate design elements like **balance, symmetry, typography, and color harmony**.  
  - Ensure the image is **visually appealing and professional**.  

#### **8. Compliance & Ethical Analysis**  
- **Regulatory Compliance:**  
  - Verify that the content adheres to **advertising regulations and industry standards**.  
  - Ensure there are **no misleading claims, false advertising, or unverified testimonials**.  

- **Ethical Considerations:**  
  - Assess whether the content is **culturally sensitive, non-offensive, and inclusive**.  
  - Ensure there are **no stereotypes, biases, or potentially harmful messaging**.  
---
### **Final Summary & Recommendations**  

At the end of the evaluation, **provide a well-structured summary** with:  

1. **Overall Effectiveness Score** (average of all individual scores).  
2. **Key Strengths:** A **summary of what works well** in the text and design.  
3. **Areas for Improvement:** The **top 3-5 areas** that require enhancement.  
4. **Action Plan:** A **concise, prioritized list of recommended changes** to optimize clarity, engagement, and marketing impact.
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
### **Objective:**  
Analyze the provided **image content** for **textual, semantic, sentiment, structural, and compliance-related aspects** to ensure its marketing effectiveness. Evaluate and provide **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Translation (if applicable)**  
- If the content is **not in English**, first **translate it accurately into English** before proceeding with the analysis.

---

### **Step 2: Evaluation Criteria**  
- **Each aspect** should be assigned a **score from 1 (poor) to 5 (excellent)** in **0.5 increments**.  
- Provide a **concise explanation** of strengths and weaknesses.  
- Offer **specific, actionable recommendations** to enhance **clarity, engagement, and effectiveness**.  
- **Results should be structured in a table format** with the following columns:  

| **Aspect**                     | **Score** | **Explanation** (Strengths & Weaknesses) | **Improvement** (Specific Actionable Steps) |
|----------------------------|----------|-----------------------------------------|---------------------------------------------|

---

### **Step 3: Detailed Image Analysis**  

#### **1. Textual Analysis**  
- **Readability Analysis:**  
  - Evaluate readability using **Flesch-Kincaid readability tests** to determine how **easy or difficult** the content is to read.  
  - Ensure that the language is appropriate for the **target audience's literacy level and cognitive load**.  

- **Lexical Diversity:**  
  - Analyze the **variety of words used** in the content.  
  - **Higher diversity** can indicate **richness in language**, enhancing engagement.  
  - **Lower diversity** may be **simpler and clearer**, reducing cognitive effort.  

#### **2. Semantic Analysis**  
- **Keyword Analysis:**  
  - Identify **key terms related to the brand or product** and analyze their **frequency, prominence, and integration**.  
  - Ensure **strategic placement** of keywords for **SEO and brand messaging clarity**.  

- **Topic Modeling:**  
  - Use techniques like **Latent Dirichlet Allocation (LDA)** to identify the **main themes of the content**.  
  - Assess if these topics align with the **intended marketing message and brand strategy**.  

#### **3. Sentiment Analysis**  
- **Polarity Assessment:**  
  - Use **Natural Language Processing (NLP)** to categorize the sentiment as **positive, neutral, or negative**.  
  - Ensure the **tone aligns with the brand’s intended emotional impact**.  

- **Emotion Detection:**  
  - Beyond polarity, analyze for **specific emotions** conveyed (e.g., **joy, anger, excitement, sadness**).  
  - Assess if the content **evokes the desired emotions in the target audience**.  

#### **4. Structural Analysis**  
- **Narrative Structure:**  
  - Does the text follow a logical sequence (e.g., **introduction → problem statement → solution → CTA**)?
  - Ensure the **message is clear, compelling, and actionable**.  

- **Visual Composition Analysis:**  
  - Analyze the **layout, typography, color scheme, and imagery**.  
  - Ensure alignment with **brand guidelines** and **aesthetic best practices**.  

#### **5. Linguistic Style Matching**  
- **Consistency with Brand Voice:**  
  - Does the content match the **brand’s tone, style, and personality**?  
  - Ensure **terminology and phrasing align with prior brand communications**.  

- **Grammar & Syntax Analysis:**  
  - Check for **grammatical errors, awkward phrasing, and sentence structure**.  
  - Ensure the text is **polished, professional, and free of ambiguity**.  

#### **6. Cohesion & Coherence Analysis**  
- **Cohesion Metrics:**  
  - Evaluate how **well different parts of the text link together** for **logical consistency**.  
  - Use tools like **Coh-Metrix** to assess readability and content cohesion.  

- **Logical Flow:**  
  - Ensure the **progression of ideas** makes **sense** and **guides the reader effectively**.  

#### **7. Visual & Multimodal Analysis**  
- **Image & Text Alignment:**  
  - Assess how **text elements interact with visuals** (e.g., are they **complementary or distracting**?).  
  - Ensure **proper contrast, positioning, and readability**.  

- **Aesthetic Quality:**  
  - Evaluate design elements like **balance, symmetry, typography, and color harmony**.  
  - Ensure the image is **visually appealing, professional, and optimized for engagement**.  

#### **8. Compliance & Ethical Analysis**  
- **Regulatory Compliance:**  
  - Verify that the content adheres to **advertising regulations and industry standards**.  
  - Ensure there are **no misleading claims, false advertising, or unverified testimonials**.  

- **Ethical Considerations:**  
  - Assess whether the content is **culturally sensitive, non-offensive, and inclusive**.  
  - Ensure there are **no stereotypes, biases, or potentially harmful messaging**.  

---

### **Final Summary & Recommendations**  

At the end of the evaluation, **provide a structured summary** with:  

1. **Overall Effectiveness Score** (average of all individual scores).  
2. **Key Strengths:** A **summary of what works well** in the text and design.  
3. **Areas for Improvement:** The **top 3-5 areas** that require enhancement.  
4. **Action Plan:** A **concise, prioritized list of recommended changes** to optimize clarity, engagement, and marketing impact.  
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
### **Objective:**  
Analyze the **headline text** of a **marketing asset (image or video)** to assess its **effectiveness, clarity, relevance, and audience impact**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Headline Extraction & Context**  
For the given **image or video**, extract and identify the **headline elements**:  

1. **Main Headline:** Clearly extract the primary headline from the image or video.  
2. **Image Headline (if applicable):** If a **distinct image-specific** headline exists, state it separately.  
3. **Supporting Headline (if applicable):** Extract any **additional supporting headlines** contributing to the message.  

Ensure accurate **text extraction and formatting**, and **translate non-English text into English** for consistent evaluation.

---

### **Step 2: Headline Analysis & Evaluation**  
For each headline type (**Main, Image, and Supporting**), **analyze and rate the effectiveness** based on the **image/video content** using a **structured table format**.  

- **Each criterion should be rated from 1 (poor) to 5 (excellent) in increments of 0.5.**  
- Provide a **detailed explanation** supporting the score.  
- Offer **specific, actionable recommendations** for improvement.  

#### **2A: Main Headline Analysis**  
Evaluate the **main headline's effectiveness** considering its **synergy with the image/video content**.  

| **Criterion**           | **Score (1-5)** | **Explanation** (Strengths & Weaknesses) | **Recommendation** (Specific Actionable Steps) |
|-------------------------|----------------|-----------------------------------------|---------------------------------------------|
| **Overall Effectiveness** |                | Summarize how well the headline works overall. | Suggest general refinements. |
| **Clarity**             |                | How clearly does the headline convey its message? | Recommend improvements for readability and simplicity. |
| **Customer Focus**      |                | Does the headline prioritize customer needs? | Adjust wording to better engage the audience. |
| **Relevance**          |                | Does it accurately reflect the **image/video content**? | Suggest alignment strategies with visuals. |
| **Keywords**          |                | Are relevant keywords naturally included? | Improve SEO or messaging impact. |
| **Emotional Appeal**  |                | Does it evoke curiosity or an emotional response? | Strengthen emotional triggers. |
| **Uniqueness**        |                | How original and distinctive is the headline? | Offer ways to differentiate from competitors. |
| **Urgency & Curiosity** |                | Does it create a compelling reason to act? | Enhance urgency with stronger phrasing. |
| **Benefit-Driven**     |                | Does it clearly communicate a value proposition? | Rework wording for direct benefits. |
| **Target Audience Fit** |                | Is it tailored to the intended demographic? | Adjust tone, language, or phrasing. |
| **Length & Format**    |                | Does it fall within an optimal 6-12 word range? | Suggest modifications for ideal length. |

---

#### **2B: Image Headline Analysis**  
Evaluate the **headline specifically related to the image content** using the same structure:

| **Criterion**           | **Score (1-5)** | **Explanation** | **Recommendation** |
|-------------------------|----------------|-----------------|---------------------|
| **Overall Effectiveness** |                |                 |                     |
| **Clarity**             |                |                 |                     |
| **Customer Focus**      |                |                 |                     |
| **Relevance**          |                |                 |                     |
| **Keywords**          |                |                 |                     |
| **Emotional Appeal**  |                |                 |                     |
| **Uniqueness**        |                |                 |                     |
| **Urgency & Curiosity** |                |                 |                     |
| **Benefit-Driven**     |                |                 |                     |
| **Target Audience Fit** |                |                 |                     |
| **Length & Format**    |                |                 |                     |

---

#### **2C: Supporting Headline Analysis**  
Analyze the **effectiveness of any supporting text** in relation to the **image/video content**:

| **Criterion**           | **Score (1-5)** | **Explanation** | **Recommendation** |
|-------------------------|----------------|-----------------|---------------------|
| **Overall Effectiveness** |                |                 |                     |
| **Clarity**             |                |                 |                     |
| **Customer Focus**      |                |                 |                     |
| **Relevance**          |                |                 |                     |
| **Keywords**          |                |                 |                     |
| **Emotional Appeal**  |                |                 |                     |
| **Uniqueness**        |                |                 |                     |
| **Urgency & Curiosity** |                |                 |                     |
| **Benefit-Driven**     |                |                 |                     |
| **Target Audience Fit** |                |                 |                     |
| **Length & Format**    |                |                 |                     |

---

### **Step 3: Improved Headline Suggestions**  
After completing the evaluation, propose **three alternative headline suggestions** for each headline type (**Main, Image, Supporting**), along with a **justification for each revision**.

| **Headline Type** | **Headline Recommendation** | **Explanation (Why This Works Better)** |
|------------------|----------------------------|----------------------------------------|
| **Main Headline** | Suggested alternative 1 | Justification for change |
|                  | Suggested alternative 2 | Justification for change |
|                  | Suggested alternative 3 | Justification for change |
| **Image Headline** | Suggested alternative 1 | Justification for change |
|                  | Suggested alternative 2 | Justification for change |
|                  | Suggested alternative 3 | Justification for change |
| **Supporting Headline** | Suggested alternative 1 | Justification for change |
|                  | Suggested alternative 2 | Justification for change |
|                  | Suggested alternative 3 | Justification for change |

---

### **Final Summary & Recommendations**  

1. **Overall Headline Effectiveness Score** (average of all individual headline scores).  
2. **Key Strengths:** Summarize what **works well** across the headlines.  
3. **Top Areas for Improvement:** Identify **critical weaknesses** and how they can be addressed.  
4. **Action Plan:** Provide **concrete steps** to enhance clarity, engagement, and marketing impact.
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
### **Objective:**  
Assess the **effectiveness, clarity, and impact** of the **headline text** in a **marketing asset (image or video)** by evaluating key linguistic and marketing criteria. Provide structured insights and **actionable recommendations** for optimization.
---
### **Instructions:**  

#### **Step 1: Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the different headline elements**:  

1. **Main Headline:** The primary headline conveys the core message.  
2. **Image Headline (if applicable):** A separate **image-specific** headline.  
3. **Supporting Headline (if applicable):** Any **additional text elements** that contribute to the message.  

Ensure **text extraction accuracy**, maintain the **original structure**, and **translate non-English text into English** before analysis.

---

### **Step 2: Headline Optimization Analysis**  
Each headline type (**Main, Image, and Supporting**) should be **analyzed independently** based on the provided **image/video content** using a **structured table format**.

- **Each criterion should be assessed quantitatively and qualitatively.**  
- Provide a **clear explanation** for the assessment.  
- Offer **specific, actionable recommendations** for improvement.  

---

### **Part 1A: Main Headline Optimization Analysis**  
"Analyze the **main headline** in relation to the image content. Evaluate the following criteria and provide an **explanation and recommendations** in a structured table format."

| **Criterion**            | **Assessment** (Quantitative/Qualitative) | **Explanation** (Strengths & Weaknesses) | **Recommendation** (Actionable Steps) |
|--------------------------|--------------------------------|----------------------------------|---------------------------------|
| **Word Count**           | [X words] | Does the length optimize clarity and impact? | Suggest ideal length adjustments. |
| **Keyword Relevance**    | [High/Moderate/Low] | Are relevant keywords naturally incorporated? | Improve keyword placement or integration. |
| **Common Words**         | [X words] | Do frequently used words enhance or weaken clarity? | Adjust word choice for precision and engagement. |
| **Uncommon Words**       | [X words] | Do rare words add uniqueness or create confusion? | Optimize for readability and impact. |
| **Power Words**          | [X words] | Does the headline include persuasive words? | Strengthen headline with emotionally compelling language. |
| **Emotional Words**      | [X words] (Positive/Negative/Neutral) | Does the language evoke the right emotions? | Adjust to align with the intended emotional impact. |
| **Sentiment**            | [Positive/Negative/Neutral] | Does the sentiment align with the marketing goal? | Modify wording to enhance positivity or urgency. |
| **Reading Grade Level**  | [X Grade Level] | Is the readability appropriate for the target audience? | Simplify or enhance complexity based on audience needs. |

---

### **Part 1B: Image Headline Optimization Analysis**  
"Analyze the **image headline** for effectiveness, clarity, and marketing impact. Evaluate the following criteria and provide **explanation and recommendations**."

| **Criterion**            | **Assessment** | **Explanation** | **Recommendation** |
|--------------------------|---------------|-----------------|---------------------|
| **Word Count**           | X words       |                 |                     |
| **Keyword Relevance**    | High/Medium/Low |                 |                     |
| **Common Words**         | X words       |                 |                     |
| **Uncommon Words**       | X words       |                 |                     |
| **Power Words**          | X words       |                 |                     |
| **Emotional Words**      | X words       |                 |                     |
| **Sentiment**            | Positive/Negative/Neutral |                 |                     |
| **Reading Grade Level**  | X Grade Level |                 |                     |

---

### **Part 1C: Supporting Headline Optimization Analysis**  
"Analyze the **supporting headline** to determine its contribution to the overall marketing message."

| **Criterion**            | **Assessment** | **Explanation** | **Recommendation** |
|--------------------------|---------------|-----------------|---------------------|
| **Word Count**           | X words       |                 |                     |
| **Keyword Relevance**    | High/Medium/Low |                 |                     |
| **Common Words**         | X words       |                 |                     |
| **Uncommon Words**       | X words       |                 |                     |
| **Power Words**          | X words       |                 |                     |
| **Emotional Words**      | X words       |                 |                     |
| **Sentiment**            | Positive/Negative/Neutral |                 |                     |
| **Reading Grade Level**  | X Grade Level |                 |                     |
---
### **Final Summary & Recommendations**  

1. **Overall Headline Effectiveness Score** (average of all headline scores).  
2. **Key Strengths:** Summarize the **headline elements that perform well**.  
3. **Areas for Improvement:** Identify **critical weaknesses** and **how they can be optimized**.  
4. **Action Plan:** Provide a **step-by-step strategy** for enhancing clarity, engagement, and marketing effectiveness.    
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
### **Objective:**  
Evaluate the **effectiveness, clarity, and impact** of the **main headline** in a **marketing asset (image or video)** based on key **linguistic and marketing principles**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the different headline elements**:  

1. **Main Headline:** The primary headline conveying the core message.  
2. **Image Headline (if applicable):** A separate **image-specific** headline.  
3. **Supporting Headline (if applicable):** Any **additional text elements** that contribute to the message.  

Ensure **text extraction accuracy**, maintain the **original structure**, and **translate non-English text into English** before analysis.

---

### **Step 2: Headline Effectiveness Analysis**  
Analyze the extracted **main headline** based on the following key **marketing and linguistic criteria**.  

- **Each criterion should be assessed on a scale of 1 to 5** (in increments of 0.5).  
- Provide a **clear explanation** for the assessment.  
- Offer **specific, actionable recommendations** for improvement.  

---

#### **Structured Table Format for Headline Analysis**  

**Headline being analyzed:** _[Main Headline]_  

| **Criterion**            | **Score (1-5)** | **Explanation (Strengths & Weaknesses)** | **Suggested Improvement** |
|--------------------------|----------------|------------------------------------------|---------------------------|
| **Clarity**             | _[1-5]_         | _Is the headline clear and easy to understand?_ | _Simplify or rephrase unclear words._ |
| **Customer Focus**      | _[1-5]_         | _Does the headline address customer needs?_ | _Refine wording to highlight customer benefits._ |
| **Relevance**          | _[1-5]_         | _Does the headline align with the product and audience?_ | _Ensure alignment with marketing message._ |
| **Emotional Appeal**   | _[1-5]_         | _Does the headline evoke emotions or curiosity?_ | _Incorporate emotionally compelling words._ |
| **Uniqueness**         | _[1-5]_         | _Is the headline distinctive and not generic?_ | _Improve originality to stand out._ |
| **Urgency & Curiosity** | _[1-5]_         | _Does it create urgency or entice curiosity?_ | _Use stronger action words or time-limited phrases._ |
| **Benefit-Driven**     | _[1-5]_         | _Does it communicate a clear value proposition?_ | _Highlight direct benefits for the reader._ |
| **Target Audience**    | _[1-5]_         | _Is it relevant to the target demographic?_ | _Adjust language to better suit the audience._ |
| **Length & Format**    | _[1-5]_         | _Is the headline concise (6-12 words)?_ | _Refine for optimal readability._ |
| **Overall Effectiveness** | _[1-5]_     | _How well does it perform as a marketing tool?_ | _Summarize key areas for improvement._ |

**Total Score:** _[Sum of all scores]_
---

### **Step 3: Improved Headline Suggestions**  
Generate three alternative **optimized headlines** for the main headline.  

- Ensure **clarity, emotional appeal, and engagement**.  
- Align **better with marketing goals and audience needs**.  

**Structured Table Format for Headline Suggestions:**  

| **Option** | **Suggested Headline** | **Rationale for Improvement** |
|-----------|----------------------|-----------------------------|
| **Option 1** | _[New Headline]_ | _Explain why this version is more engaging or effective._ |
| **Option 2** | _[New Headline]_ | _Explain the benefits of this variation._ |
| **Option 3** | _[New Headline]_ | _Explain how this alternative improves on the original._ |
---
### **Final Summary & Recommendations**  

1. **Overall Headline Effectiveness Score** (Average of all ratings).  
2. **Key Strengths:** Highlight what works well.  
3. **Critical Areas for Improvement:** Summarize the **main weaknesses** and **how to fix them**.  
4. **Next Steps:** Offer a **step-by-step strategy** to **refine and test the improved headlines**.
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
### **Objective:**  
Evaluate the **effectiveness, clarity, and marketing impact** of the **image headline** in a **marketing asset (image or video)** using key **linguistic and marketing principles**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the different headline elements**:  

1. **Main Headline:** The **primary** headline conveying the **core message**.  
2. **Image Headline (if applicable):** A distinct **headline embedded within the image**.  
3. **Supporting Headline (if applicable):** Additional **text elements that reinforce the message**.  

Ensure **text extraction accuracy**, **preserve the original structure**, and **translate non-English text into English** before proceeding with the analysis.

---

### **Step 2: Image Headline Effectiveness Analysis**  
Analyze the extracted **image headline** based on the following **marketing and linguistic criteria**.  

- **Each criterion should be rated on a scale of 1 to 5** (in increments of 0.5).  
- Provide a **clear, concise explanation** for the rating.  
- Offer **specific, actionable recommendations** for improvement.  

---

#### **Structured Table Format for Image Headline Analysis**  

**Headline being analyzed:** _[Image Headline]_  

| **Criterion**            | **Score (1-5)** | **Explanation (Strengths & Weaknesses)** | **Suggested Improvement** |
|--------------------------|----------------|------------------------------------------|---------------------------|
| **Clarity**             | _[1-5]_         | _Is the headline easy to understand at a glance?_ | _Simplify complex wording or improve readability._ |
| **Customer Focus**      | _[1-5]_         | _Does the headline address the customer's needs or desires?_ | _Reframe to make it more customer-centric._ |
| **Relevance**          | _[1-5]_         | _Does the headline align with the product and audience expectations?_ | _Ensure alignment with the image's marketing message._ |
| **Emotional Appeal**   | _[1-5]_         | _Does the headline evoke emotions (curiosity, excitement, urgency)?_ | _Strengthen emotional triggers where necessary._ |
| **Uniqueness**         | _[1-5]_         | _Is the headline distinct and engaging?_ | _Make it stand out from competitors._ |
| **Urgency & Curiosity** | _[1-5]_         | _Does the headline create urgency or intrigue?_ | _Use time-sensitive language or curiosity-driven phrasing._ |
| **Benefit-Driven**     | _[1-5]_         | _Does the headline highlight a clear benefit?_ | _Emphasize tangible customer benefits._ |
| **Target Audience**    | _[1-5]_         | _Is the language and tone tailored to the intended audience?_ | _Adjust the tone and vocabulary to better fit the demographic._ |
| **Length & Format**    | _[1-5]_         | _Is the headline concise and visually balanced within the image?_ | _Refine for optimal readability and impact._ |
| **Overall Effectiveness** | _[1-5]_       | _How well does it function as an attention-grabbing marketing tool?_ | _Summarize key areas for improvement._ |

**Total Score:** _[Sum of all scores]_  

---

### **Step 3: Improved Image Headline Suggestions**  
Generate **three alternative optimized headlines** for the **image headline**, ensuring:  

- **Improved clarity, engagement, and emotional impact**.  
- **Better alignment with the marketing message and audience expectations**.  

**Structured Table Format for Headline Suggestions:**  

| **Option** | **Suggested Headline** | **Rationale for Improvement** |
|-----------|----------------------|-----------------------------|
| **Option 1** | _[New Headline]_ | _Explain how this version improves clarity, engagement, or relevance._ |
| **Option 2** | _[New Headline]_ | _Describe how this variation strengthens customer focus or emotional appeal._ |
| **Option 3** | _[New Headline]_ | _Explain why this alternative is more compelling than the original._ |

---

### **Final Summary & Recommendations**  

1. **Overall Headline Effectiveness Score** (Average of all ratings).  
2. **Key Strengths:** Highlight what works well.  
3. **Critical Areas for Improvement:** Summarize **weaknesses and how to fix them**.  
4. **Next Steps:** Offer a **step-by-step strategy** to **refine and test the improved headlines**.
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
### **Objective:**  
Evaluate the **effectiveness, clarity, and marketing impact** of the **supporting headline** in a **marketing asset (image or video)** using key **linguistic and marketing principles**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the different headline elements**:  

1. **Main Headline:** The **primary** headline conveying the **core message**.  
2. **Image Headline (if applicable):** A distinct **headline embedded within the image**.  
3. **Supporting Headline (if applicable):** Additional **text elements that reinforce the message**.  

Ensure **text extraction accuracy**, **preserve the original structure**, and **translate non-English text into English** before proceeding with the analysis.

---

### **Step 2: Supporting Headline Effectiveness Analysis**  
Analyze the extracted **supporting headline** based on the following **marketing and linguistic criteria**.  

- **Each criterion should be rated on a scale of 1 to 5** (in increments of 0.5).  
- Provide a **clear, concise explanation** for the rating.  
- Offer **specific, actionable recommendations** for improvement.  

---

#### **Structured Table Format for Supporting Headline Analysis**  

**Headline being analyzed:** _[Supporting Headline]_  

| **Criterion**            | **Score (1-5)** | **Explanation (Strengths & Weaknesses)** | **Suggested Improvement** |
|--------------------------|----------------|------------------------------------------|---------------------------|
| **Clarity**             | _[1-5]_         | _Is the headline easy to understand at a glance?_ | _Simplify complex wording or improve readability._ |
| **Customer Focus**      | _[1-5]_         | _Does the headline address the customer's needs or desires?_ | _Reframe to make it more customer-centric._ |
| **Relevance**          | _[1-5]_         | _Does the headline align with the product and audience expectations?_ | _Ensure alignment with the image's marketing message._ |
| **Emotional Appeal**   | _[1-5]_         | _Does the headline evoke emotions (curiosity, excitement, urgency)?_ | _Strengthen emotional triggers where necessary._ |
| **Uniqueness**         | _[1-5]_         | _Is the headline distinct and engaging?_ | _Make it stand out from competitors._ |
| **Urgency & Curiosity** | _[1-5]_         | _Does the headline create urgency or intrigue?_ | _Use time-sensitive language or curiosity-driven phrasing._ |
| **Benefit-Driven**     | _[1-5]_         | _Does the headline highlight a clear benefit?_ | _Emphasize tangible customer benefits._ |
| **Target Audience**    | _[1-5]_         | _Is the language and tone tailored to the intended audience?_ | _Adjust the tone and vocabulary to better fit the demographic._ |
| **Length & Format**    | _[1-5]_         | _Is the headline concise and visually balanced within the image?_ | _Refine for optimal readability and impact._ |
| **Overall Effectiveness** | _[1-5]_       | _How well does it function as an attention-grabbing marketing tool?_ | _Summarize key areas for improvement._ |

**Total Score:** _[Sum of all scores]_  

---

### **Step 3: Improved Supporting Headline Suggestions**  
Generate **three alternative optimized headlines** for the **supporting headline**, ensuring:  

- **Improved clarity, engagement, and emotional impact**.  
- **Better alignment with the marketing message and audience expectations**.  

**Structured Table Format for Headline Suggestions:**  

| **Option** | **Suggested Headline** | **Rationale for Improvement** |
|-----------|----------------------|-----------------------------|
| **Option 1** | _[New Headline]_ | _Explain how this version improves clarity, engagement, or relevance._ |
| **Option 2** | _[New Headline]_ | _Describe how this variation strengthens customer focus or emotional appeal._ |
| **Option 3** | _[New Headline]_ | _Explain why this alternative is more compelling than the original._ |

---

### **Final Summary & Recommendations**  

1. **Overall Supporting Headline Effectiveness Score** (Average of all ratings).  
2. **Key Strengths:** Highlight what works well.  
3. **Critical Areas for Improvement:** Summarize **weaknesses and how to fix them**.  
4. **Next Steps:** Offer a **step-by-step strategy** to **refine and test the improved headlines**.
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
### **Objective:**  
Evaluate the **effectiveness, clarity, and marketing impact** of the **main headline** in a **marketing asset (image or video)** using key **linguistic and marketing principles**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the main headline**:  

1. **Main Headline Identification:**  
   - Extract the **exact text** of the main headline.  
   - If the content is **non-English**, translate it into **English** before analysis.  

---

### **Step 2: Main Headline Effectiveness Analysis**  
Analyze the extracted **main headline** based on the following **marketing and linguistic criteria**:  

- **Each criterion should be rated on a scale of 1 to 5** (in increments of 0.5).  
- Provide a **clear, concise explanation** for the rating.  
- Offer **specific, actionable recommendations** for improvement.  

---

#### **Structured Table Format for Main Headline Analysis**  

**Headline being analyzed:** _[Main Headline]_  

| **Criterion**            | **Assessment**               | **Explanation (Strengths & Weaknesses)** | **Suggested Improvement** |
|--------------------------|-----------------------------|------------------------------------------|---------------------------|
| **Word Count**           | _[X words]_                 | _The headline has [X] words, making it [appropriate/too short/too long]._ | _Consider adjusting to [Y] words for optimal impact._ |
| **Keyword Relevance**    | _[High/Moderate/Low]_       | _The headline [includes/misses] relevant keywords such as [X]._ | _Incorporate [more/specific] keywords like [Y]._ |
| **Common Words**        | _[Number]_                   | _Common words [enhance/reduce] readability and appeal._ | _[Increase/reduce] the use of common words._ |
| **Uncommon Words**      | _[Number]_                   | _Uncommon words make the headline [stand out/confusing]._ | _Balance [common/uncommon] words for clarity._ |
| **Power Words**         | _[Number]_                   | _Power words [create urgency/may overwhelm] the reader._ | _Use power words [more sparingly/more effectively]._ |
| **Emotional Words**     | _[Number]_                   | _Emotional tone is [effective/overdone/subtle]._ | _Adjust the emotional tone by [modifying X]._ |
| **Sentiment**           | _[Positive/Negative/Neutral]_ | _The sentiment is [not aligning well/matching] with the image._ | _Match the sentiment more closely with the image._ |
| **Reading Grade Level** | _[Grade level]_              | _The headline is [too complex/simple] for the target audience._ | _Adapt the reading level to [simplify/complexify]._ |

---

### **Step 3: Improved Main Headline Suggestions**  
Generate **three alternative optimized headlines**, ensuring:  

- **Improved clarity, engagement, and emotional impact**.  
- **Better alignment with the marketing message and audience expectations**.  

**Structured Table Format for Headline Suggestions:**  

| **Option** | **Suggested Headline** | **Rationale for Improvement** |
|-----------|----------------------|-----------------------------|
| **Option 1** | _[New Headline]_ | _Explain how this version improves clarity, engagement, or relevance._ |
| **Option 2** | _[New Headline]_ | _Describe how this variation strengthens customer focus or emotional appeal._ |
| **Option 3** | _[New Headline]_ | _Explain why this alternative is more compelling than the original._ |

---

### **Final Summary & Recommendations**  

1. **Overall Headline Effectiveness Score** (Average of all ratings).  
2. **Key Strengths:** Highlight what works well.  
3. **Critical Areas for Improvement:** Summarize **weaknesses and how to fix them**.  
4. **Next Steps:** Offer a **step-by-step strategy** to **refine and test the improved headlines**.
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
### **Objective:**  
Evaluate the **effectiveness, clarity, and marketing impact** of the **image headline** in a **marketing asset (image or video)** using key **linguistic and marketing principles**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Image Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the image headline**:  

1. **Image Headline Identification:**  
   - Extract the **exact text** of the image headline.  
   - If the content is **non-English**, translate it into **English** before analysis.  

---

### **Step 2: Image Headline Effectiveness Analysis**  
Analyze the extracted **image headline** based on the following **marketing and linguistic criteria**:  

- **Each criterion should be rated on a scale of 1 to 5** (in increments of 0.5).  
- Provide a **clear, concise explanation** for the rating.  
- Offer **specific, actionable recommendations** for improvement.  

---

#### **Structured Table Format for Image Headline Analysis**  

**Headline being analyzed:** _[Image Headline]_  

| **Criterion**            | **Assessment**               | **Explanation (Strengths & Weaknesses)** | **Suggested Improvement** |
|--------------------------|-----------------------------|------------------------------------------|---------------------------|
| **Word Count**           | _[X words]_                 | _The headline has [X] words, making it [appropriate/too short/too long]._ | _Consider adjusting to [Y] words for optimal impact._ |
| **Keyword Relevance**    | _[High/Moderate/Low]_       | _The headline [includes/misses] relevant keywords such as [X]._ | _Incorporate [more/specific] keywords like [Y]._ |
| **Common Words**        | _[Number]_                   | _Common words [enhance/reduce] readability and appeal._ | _[Increase/reduce] the use of common words._ |
| **Uncommon Words**      | _[Number]_                   | _Uncommon words make the headline [stand out/confusing]._ | _Balance [common/uncommon] words for clarity._ |
| **Power Words**         | _[Number]_                   | _Power words [create urgency/may overwhelm] the reader._ | _Use power words [more sparingly/more effectively]._ |
| **Emotional Words**     | _[Number]_                   | _Emotional tone is [effective/overdone/subtle]._ | _Adjust the emotional tone by [modifying X]._ |
| **Sentiment**           | _[Positive/Negative/Neutral]_ | _The sentiment is [not aligning well/matching] with the image._ | _Match the sentiment more closely with the image._ |
| **Reading Grade Level** | _[Grade level]_              | _The headline is [too complex/simple] for the target audience._ | _Adapt the reading level to [simplify/complexify]._ |

---

### **Step 3: Improved Image Headline Suggestions**  
Generate **three alternative optimized headlines**, ensuring:  

- **Improved clarity, engagement, and emotional impact**.  
- **Better alignment with the marketing message and audience expectations**.  

**Structured Table Format for Headline Suggestions:**  

| **Option** | **Suggested Headline** | **Rationale for Improvement** |
|-----------|----------------------|-----------------------------|
| **Option 1** | _[New Headline]_ | _Explain how this version improves clarity, engagement, or relevance._ |
| **Option 2** | _[New Headline]_ | _Describe how this variation strengthens customer focus or emotional appeal._ |
| **Option 3** | _[New Headline]_ | _Explain why this alternative is more compelling than the original._ |

---

### **Final Summary & Recommendations**  

1. **Overall Image Headline Effectiveness Score** (Average of all ratings).  
2. **Key Strengths:** Highlight what works well.  
3. **Critical Areas for Improvement:** Summarize **weaknesses and how to fix them**.  
4. **Next Steps:** Offer a **step-by-step strategy** to **refine and test the improved headlines**. 
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
### **Objective:**  
Evaluate the **effectiveness, clarity, and marketing impact** of the **supporting headline** in a **marketing asset (image or video)** using key **linguistic and marketing principles**. Provide structured insights and **actionable recommendations** for improvement.

---

### **Instructions:**  

#### **Step 1: Supporting Headline Extraction & Context**  
For the given **image or video**, extract and **clearly identify the supporting headline**:  

1. **Supporting Headline Identification:**  
   - Extract the **exact text** of the supporting headline.  
   - If the content is **non-English**, translate it into **English** before analysis.  

---

### **Step 2: Supporting Headline Effectiveness Analysis**  
Analyze the extracted **supporting headline** based on the following **marketing and linguistic criteria**:  

- **Each criterion should be rated on a scale of 1 to 5** (in increments of 0.5).  
- Provide a **clear, concise explanation** for the rating.  
- Offer **specific, actionable recommendations** for improvement.  

---

#### **Structured Table Format for Supporting Headline Analysis**  

**Headline being analyzed:** _[Supporting Headline]_  

| **Criterion**            | **Assessment**               | **Explanation (Strengths & Weaknesses)** | **Suggested Improvement** |
|--------------------------|-----------------------------|------------------------------------------|---------------------------|
| **Word Count**           | _[X words]_                 | _The headline has [X] words, making it [appropriate/too short/too long]._ | _Consider adjusting to [Y] words for optimal impact._ |
| **Keyword Relevance**    | _[High/Moderate/Low]_       | _The headline [includes/misses] relevant keywords such as [X]._ | _Incorporate [more/specific] keywords like [Y]._ |
| **Common Words**        | _[Number]_                   | _Common words [enhance/reduce] readability and appeal._ | _[Increase/reduce] the use of common words._ |
| **Uncommon Words**      | _[Number]_                   | _Uncommon words make the headline [stand out/confusing]._ | _Balance [common/uncommon] words for clarity._ |
| **Power Words**         | _[Number]_                   | _Power words [create urgency/may overwhelm] the reader._ | _Use power words [more sparingly/more effectively]._ |
| **Emotional Words**     | _[Number]_                   | _Emotional tone is [effective/overdone/subtle]._ | _Adjust the emotional tone by [modifying X]._ |
| **Sentiment**           | _[Positive/Negative/Neutral]_ | _The sentiment is [not aligning well/matching] with the image._ | _Match the sentiment more closely with the image._ |
| **Reading Grade Level** | _[Grade level]_              | _The headline is [too complex/simple] for the target audience._ | _Adapt the reading level to [simplify/complexify]._ |

---

### **Step 3: Improved Supporting Headline Suggestions**  
Generate **three alternative optimized headlines**, ensuring:  

- **Improved clarity, engagement, and emotional impact**.  
- **Better alignment with the marketing message and audience expectations**.  

**Structured Table Format for Headline Suggestions:**  

| **Option** | **Suggested Headline** | **Rationale for Improvement** |
|-----------|----------------------|-----------------------------|
| **Option 1** | _[New Headline]_ | _Explain how this version improves clarity, engagement, or relevance._ |
| **Option 2** | _[New Headline]_ | _Describe how this variation strengthens customer focus or emotional appeal._ |
| **Option 3** | _[New Headline]_ | _Explain why this alternative is more compelling than the original._ |

---

### **Final Summary & Recommendations**  

1. **Overall Supporting Headline Effectiveness Score** (Average of all ratings).  
2. **Key Strengths:** Highlight what works well.  
3. **Critical Areas for Improvement:** Summarize **weaknesses and how to fix them**.  
4. **Next Steps:** Offer a **step-by-step strategy** to **refine and test the improved headlines**.
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
### **Objective:**  
Identify **4 key persona types** based on **Facebook's targeting parameters** and analyze how they would respond to an advertisement. Provide structured insights, including demographic, behavioral, and interest-based details, along with a **detailed persona analysis.**  

---

### **Instructions:**  

#### **Step 1: Persona Type Identification**  
- Based on **Facebook targeting options**, define **4 persona types** that are most likely to respond to the ad.  
- Present the personas in a structured table format:  

---

#### **Table 1: Persona Type Identification**  
| **Persona Type** | **Description** |  
|-----------------|---------------|  
| **[Persona Type 1]** | _Detailed description of this audience segment, including key interests, behaviors, and demographics._ |  
| **[Persona Type 2]** | _Detailed description of this audience segment, including key interests, behaviors, and demographics._ |  
| **[Persona Type 3]** | _Detailed description of this audience segment, including key interests, behaviors, and demographics._ |  
| **[Persona Type 4]** | _Detailed description of this audience segment, including key interests, behaviors, and demographics._ |  

---

#### **Step 2: Persona Development**  
For each of the **4 personas**, develop a **detailed customer profile**, including **name, characteristics, and expected ad response**.  
- Present the information in a structured table format.  

---

#### **Table 2: Detailed Persona Analysis**  

| **Persona Name** | **Persona Type** | **Description** | **Analysis (How they would react to the ad)** |  
|-----------------|----------------|----------------|--------------------------------|  
| **[Persona 1 Name]** | _[Persona Type]_ | _[Detailed demographic and interest-based details]_ | _[How this persona would engage with the ad, their likely behavior, and purchase intent]_ |  
| **[Persona 2 Name]** | _[Persona Type]_ | _[Detailed demographic and interest-based details]_ | _[How this persona would engage with the ad, their likely behavior, and purchase intent]_ |  
| **[Persona 3 Name]** | _[Persona Type]_ | _[Detailed demographic and interest-based details]_ | _[How this persona would engage with the ad, their likely behavior, and purchase intent]_ |  
| **[Persona 4 Name]** | _[Persona Type]_ | _[Detailed demographic and interest-based details]_ | _[How this persona would engage with the ad, their likely behavior, and purchase intent]_ |  

---

### **Step 3: Facebook Targeting Parameters for Each Persona**  
For each persona, specify **which Facebook targeting options** should be selected.  

| **Persona Name** | **Targeting Criteria** | **Selected Options** |  
|-----------------|----------------|----------------|  
| **[Persona 1 Name]** | Location | _[Specify targeted countries, cities, or zip codes]_ |  
|  | Age Range | _[Specify selected age bracket]_ |  
|  | Gender | _[Targeted gender: Men/Women/All]_ |  
|  | Language | _[Targeted language preferences]_ |  
|  | Interests | _[Pages liked, entertainment, fitness, hobbies, etc.]_ |  
|  | Behaviors | _[Device usage, travel, purchase behavior, etc.]_ |  
|  | Purchase Behavior | _[Categories of past purchases]_ |  
|  | Job Title | _[Specific job titles or industries]_ |  
| **[Persona 2 Name]** | _(Repeat for each persona)_ | _(Customized targeting options)_ |  

---

### **Final Recommendations & Summary**  

1. **Which persona is the most valuable target for the ad?**  
2. **Which targeting parameters are the most effective for achieving conversions?**  
3. **How should the ad be adjusted to maximize engagement based on the persona insights?**
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
### **Objective:**  
Identify **4 key persona types** based on **LinkedIn’s targeting parameters** and analyze how they would respond to an advertisement. Provide structured insights, including demographic, behavioral, and professional attributes, along with a **detailed persona analysis** for marketing effectiveness.

---

### **Instructions:**  

#### **Step 1: Persona Type Identification**  
- Based on **LinkedIn’s targeting options**, define **4 persona types** that are most likely to respond to the ad.  
- Present the personas in a structured table format:  

---

#### **Table 1: Persona Type Identification**  
| **Persona Type** | **Description** |  
|-----------------|---------------|  
| **[Persona Type 1]** | _Detailed description of this professional segment, including key industries, roles, and behaviors._ |  
| **[Persona Type 2]** | _Detailed description of this professional segment, including key industries, roles, and behaviors._ |  
| **[Persona Type 3]** | _Detailed description of this professional segment, including key industries, roles, and behaviors._ |  
| **[Persona Type 4]** | _Detailed description of this professional segment, including key industries, roles, and behaviors._ |  

---

#### **Step 2: Persona Development**  
For each of the **4 personas**, develop a **detailed customer profile**, including **name, characteristics, and expected ad response**.  
- Present the information in a structured table format.  

---

#### **Table 2: Detailed Persona Analysis**  

| **Persona Name** | **Persona Type** | **Description** | **Analysis (How they would react to the ad)** |  
|-----------------|----------------|----------------|--------------------------------|  
| **[Persona 1 Name]** | _[Persona Type]_ | _[Detailed professional background, skills, and behaviors]_ | _[How this persona would engage with the ad, their likely response, and purchase intent]_ |  
| **[Persona 2 Name]** | _[Persona Type]_ | _[Detailed professional background, skills, and behaviors]_ | _[How this persona would engage with the ad, their likely response, and purchase intent]_ |  
| **[Persona 3 Name]** | _[Persona Type]_ | _[Detailed professional background, skills, and behaviors]_ | _[How this persona would engage with the ad, their likely response, and purchase intent]_ |  
| **[Persona 4 Name]** | _[Persona Type]_ | _[Detailed professional background, skills, and behaviors]_ | _[How this persona would engage with the ad, their likely response, and purchase intent]_ |  

---

### **Step 3: LinkedIn Targeting Parameters for Each Persona**  
For each persona, specify **which LinkedIn targeting options** should be selected.  

| **Persona Name** | **Targeting Criteria** | **Selected Options** |  
|-----------------|----------------|----------------|  
| **[Persona 1 Name]** | Location | _[Specify targeted country, city, or region]_ |  
|  | Company Industry | _[Specify industry type]_ |  
|  | Company Size | _[Target specific company sizes]_ |  
|  | Job Function | _[Specify job function, e.g., Marketing, IT, Sales]_ |  
|  | Job Seniority | _[Specify experience level, e.g., entry-level, director, executive]_ |  
|  | Job Title | _[Target specific job titles]_ |  
|  | Years of Experience | _[Target professionals based on experience]_ |  
|  | Schools | _[Target alumni of specific institutions]_ |  
|  | Degrees | _[Specify relevant degree qualifications]_ |  
|  | Fields of Study | _[Target professionals based on academic background]_ |  
|  | Skills | _[Specify LinkedIn-listed skills]_ |  
|  | Member Groups | _[Target professionals based on group memberships]_ |  
|  | Interests | _[Specify LinkedIn content interaction and topic interests]_ |  
| **[Persona 2 Name]** | _(Repeat for each persona)_ | _(Customized targeting options)_ |  

---

### **Final Recommendations & Summary**  

1. **Which persona is the most valuable target for the ad?**  
2. **Which targeting parameters are the most effective for achieving engagement and conversions?**  
3. **How should the ad messaging and visual content be optimized to align with the persona insights?**
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
### **Objective:**  
Identify **4 key persona types** based on **X’s targeting parameters** and analyze how they would respond to an advertisement. Provide structured insights, including demographic, behavioral, and engagement attributes, along with a **detailed persona analysis** for marketing effectiveness.

---

### **Instructions:**  

#### **Step 1: Persona Type Identification**  
- Based on **X’s targeting options**, define **4 persona types** that are most likely to engage with the ad.  
- Present the personas in a structured table format:  

---

#### **Table 1: Persona Type Identification**  
| **Persona Type** | **Description** |  
|-----------------|---------------|  
| **[Persona Type 1]** | _Detailed description of this audience segment, including interests, behaviors, and engagement patterns._ |  
| **[Persona Type 2]** | _Detailed description of this audience segment, including interests, behaviors, and engagement patterns._ |  
| **[Persona Type 3]** | _Detailed description of this audience segment, including interests, behaviors, and engagement patterns._ |  
| **[Persona Type 4]** | _Detailed description of this audience segment, including interests, behaviors, and engagement patterns._ |  

---

#### **Step 2: Persona Development**  
For each of the **4 personas**, develop a **detailed customer profile**, including **name, characteristics, and expected ad response**.  
- Present the information in a structured table format.  

---

#### **Table 2: Detailed Persona Analysis**  

| **Persona Name** | **Persona Type** | **Description** | **Analysis (How they would react to the ad)** |  
|-----------------|----------------|----------------|--------------------------------|  
| **[Persona 1 Name]** | _[Persona Type]_ | _[Detailed demographic, behavioral, and interest-based attributes]_ | _[How this persona would engage with the ad, their likely response, and intent]_ |  
| **[Persona 2 Name]** | _[Persona Type]_ | _[Detailed demographic, behavioral, and interest-based attributes]_ | _[How this persona would engage with the ad, their likely response, and intent]_ |  
| **[Persona 3 Name]** | _[Persona Type]_ | _[Detailed demographic, behavioral, and interest-based attributes]_ | _[How this persona would engage with the ad, their likely response, and intent]_ |  
| **[Persona 4 Name]** | _[Persona Type]_ | _[Detailed demographic, behavioral, and interest-based attributes]_ | _[How this persona would engage with the ad, their likely response, and intent]_ |  

---

### **Step 3: X Targeting Parameters for Each Persona**  
For each persona, specify **which X targeting options** should be selected.  

| **Persona Name** | **Targeting Criteria** | **Selected Options** |  
|-----------------|----------------|----------------|  
| **[Persona 1 Name]** | Location | _[Specify targeted country, region, or metro area]_ |  
|  | Gender | _[Specify male, female, or all genders]_ |  
|  | Language | _[Target users based on language preferences]_ |  
|  | Interests | _[Specify interest categories inferred from engagement patterns]_ |  
|  | Events | _[Target users engaging with relevant global/local events]_ |  
|  | Behaviors | _[Target based on user behaviors such as tweets, engagements, retweets, etc.]_ |  
|  | Keywords | _[Target keywords from tweets and engagement trends]_ |  
|  | Topics | _[Engage users in predefined or custom topic conversations]_ |  
|  | Device | _[Specify device targeting (mobile, desktop, OS, etc.)]_ |  
|  | Carrier | _[Target based on mobile carrier (if relevant)]_ |  
|  | Geography | _[Ensure cultural context and regional targeting alignment]_ |  
| **[Persona 2 Name]** | _(Repeat for each persona)_ | _(Customized targeting options)_ |  

---

### **Final Recommendations & Summary**  

1. **Which persona is the most valuable target for the ad?**  
2. **Which targeting parameters are the most effective for engagement and conversions?**  
3. **How should the ad messaging and visual content be optimized to align with the persona insights?**
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
### **Objective:**  
Evaluate content using **various personality trait models**, analyzing how different personalities would respond to it. Provide structured insights with **scoring, analysis, and recommendations** in a table format. Include a **summary and overall recommendations** at the end.  

---

### **Instructions:**  

#### **Step 1: Content Translation (If Applicable)**  
- If the content is **not in English**, translate it to **English** before proceeding with the analysis.  

---

#### **Step 2: Personality Trait-Based Evaluation**  
- Assess the content based on **12 different personality models**, each containing unique traits.  
- Present results in a structured table format, including **score (1-5 in increments of 0.5), analysis, and improvement recommendations**.  

---

#### **Table Format for Personality Trait Evaluation**  

| **Personality Model** | **Trait** | **Score (1-5)** | **Analysis** | **Recommendation** |  
|----------------------|----------|---------------|------------|------------------|  
| **Big Five (OCEAN)** | Openness to Experience | _[Score]_ | _[Analysis of how well content aligns with this trait]_ | _[How to enhance content appeal to this trait]_ |  
|  | Conscientiousness | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Extraversion | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Agreeableness | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Neuroticism | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **Eysenck’s PEN Model** | Psychoticism | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Extraversion | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Neuroticism | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **HEXACO Model** | Honesty-Humility | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Emotionality | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **Cattell’s 16PF** | Select Key Traits | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **MBTI** | Extraversion vs. Introversion | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Sensing vs. Intuition | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **Dark Triad** | Machiavellianism | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **Enneagram** | Type 1: Reformer | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Type 2: Helper | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **DISC Model** | Dominance | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
|  | Influence | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **Keirsey Temperaments** | Artisan | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **Jungian Archetypes** | The Hero | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  

---

#### **Step 3: Summary & Overall Recommendations**  
- **Summarize** which personality types respond best and which are least engaged.  
- Provide **key improvements** to make the content more **universally effective** or tailored for **specific personality-driven audiences**.
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
### **Objective:**  
Evaluate the content based on **MBTI personality types**, analyzing how different types would perceive and respond to it. Provide structured insights with **scoring, analysis, and recommendations** in a **table format**. Include a **summary and overall recommendations** at the end.  

---

### **Instructions:**  

#### **Step 1: Content Translation (If Applicable)**  
- If the content is **not in English**, first **translate it into English** before proceeding with the analysis.  

---

#### **Step 2: MBTI Personality Type Evaluation**  
- Assess the content based on **16 MBTI personality types**.  
- Provide a **score (1-5 in increments of 0.5)** for each type, evaluating how well they would engage with the content.  
- Present results in a **structured table** including **analysis and recommendations**.  

---

#### **Table Format for MBTI Content Evaluation**  

| **MBTI Personality Type** | **Score (1-5)** | **Analysis** | **Recommendation** |  
|--------------------------|---------------|------------|------------------|  
| **ISTJ – The Inspector** | _[Score]_ | _[Analysis of how this type perceives the content]_ | _[How to refine content for this type]_ |  
| **ISFJ – The Protector** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **INFJ – The Advocate** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **INTJ – The Architect** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ISTP – The Virtuoso** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ISFP – The Adventurer** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **INFP – The Mediator** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **INTP – The Logician** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ESTP – The Entrepreneur** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ESFP – The Entertainer** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ENFP – The Campaigner** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ENTP – The Debater** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ESTJ – The Executive** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ESFJ – The Consul** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ENFJ – The Protagonist** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  
| **ENTJ – The Commander** | _[Score]_ | _[Analysis]_ | _[Recommendation]_ |  

---

#### **Step 3: Summary & Overall Recommendations**  
- **Summarize** which MBTI personality types are **most and least likely to engage** with the content.  
- **Provide key improvements** to enhance engagement across **multiple personality types** or refine content for **specific audiences**.
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
### **Objective:**  
Evaluate the provided **image** against multiple marketing and design principles. Each aspect should be scored **from 1 to 5 in increments of 0.5** (1 being low, 5 being high) and **include an explanation and suggested improvements**.  

- The results should be presented in a **structured table format** with the following columns:  
  **Aspect | Score | Explanation | Suggested Improvement**  
- After the table, provide an **overall summary** with **key recommendations for improvement**.  

---

### **Evaluation Criteria:**  

#### **1. Visual Appeal**  
- **Impact:** Attracts attention and conveys emotions.  
- **Analysis:** Assess **color scheme, composition, clarity, and aesthetic quality**.  
- **Application:** Ensure the image is **visually appealing, high-quality, and professionally designed**.  

#### **2. Relevance**  
- **Impact:** Resonates with the **target audience**.  
- **Analysis:** Determine if the image **aligns with audience preferences, brand messaging, and context**.  
- **Application:** Ensure the image **matches the audience’s interests, expectations, and brand identity**.  

#### **3. Emotional Impact**  
- **Impact:** Evokes **intended emotions** in viewers.  
- **Analysis:** Examine how the **visual elements and composition** contribute to emotional resonance.  
- **Application:** Incorporate **storytelling and relatable scenarios** to enhance emotional connection.  

#### **4. Message Clarity**  
- **Impact:** Effectively communicates the **main message**.  
- **Analysis:** Ensure the **main subject is clear** and the design is **not cluttered**.  
- **Application:** Maintain a **simple, well-structured design** that directs attention to the core message.  

#### **5. Engagement Potential**  
- **Impact:** Captures and retains audience **interest**.  
- **Analysis:** Evaluate elements that **draw attention and encourage interaction**.  
- **Application:** Utilize **compelling visuals, storytelling, and dynamic elements** to increase engagement.  

#### **6. Brand Recognition**  
- **Impact:** Enhances **brand recall** and association.  
- **Analysis:** Assess whether **branding elements (logos, colors, fonts) are well-integrated**.  
- **Application:** Maintain a **consistent brand style** for stronger **brand identity reinforcement**.  

#### **7. Cultural Sensitivity**  
- **Impact:** Ensures the image **respects cultural norms and diversity**.  
- **Analysis:** Check for **cultural appropriateness, inclusivity, and potential misinterpretations**.  
- **Application:** Ensure **visual representation aligns with a diverse audience and cultural nuances**.  

#### **8. Technical Quality**  
- **Impact:** Maintains **professional editing and high resolution**.  
- **Analysis:** Evaluate **sharpness, lighting, and post-processing quality**.  
- **Application:** Use **high-resolution images** with **proper lighting and editing** for a polished look.  

#### **9. Color Usage**  
- **Impact:** Influences **mood, perception, and attention**.  
- **Analysis:** Examine the **psychological effects of the chosen colors**.  
- **Application:** Use colors that **evoke intended emotions** and align with the **brand identity**.  

#### **10. Typography**  
- **Impact:** Affects **readability and engagement**.  
- **Analysis:** Assess **font choice, size, placement, and overall readability**.  
- **Application:** Ensure typography is **easy to read and visually harmonious** with the design.  

#### **11. Symbolism**  
- **Impact:** Conveys **complex ideas quickly**.  
- **Analysis:** Examine the **use of symbols, icons, or imagery** for enhanced message delivery.  
- **Application:** Utilize **recognizable and meaningful symbols** aligned with the brand and message.  

#### **12. Contrast**  
- **Impact:** Highlights **important elements** and enhances visibility.  
- **Analysis:** Evaluate **contrast between text, visuals, and background**.  
- **Application:** Ensure **strong contrast** to improve readability and **focus on key elements**.  

#### **13. Layout Balance**  
- **Impact:** Creates a **visually balanced composition**.  
- **Analysis:** Assess the **distribution of elements** to maintain harmony.  
- **Application:** Avoid **clutter and imbalance** by distributing elements **evenly and strategically**.  

#### **14. Hierarchy**  
- **Impact:** Guides the viewer’s **attention effectively**.  
- **Analysis:** Evaluate whether the **most important elements stand out first**.  
- **Application:** Use **size, color, and positioning** to create a clear **visual hierarchy**.  

---

### **Structured Table for Analysis**  

| **Aspect**           | **Score (1-5)** | **Explanation** | **Suggested Improvement** |  
|----------------------|---------------|----------------|--------------------------|  
| **Visual Appeal**    | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Relevance**       | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Emotional Impact** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Message Clarity**  | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Engagement**       | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Brand Recognition**| _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Cultural Sensitivity** | _[Score]_  | _[Analysis]_   | _[Recommendation]_       |  
| **Technical Quality**| _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Color Usage**      | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Typography**       | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Symbolism**        | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Contrast**        | _[Score]_       | _[Analysis]_   | _[Recommendation]_       |  
| **Layout Balance**  | _[Score]_       | _[Analysis]_   | _[Recommendation]_       |  
| **Hierarchy**       | _[Score]_       | _[Analysis]_   | _[Recommendation]_       |  

---

### **Final Summary & Recommendations**  
- Provide an **overall summary** of the image’s **strengths and weaknesses**.  
- Identify **key areas that need improvement** and suggest **general enhancements**.
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
### **Objective:**  
Evaluate the **image** against multiple marketing and design principles. Each aspect should be scored **from 1 to 5 in increments of 0.5** (1 being low, 5 being high), with an **explanation and suggested improvements**.  

- Present the results in a **structured table format** with columns:  
  **Aspect | Score | Explanation | Suggested Improvement**  
- After the table, provide an **overall summary** with **key recommendations for improvement**.  

---

### **Evaluation Criteria:**  

#### **1. Emotional Appeal**  
- **Impact:** Does the image evoke a strong emotional response?  
- **Analysis:** Identify the **emotions triggered** (e.g., happiness, nostalgia, excitement, urgency).  
- **Application:** Ensure the emotional tone **aligns with the campaign’s goal and the target audience’s expectations**.  

#### **2. Eye Attraction**  
- **Impact:** Does the image grab attention immediately?  
- **Analysis:** Identify which **visual elements** (color, subject, composition) are most effective.  
- **Application:** Optimize the focal point to **draw and maintain viewer attention**.  

#### **3. Visual Appeal**  
- **Impact:** How aesthetically pleasing is the image overall?  
- **Analysis:** Assess balance, **symmetry, composition, and creative visual techniques**.  
- **Application:** Ensure **harmonious elements** that enhance brand perception.  

#### **4. Text Overlay (Clarity, Emotional Connection, Readability)**  
- **Impact:** Is the text readable and well-integrated?  
- **Analysis:** Assess **contrast, font style, placement, and messaging impact**.  
- **Application:** Improve readability and emotional resonance **without overwhelming the design**.  

#### **5. Contrast and Clarity**  
- **Impact:** Highlights important elements and improves visibility.  
- **Analysis:** Check if **foreground and background elements** are distinct.  
- **Application:** Adjust contrast to **improve message visibility and clarity**.  

#### **6. Visual Hierarchy**  
- **Impact:** Guides the viewer’s **eye movement** naturally.  
- **Analysis:** Ensure the **key message, product, or CTA** is prioritized.  
- **Application:** Use **size, color, and placement** to direct attention effectively.  

#### **7. Negative Space**  
- **Impact:** Enhances balance and focus.  
- **Analysis:** Ensure negative space **prevents clutter and improves readability**.  
- **Application:** Optimize spacing to **improve clarity and organization**.  

#### **8. Color Psychology**  
- **Impact:** Influences **mood, perception, and attention**.  
- **Analysis:** Examine how the colors **evoke emotions and reinforce branding**.  
- **Application:** Use colors strategically to **strengthen brand identity and emotional response**.  

#### **9. Depth and Texture**  
- **Impact:** Enhances realism and engagement.  
- **Analysis:** Evaluate **shadows, gradients, and layering techniques**.  
- **Application:** Ensure texture and depth **add value without complicating the message**.  

#### **10. Brand Consistency**  
- **Impact:** Reinforces **brand recognition and trust**.  
- **Analysis:** Check if branding elements (**logo, fonts, colors**) are well-integrated.  
- **Application:** Maintain **consistent brand aesthetics** across all marketing assets.  

#### **11. Psychological Triggers**  
- **Impact:** Encourages desired actions.  
- **Analysis:** Identify triggers like **scarcity, authority, or social proof**.  
- **Application:** Use subtle psychological cues **without appearing manipulative**.  

#### **12. Emotional Connection**  
- **Impact:** Strengthens audience engagement.  
- **Analysis:** Does the image **align with audience values and desires**?  
- **Application:** Use **authentic messaging** that resonates with the target demographic.  

#### **13. Suitable Effect Techniques**  
- **Impact:** Enhances message delivery.  
- **Analysis:** Are **filters, lighting, or special effects** beneficial or distracting?  
- **Application:** Ensure effects **support the message rather than overshadow it**.  

#### **14. Key Message and Subject**  
- **Impact:** Clearly communicates the intended message.  
- **Analysis:** Is the message **easily understood at a glance**?  
- **Application:** Ensure the **main subject stands out** and supports the **intended action**.  

---

### **Structured Table for Analysis**  

| **Aspect**           | **Score (1-5)** | **Explanation** | **Suggested Improvement** |  
|----------------------|---------------|----------------|--------------------------|  
| **Emotional Appeal** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Eye Attraction**   | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Visual Appeal**    | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Text Overlay**     | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Contrast & Clarity** | _[Score]_    | _[Analysis]_   | _[Recommendation]_       |  
| **Visual Hierarchy** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Negative Space**   | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Color Psychology** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Depth & Texture**  | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Brand Consistency**| _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Psychological Triggers** | _[Score]_ | _[Analysis]_   | _[Recommendation]_       |  
| **Emotional Connection** | _[Score]_  | _[Analysis]_   | _[Recommendation]_       |  
| **Effect Techniques**| _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Key Message & Subject** | _[Score]_ | _[Analysis]_   | _[Recommendation]_       |  

---

### **Final Summary & Recommendations**  
- Provide an **overall summary** of the **image’s strengths and weaknesses**.  
- Identify **key areas that need improvement** and suggest **general enhancements**.
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
### **Objective:**  
Evaluate the **image** against multiple **marketing, design, and psychological** principles. Each aspect should be scored **from 1 to 5 in increments of 0.5** (1 being low, 5 being excellent), with an **explanation and suggested improvements**.  

- Present the results in a **structured table format** with columns:  
  **Aspect | Score | Explanation | Suggested Improvement**  
- After the table, provide an **overall summary** with **key recommendations for improvement**.  

---

### **Evaluation Criteria:**  

#### **1. Emotional Appeal**  
- **Impact:** Does the image evoke a strong emotional response?  
- **Analysis:** Identify the **emotions triggered** (e.g., happiness, nostalgia, excitement, urgency).  
- **Application:** Ensure the emotional tone **aligns with the campaign’s goal and the target audience’s expectations**.  

#### **2. Eye Attraction**  
- **Impact:** Does the image grab attention immediately?  
- **Analysis:** Identify which **visual elements** (color, subject, composition) are most effective.  
- **Application:** Optimize the focal point to **draw and maintain viewer attention**.  

#### **3. Visual Appeal**  
- **Impact:** How aesthetically pleasing is the image overall?  
- **Analysis:** Assess balance, **symmetry, composition, and creative visual techniques**.  
- **Application:** Ensure **harmonious elements** that enhance brand perception.  

#### **4. Text Overlay (Clarity, Emotional Connection, Readability)**  
- **Impact:** Is the text readable and well-integrated?  
- **Analysis:** Assess **contrast, font style, placement, and messaging impact**.  
- **Application:** Improve readability and emotional resonance **without overwhelming the design**.  

#### **5. Contrast and Clarity**  
- **Impact:** Highlights important elements and improves visibility.  
- **Analysis:** Check if **foreground and background elements** are distinct.  
- **Application:** Adjust contrast to **improve message visibility and clarity**.  

#### **6. Visual Hierarchy**  
- **Impact:** Guides the viewer’s **eye movement** naturally.  
- **Analysis:** Ensure the **key message, product, or CTA** is prioritized.  
- **Application:** Use **size, color, and placement** to direct attention effectively.  

#### **7. Negative Space**  
- **Impact:** Enhances balance and focus.  
- **Analysis:** Ensure negative space **prevents clutter and improves readability**.  
- **Application:** Optimize spacing to **improve clarity and organization**.  

#### **8. Color Psychology**  
- **Impact:** Influences **mood, perception, and attention**.  
- **Analysis:** Examine how the colors **evoke emotions and reinforce branding**.  
- **Application:** Use colors strategically to **strengthen brand identity and emotional response**.  

#### **9. Depth and Texture**  
- **Impact:** Enhances realism and engagement.  
- **Analysis:** Evaluate **shadows, gradients, and layering techniques**.  
- **Application:** Ensure texture and depth **add value without complicating the message**.  

#### **10. Brand Consistency**  
- **Impact:** Reinforces **brand recognition and trust**.  
- **Analysis:** Check if branding elements (**logo, fonts, colors**) are well-integrated.  
- **Application:** Maintain **consistent brand aesthetics** across all marketing assets.  

#### **11. Psychological Triggers**  
- **Impact:** Encourages desired actions.  
- **Analysis:** Identify triggers like **scarcity, authority, or social proof**.  
- **Application:** Use subtle psychological cues **without appearing manipulative**.  

#### **12. Emotional Connection**  
- **Impact:** Strengthens audience engagement.  
- **Analysis:** Does the image **align with audience values and desires**?  
- **Application:** Use **authentic messaging** that resonates with the target demographic.  

#### **13. Suitable Effect Techniques**  
- **Impact:** Enhances message delivery.  
- **Analysis:** Are **filters, lighting, or special effects** beneficial or distracting?  
- **Application:** Ensure effects **support the message rather than overshadow it**.  

#### **14. Key Message and Subject**  
- **Impact:** Clearly communicates the intended message.  
- **Analysis:** Is the message **easily understood at a glance**?  
- **Application:** Ensure the **main subject stands out** and supports the **intended action**.  

---

### **Structured Table for Analysis**  

| **Aspect**           | **Score (1-5)** | **Explanation** | **Suggested Improvement** |  
|----------------------|---------------|----------------|--------------------------|  
| **Emotional Appeal** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Eye Attraction**   | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Visual Appeal**    | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Text Overlay**     | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Contrast & Clarity** | _[Score]_    | _[Analysis]_   | _[Recommendation]_       |  
| **Visual Hierarchy** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Negative Space**   | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Color Psychology** | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Depth & Texture**  | _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Brand Consistency**| _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Psychological Triggers** | _[Score]_ | _[Analysis]_   | _[Recommendation]_       |  
| **Emotional Connection** | _[Score]_  | _[Analysis]_   | _[Recommendation]_       |  
| **Effect Techniques**| _[Score]_      | _[Analysis]_   | _[Recommendation]_       |  
| **Key Message & Subject** | _[Score]_ | _[Analysis]_   | _[Recommendation]_       |  

---

### **Final Summary & Recommendations**  
- Provide an **overall summary** of the **image’s strengths and weaknesses**.  
- Identify **key areas that need improvement** and suggest **general enhancements**.
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
### **Objective:**  
Analyze a **marketing asset** ({'image' if is_image else 'video'}) for a **detailed, structured, and marketing-focused** evaluation. The goal is to provide **objective insights** that help in making informed **marketing decisions**.

---

### **Evaluation Criteria:**  

#### **1. Detailed Description**  
✅ **For Images:**  
   - **Visual Elements:** Describe the prominent objects, people, animals, or settings.  
   - **Color Palette & Mood:** Identify dominant colors and their effect on perception.  
   - **Text Elements:** Describe any visible text, font style, size, and placement.  
   - **Composition & Layout:** Analyze how elements are positioned and balanced.  

✅ **For Videos:**  
   - **Key Scenes & Actions:** Describe what happens in the most representative frames.  
   - **Visual Style & Color Scheme:** Identify editing techniques and thematic consistency.  
   - **Text Overlays & Captions:** Transcribe any visible text or spoken content.  
   - **Audio Elements:** Note background music, sound effects, and voiceovers.  

---

#### **2. Cultural References and Symbolism**  
✅ **Cultural Sensitivity & Meaning:**  
   - Identify **symbols, gestures, or references** that may have **cultural significance**.  
   - Consider how these elements **align with the target audience’s values and perceptions**.  

✅ **Marketing Implications:**  
   - How might these cultural cues **influence brand perception**?  
   - Are there any potential **risks or misinterpretations** in global or diverse markets?  

---

#### **3. Marketing Implications**  
✅ **Target Audience Alignment:**  
   - How does the content appeal to **different demographics, interests, and emotions**?  
   - Does the design reinforce a **positive brand identity and message**?  

✅ **Brand Impact & Emotional Connection:**  
   - Does the asset create a strong **emotional response** that encourages engagement?  
   - Are there any **positive or negative associations** that could affect the brand?  

✅ **Conversion Potential:**  
   - Does the content **clearly support the marketing objective** (awareness, engagement, conversion)?  
   - How effective is the **call to action (CTA), if present**?  

---

#### **4. Additional Notes (For Videos)**  
✅ **Key Frame Selection:**  
   - Identify **the most visually impactful frames** for initial assessment.  
   - Mention **any significant transitions, cuts, or scene variations**.  

✅ **Overall Content Consistency:**  
   - Ensure **visual and textual elements** align with the **brand’s tone, style, and messaging**.  
   - Highlight **potential inconsistencies** that could affect the asset’s effectiveness.  

---

### **Presentation Format**  

#### **Structured Report Output:**  
**Section 1: Visual & Textual Breakdown**  
- **Asset Type:** [Image/Video]  
- **Primary Elements:** [List of key objects, people, settings]  
- **Color & Mood:** [Description of dominant colors & their effect]  
- **Text Presence:** [List of all visible text, font style, and placement]  
- **Composition & Layout:** [Analysis of element positioning & balance]  

**Section 2: Cultural References & Symbolism**  
- **Cultural Elements:** [Any relevant cultural themes, colors, or symbols]  
- **Potential Interpretations:** [How these might be perceived by different audiences]  
- **Marketing Considerations:** [Any risks or opportunities related to cultural context]  

**Section 3: Marketing Implications**  
- **Target Audience Fit:** [How the asset aligns with key demographics]  
- **Brand Reinforcement:** [Does it align with brand messaging?]  
- **Call to Action Effectiveness:** [Does it encourage engagement or conversion?]  
- **Potential Risks & Opportunities:** [Any elements that could help or hurt marketing effectiveness]  

**Section 4: Final Insights & Recommendations**  
- **Overall Effectiveness Score:** [1-5 scale]  
- **Improvement Areas:** [Clear, actionable suggestions for enhancement]
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
### **Objective:**  
Evaluate a **marketing asset** ({'image' if is_image else 'video'}) using **Self-Determination Theory (SDT)** to determine its ability to **motivate audience action and purchase decisions**. The analysis will assess how well the content satisfies three core psychological needs: **Autonomy, Competence, and Relatedness**—which drive intrinsic motivation.

---

### **Evaluation Method:**  

For each **SDT principle**, assign a score from **1 to 5** (increments of **0.5**), with **1 being low and 5 being high**. Provide a concise **explanation** and **specific recommendations for improvement**. Present findings in a **structured table** format.  

At the end of the evaluation, compute a **Motivational Score**, which is weighted as follows:  
- **Autonomy:** 50% of total score  
- **Competence:** 30% of total score  
- **Relatedness:** 20% of total score  

---

### **Evaluation Criteria**  

#### **1. Autonomy (The Need for Control & Choice)**  
✅ **Decision Empowerment**  
- Does the content allow consumers to feel in control of their decision?  
- Is the messaging **transparent** and **unbiased**, avoiding pressure tactics?  

✅ **Customization & Personalization**  
- Does the content offer **options or customizable features**?  
- Does it acknowledge **individual preferences and lifestyles**?  

✅ **Respect for Audience Intelligence**  
- Does the content avoid **manipulative language or fear-based tactics**?  
- Is the tone empowering rather than coercive?  

---

#### **2. Competence (The Need to Feel Effective & Capable)**  
✅ **Ease of Use & Understanding**  
- Does the content make the audience feel confident in using the product/service?  
- Are instructions or **benefits clearly explained**?  

✅ **Problem-Solving & Skill Enhancement**  
- Does the product **solve a problem or improve the audience’s capabilities**?  
- Is the value of the product communicated effectively?  

✅ **Social Proof & Success Stories**  
- Are there **testimonials, case studies, or real-life examples** that reinforce confidence?  
- Does the content demonstrate **how others have successfully used the product**?  

---

#### **3. Relatedness (The Need for Connection & Social Belonging)**  
✅ **Community & Social Validation**  
- Does the content create a **sense of belonging** through customer reviews or user-generated content?  
- Does it align with a **community, cause, or shared interest**?  

✅ **Emotional Resonance**  
- Does the content build **trust, warmth, and empathy** with the audience?  
- Is the **brand voice supportive and relationship-driven** rather than transactional?  

✅ **Alignment with Audience Values**  
- Does the messaging **reflect audience beliefs, aspirations, and social identity**?  
- How well does it **connect to a larger purpose** that matters to the target audience?  

---

### **Final Score Calculation**  
✅ **Motivational Score Formula:**  
- **50% of Autonomy Score**  
- **30% of Competence Score**  
- **20% of Relatedness Score**  

✅ **Final Recommendations & Action Plan**  
- Highlight key areas **for optimization** to improve engagement and conversion.  
- Offer **specific marketing strategy suggestions** aligned with SDT principles.  

---

### **Presentation Format**  

#### **Structured Report Output:**  
**Section 1: Content Breakdown**  
- **Asset Type:** [Image/Video]  
- **Primary Messaging Focus:** [Description of core content message]  
- **Emotional & Motivational Themes:** [Overview of how the content drives action]  

**Section 2: SDT-Based Evaluation**  
| Aspect         | Score (1-5) | Explanation | Suggested Improvement |  
|--------------|------------|-------------|------------------------|  
| **Autonomy** | _[Score]_ | _[Reasoning]_ | _[Enhancement Idea]_ |  
| **Competence** | _[Score]_ | _[Reasoning]_ | _[Enhancement Idea]_ |  
| **Relatedness** | _[Score]_ | _[Reasoning]_ | _[Enhancement Idea]_ |  

**Section 3: Final Insights**  
- **Motivational Score:** _[Weighted Calculation]_  
- **Overall Effectiveness:** _[Summary of how well the content encourages audience action]_  
- **Optimization Strategy:** _[Key recommendations for refinement]_ 
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
            st.image(image, caption="Uploaded Image", use_container_width='auto')
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
