import os
import io
import json
import base64
import tempfile
import re
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import google.generativeai as genai
import cv2
import imageio
import xml.etree.ElementTree as ET

# Load environment variables from .env file
load_dotenv()

# Get the API key and credentials file from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Check if credentials_path is set
if credentials_path is None:
    raise Exception("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please check your .env file.")
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

app = FastAPI()

def resize_image(image, max_size=(300, 250)):
    image.thumbnail(max_size)
    return image

def extract_frames(video_file_path, num_frames=5):
    """Extracts frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file {video_file_path}. Check if the file is corrupt or format is unsupported.")
    
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
        raise Exception("No frames were extracted, possibly due to an error in reading the video.")
    return frames

@app.post("/analyze_media")
async def analyze_media(uploaded_file: UploadFile = File(...), is_image: bool = True):
    prompt = (
        "Analyze the media (image or video frame) for various marketing aspects, ensuring consistent results for each aspect. "
        "Respond in single words or short phrases separated by commas for each attribute: text amount (High or Low), "
        "Color usage (Effective or Not effective), visual cues (Present or Absent), emotion (Positive or Negative), "
        "Focus (Central message or Scattered), customer-centric (Yes or No), credibility (High or Low), "
        "User interaction (High, Moderate, or Low), CTA presence (Yes or No), CTA clarity (Clear or Unclear)."
    )
    try:
        if is_image:
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name
    
            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
    
            response = model.generate_content([prompt, frames[0]])  # Analyzing the first frame
    
        attributes = ["text_amount", "color_usage", "visual_cues", "emotion", "focus", "customer_centric", "credibility", "user_interaction", "cta_presence", "cta_clarity"]
        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            values = raw_response.split(',')
            if len(attributes) == len(values):
                structured_response = {attr: val.strip() for attr, val in zip(attributes, values)}
                return structured_response
            else:
                raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
        else:
            raise HTTPException(status_code=500, detail="Model did not provide a response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/combined_marketing_analysis_v6")
async def combined_marketing_analysis_v6(uploaded_file: UploadFile = File(...), is_image: bool = True):
    prompt = """
    Analyze the provided image for marketing effectiveness. Begin by addressing the following key points:

    1. **Asset Type:** Identify and describe the type of marketing asset. Possible types include email, social media posts, advertisements, flyers, brochures, landing pages, etc.
    2. **Purpose:** State and explain the specific purpose of the marketing asset, such as selling a product, increasing sign-ups, driving traffic, enhancing brand awareness, or engaging customers.
    3. **Asset Audience:** Identify the target audience for the marketing asset, including their demographics, interests, and needs (age, gender, location, income level, education, etc.).

    Proceed to evaluate the asset on the following aspects, providing a score from 1 to 5 (in increments of 0.5), along with an explanation for each score and suggestions for improvement. Present the results in a table format with columns for Aspect, Score, Explanation, and Improvement:

    **Aspects to Consider:**
    1. **Creative Score:** Assess the design's creativity and its ability to capture attention through innovative elements.
    2. **Attention:** Evaluate the order in which content is consumed, starting with the headline, followed by text and images, and any interactive elements. Analyze if the content effectively prioritizes important information and maintains viewer attention.
    3. **Distinction:** Determine if the pictures and overall content design grab user attention and appeal on a visceral level, both with and without text.
    4. **Purpose and Value:** Assess if the asset's purpose and value are immediately clear within the first 3 seconds and whether the content is product or customer-centric.
    5. **Clarity:** Evaluate the clarity of design elements, including visuals and text.
    6. **First Impressions:** Analyze the initial impact of the design and whether it creates a positive first impression.
    7. **Cognitive Demand:** Evaluate how much cognitive effort is required to understand and navigate the design.
    8. **Headline Review:** Review the headline for clarity, customer centricity, SEO integration, emotional appeal, uniqueness, urgency, benefits, audience targeting, length, use of numbers/lists, brand consistency, and use of power words.
    9. **Headline Keywords and Emotional Appeal:** Assess if the headline includes effective keywords and evokes an emotional response.
    10. **Visual Cues and Color Usage:** Analyze how visual cues and color choices guide attention to key elements.
    11. **Labeling and Button Clarity:** Evaluate the clarity and effectiveness of any labels or buttons in terms of text size, font choice, and placement.
    12. **Engagement:** Assess the level of user engagement and satisfaction with the UX design.
    13. **Trust:** Evaluate the trustworthiness of the content based on visual and textual elements.
    14. **Motivation:** Assess how well the design aligns with user motivators and whether it demonstrates authority or provides social proof.
    15. **Influence:** Analyze the design's effectiveness in persuading viewers and leading them towards the desired action.
    16. **Calls to Action:** Examine the presence, prominence, and language of calls to action and their benefits.
    17. **Experience:** Assess the overall user experience and how well the design facilitates a smooth and enjoyable interaction.
    18. **Memorability:** Evaluate the memorability of the design.
    19. **Effort:** Assess the clarity and conciseness of the text and its effectiveness in conveying the message.
    20. **Tone:** Evaluate if the tone used increases the effectiveness of the asset.
    21. **Framing:** Assess if the framing of the message increases the effectiveness of the asset.

    Conclude with a concise overall analysis including general suggestions for improvement.
    """
    try:
        if is_image:
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name
    
            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")
    
            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis
    
        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/text_analysis")
async def text_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/headline_analysis")
async def headline_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")
@app.post("/headline_detailed_analysis")
async def headline_detailed_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/main_headline_detailed_analysis")
async def main_headline_detailed_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/image_headline_detailed_analysis")
async def image_headline_detailed_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/supporting_headline_detailed_analysis")
async def supporting_headline_detailed_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/main_headline_analysis")
async def main_headline_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
    prompt = """
    As a marketing consultant, analyze the main headline of a marketing asset ({'image' if is_image else 'video'}) for a client.
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/image_headline_analysis")
async def image_headline_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
    prompt = """
    Assess the distinct headline within an image or video's first frame as a marketing consultant.
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/supporting_headline_analysis")
async def supporting_headline_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
    prompt = """
    Review any supporting headlines in the provided image or video frame as a marketing consultant.
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
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise HTTPException(status_code=400, detail="No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            return {"results": response.candidates[0].content.parts[0].text.strip()}
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/flash_analysis")
async def flash_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/custom_prompt_analysis")
async def custom_prompt_analysis(uploaded_file: UploadFile = File(...), is_image: bool = True, custom_prompt: str = ""):
    if not custom_prompt:
        raise HTTPException(status_code=400, detail="Custom prompt is required.")
    
    try:
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.file.read()))
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.file.read())
                tmp_path = tmp.name
                frames = extract_frames(tmp_path)
                if not frames:
                    raise HTTPException(status_code=400, detail="No frames extracted from video")
                image = frames[0]  # Use the first frame for analysis

        response = model.generate_content([custom_prompt, image])

        if response.candidates and len(response.candidates[0].content.parts) > 0:
            return HTMLResponse(content=response.candidates[0].content.parts[0].text.strip())
        else:
            raise HTTPException(status_code=500, detail="Model did not provide a valid response or the response structure was unexpected.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/meta_profile")
async def meta_profile(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/linkedin_profile")
async def linkedin_profile(uploaded_file: UploadFile = File(...), is_image: bool = True):
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
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

@app.post("/x_profile")
async def x_profile(uploaded_file: UploadFile = File(...), is_image: bool = True):
    prompt = f"""
Based on the following targeting elements for X, please describe 4 persona types that are
most likely to respond to the add. Please present these in a table (Persona Type,
Description). Once you have identified these, create 4 personas (including names) who
would be likely to purchase this product, and describe how you would expect them to react
to it detailing the characteristics. Present each persona with a table (Persona Type,
Description, Analysis) of the characteristics and analysis. Please include each of the
characteristic that can be selected in the X targeting, and what you would select.

Location: Target users by country, region, or metro area. More granular targeting, such as
city or postal code, is also available.
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
            image = Image.open(io.BytesIO(await uploaded_file.read()))
            response = model.generate_content([prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:  # Check if frames were extracted successfully
                raise Exception("No frames were extracted from the video. Please check the video format.")

            response = model.generate_content([prompt, frames[0]])  # Using the first frame for analysis

        if response.candidates:
            raw_response = response.candidates[0].content.parts[0].text.strip()
            return HTMLResponse(content=raw_response)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response structure from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or process the media: {e}")

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Marketing Media Analysis AI Assistant API"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
