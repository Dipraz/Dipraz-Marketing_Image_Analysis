import os
import io
import json
import base64
import tempfile
import re
import ssl
import cv2
import imageio
import xml.etree.ElementTree as ET
from flask_talisman import Talisman
from threading import Thread
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key and credentials file from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Check if credentials_path is set
if credentials_path is None:
    raise Exception("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please check your .env file.")
else:
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_image(image, max_size=(300, 250)):
    """Resize image to a maximum size."""
    image.thumbnail(max_size)
    return image

def extract_frames(video_file_path, num_frames=5):
    """Extract frames from a video file using OpenCV."""
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
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
    
    cap.release()
    if len(frames) == 0:
        raise Exception("No frames were extracted, possibly due to an error in reading the video.")
    return frames

@app.before_request
def enforce_https_in_production():
    if not request.is_secure and not app.debug:
        url = request.url.replace("http://", "https://", 1)
        return redirect(url, code=301)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
@app.route('/analyze_multiple', methods=['POST'])
def analyze_multiple():
    uploaded_file = request.files.get('uploaded_file')
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400
    
    is_image = request.form.get('is_image', 'true').lower() == 'true'

    analysis_functions = [
        analyze_media,
        overall_analysis,
        story_telling_analysis,
        emotional_resonance,
        emotional_analysis,
        Emotional_Appraisal_Models,
        behavioural_principles,
        nlp_principles_analysis,
        text_analysis,
        Text_Analysis_2,
        Text_Analysis_2_table,
        headline_analysis,
        headline_detailed_analysis,
        main_headline_detailed_analysis,
        image_headline_detailed_analysis,
        supporting_headline_detailed_analysis,
        main_headline_analysis,
        image_headline_analysis,
        supporting_headline_analysis,
        meta_profile,
        linkedin_profile,
        x_profile,
        image_analysis,
        image_analysis_2,
        image_analysis_2_table
    ]

    results = {}
    for func in analysis_functions:
        try:
            result = func(uploaded_file, is_image)
            results.update(result)
        except Exception as e:
            results[func.__name__] = {"error": str(e)}

    return jsonify(results)

@app.route('/analyze_media', methods=['POST'])
def analyze_media():
    uploaded_file = request.files.get('uploaded_file')
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400
    
    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = "Your analysis prompt here"

    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/overall_analysis", methods=["GET", "POST"])
def overall_analysis():
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
            "10. Engagement: Assess the engagement level of the user experience. Is the UX design captivating and satisfying to interact with?\n"
            "11. Trust: Assess the trustworthiness of the content based on visual and textual elements. Is the content brand or customer-centric (customer-centric content has a higher trustworthiness)? Assess the credibility, reliability, and intimacy conveyed by the content.\n"
            "12. Motivation: Assess the design's ability to motivate users. Does it align with user motivators and demonstrate authority or provide social proof?\n"
            "13. Influence: Analyze the influence of the design. Does the asset effectively persuade viewers and lead them towards a desired action?\n"
            "14. Calls to Action: Analyze the presence, prominence, benefits, and language of CTAs.\n"
            "15. Experience: Assess the overall user experience. How well does the design facilitate a smooth and enjoyable interaction?\n"
            "16. Memorability: Evaluate how memorable the design is. Does it leave a lasting impression?\n"
            "17. Effort: Evaluate the clarity and conciseness of the text. Does it convey the message effectively without being overly wordy? (1: Very Dense & Difficult, 5: Clear & Easy to Understand)\n"
            "18. Tone: Is the tone used to increase the effectiveness of the asset effectively?\n"
            "19. Framing: Is framing of the message used to increase the effectiveness of the asset effectively?\n"
            "20. Content Investment: Blocks containing paragraphs of text will not be consumed by busy users and would require time to read – this is negative, as the users will not spend the time. Is the amount of content presented kept short and clear?\n"
        """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/story_telling_analysis", methods=["POST"])
def story_telling_analysis():
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
Storytelling has a significant impact on creative, enriching the content and enhancing its effectiveness in various ways. Here are some key impacts of storytelling on static creative:

1. Emotional Engagement
Impact: Storytelling evokes emotions, making the content more relatable and memorable.
Explanation: A well-crafted story can connect with the audience on an emotional level, fostering empathy, joy, sadness, or excitement. This emotional engagement makes the static creative more impactful.
Example: An image of a family enjoying a product can tell a story of togetherness and happiness, evoking positive emotions in the viewer.
2. Attention and Interest
Impact: Stories capture and hold the audience's attention.
Explanation: Humans are naturally drawn to stories. Incorporating a narrative element in static creative can intrigue viewers, encouraging them to spend more time engaging with the content.
Example: A before-and-after image showing the transformation of a product's user tells a story of change and improvement, keeping the viewer interested.
3. Memorability
Impact: Stories enhance recall and retention.
Explanation: Information presented within a story is easier to remember than standalone facts. Storytelling makes the content more memorable, ensuring that the audience retains the message.
Example: An image series depicting the journey of a product from creation to customer use embeds the brand story in the viewer's mind.
4. Brand Identity and Values
Impact: Storytelling conveys brand identity and values.
Explanation: Through stories, brands can express their mission, vision, and core values, building a strong identity. This helps in differentiating the brand from competitors and building loyalty.
Example: A static ad featuring a company’s founders working passionately on their first product conveys values of dedication and authenticity.
5. Simplification of Complex Messages
Impact: Stories simplify complex messages.
Explanation: Complex information can be conveyed more easily and understandably through storytelling. This makes the content more accessible and engaging for the audience.
Example: A static infographic that tells a story about the impact of climate change through visuals and short narratives simplifies a complex issue.
6. Connection and Trust
Impact: Stories build a connection and trust with the audience.
Explanation: Authentic stories foster trust and build a connection with the audience. When viewers relate to the story, they are more likely to trust the brand and its message.
Example: An image featuring testimonials from real customers sharing their success stories with the product builds credibility and trust.
7. Call to Action (CTA) Effectiveness
Impact: Storytelling enhances the effectiveness of CTAs.
Explanation: When a story is compelling, viewers are more likely to respond to the call to action. The narrative creates a context that makes the CTA more appealing and urgent.
Example: A static creative that tells a story of someone achieving their goals with the help of a product, followed by a CTA to “Join the success,” is more persuasive.
Practical Applications of Storytelling in Static Creative:
Visual Storytelling: Use images that depict a sequence or a moment that implies a broader story.

Example: An image of a person holding a graduation certificate can imply the story of hard work, achievement, and success.
Textual Elements: Incorporate short, compelling copy that suggests a narrative.

Example: A tagline like “From our family to yours” paired with a family photo tells a story of care and tradition.
Contextual Backgrounds: Use backgrounds and settings that imply a story.

Example: A product placed in a home setting can imply how it fits into daily life, telling a story of convenience and comfort.
Character and Journey: Introduce characters and show their journey.

Example: A static ad featuring a character's journey from a problem to a solution using the product.
User-Generated Content: Share stories from actual customers.

Example: Customer photos with quotes about their experiences tell authentic stories that resonate with new customers.

Evaluate the content using the 7 principles above. Score each element from 1-5, in increments of o.5. Please provide the information in a table, with: element, Score , evaluation, How it could be improved. at the end, please provide a summary of your recommendations.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500
@app.route("/emotional_resonance", methods=["POST"])
def emotional_resonance():
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
If the content is non-english, translate the content to English. Using the following model, please evaluate the content. Please also suggest improvements.

Evaluating the emotional resonance of a piece of content involves assessing how effectively it evokes the intended emotional responses in the target audience. Score each element from 1-5, in increments of o.5. Please provide the information in a table, with: element, Score , evaluation, how it could be improved. at the end, please provide recommendations. Here are key criteria to consider:

1. Clarity of Emotional Appeal
Criteria: The content clearly conveys the intended emotion(s).
Evaluation: Determine if the emotional message is easily understood without ambiguity.
2. Relevance to Target Audience
Criteria: The emotional appeal is relevant to the target audience’s experiences, values, and interests.
Evaluation: Assess if the content connects with the audience’s personal or professional life.
3. Authenticity
Criteria: The emotional appeal feels genuine and credible.
Evaluation: Check if the content avoids exaggeration and resonates as sincere and trustworthy.
4. Visual and Verbal Consistency
Criteria: Visual elements (images, colors, design) and verbal elements (language, tone) consistently support the emotional appeal.
Evaluation: Ensure that all elements of the content align to reinforce the intended emotion.
5. Emotional Intensity
Criteria: The strength of the emotional response elicited is appropriate for the context.
Evaluation: Measure whether the content evokes a strong enough emotional reaction without being overwhelming or underwhelming.
6. Engagement
Criteria: The content encourages audience engagement (likes, shares, comments, etc.).
Evaluation: Does the content explicitly encourage engagement, and have the means for users to share, like, comment etc.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/emotional_analysis", methods=["POST"])
def emotional_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
Using the following list of emotional resonance responses, assess whether the marketing content does or does not apply each. present the information in a table with columns: Name, Applies (None, some, A Lot), Definition, how it is applied, how it could be implemented. These are the principles to assess:

Here are different types of emotional resonance that can be leveraged in marketing to create a strong connection with the audience:

1. Empathy
Definition: The ability to understand and share the feelings of others.
Application: Crafting messages that show understanding of the audience's challenges and emotions.
2. Joy
Definition: A feeling of great pleasure and happiness.
Application: Creating content that makes the audience feel happy, excited, or entertained.
3. Surprise
Definition: A feeling of astonishment or shock caused by something unexpected.
Application: Using unexpected elements in marketing to capture attention and engage the audience.
4. Trust
Definition: Confidence in the honesty, integrity, and reliability of someone or something.
Application: Building trust through transparent communication, endorsements, and reliable information.
5. Fear
Definition: An unpleasant emotion caused by the belief that someone or something is dangerous.
Application: Highlighting potential risks or losses to motivate the audience to take action.
6. Sadness
Definition: A feeling of sorrow or unhappiness.
Application: Using stories or scenarios that evoke sympathy and compassion to drive support for a cause or product.
7. Anger
Definition: A strong feeling of displeasure or hostility.
Application: Addressing injustices or problems that provoke a sense of outrage, motivating the audience to seek solutions.
8. Anticipation
Definition: Excitement or anxiety about a future event.
Application: Creating a sense of excitement and eagerness for upcoming products, events, or announcements.
9. Disgust
Definition: A strong feeling of aversion or repulsion.
Application: Highlighting negative aspects of a competing product or undesirable conditions to steer the audience towards a better alternative.
10. Relief
Definition: A feeling of reassurance and relaxation following release from anxiety or distress.
Application: Positioning a product or service as a solution that alleviates worries or problems.
11. Love
Definition: A deep feeling of affection, attachment, or devotion.
Application: Creating campaigns that evoke feelings of love and affection towards family, friends, or the brand itself.
12. Pride
Definition: A feeling of deep pleasure or satisfaction derived from one's own achievements.
Application: Celebrating customer achievements and successes, making them feel proud of their association with the brand.
13. Belonging
Definition: The feeling of being accepted and included.
Application: Creating communities and fostering a sense of belonging among customers.
14. Nostalgia
Definition: A sentimental longing for the past.
Application: Using themes and imagery that evoke fond memories and a sense of nostalgia.
15. Hope
Definition: A feeling of expectation and desire for a particular thing to happen.
Application: Inspiring hope and optimism about the future through positive and uplifting messages.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/Emotional_Appraisal_Models", methods=["POST"])
def Emotional_Appraisal_Models():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
Firstly, translate any non-english text to english. Using the following emotional appraisal models, please evaluate the content. Please suggest possible  improvements against each model evaluation:

1. Lazarus’ Cognitive-Motivational-Relational Theory
Overview: Richard Lazarus proposed that emotions are the result of cognitive appraisals of events, which consider both personal relevance and coping potential.
Components:
Primary Appraisal: Evaluation of the significance of an event for personal well-being (e.g., Is this event beneficial or harmful?).
Secondary Appraisal: Evaluation of one's ability to cope with the event (e.g., Do I have the resources to deal with this?).
Core Relational Themes: Specific patterns of appraisal that lead to particular emotions (e.g., loss leads to sadness, threat leads to fear).
2. Scherer's Component Process Model (CPM)
Overview: Klaus Scherer’s model posits that emotions result from a sequence of appraisals along several dimensions.
Components:
Novelty: Is the event new or unexpected?
Pleasantness: Is the event pleasant or unpleasant?
Goal Significance: Does the event help or hinder the attainment of goals?
Coping Potential: Can the individual cope with or manage the event?
Norm Compatibility: Does the event conform to social and personal norms?
3. Smith and Ellsworth’s Appraisal Model
Overview: Craig Smith and Phoebe Ellsworth identified several dimensions of appraisal that influence emotional responses.
Components:
Attention: The degree to which the event draws attention.
Certainty: The certainty or predictability of the event.
Control/Coping: The degree of control one has over the event and the ability to cope.
Pleasantness: The pleasantness or unpleasantness of the event.
Perceived Obstacle: The extent to which the event is perceived as an obstacle to goals.
Responsibility: Who is responsible for the event (self, others, or circumstances).
Anticipated Effort: The amount of effort required to deal with the event.
4. Roseman’s Appraisal Theory
Overview: Ira Roseman’s model focuses on how appraisals of situations in terms of motivational congruence and agency influence emotions.
Components:
Motivational State: Whether the event is consistent or inconsistent with one’s goals.
Situational State: Whether the event is caused by the environment or the individual.
Probability: The likelihood of the event occurring.
Agency: Who is responsible for the event (self, other, or circumstance).
Power/Control: The degree of control one has over the event.
5. Weiner’s Attributional Theory of Emotion
Overview: Bernard Weiner’s model focuses on how attributions about the causes of events influence emotional reactions.
Components:
Locus: Whether the cause of the event is internal or external.
Stability: Whether the cause is stable or unstable over time.
Controllability: Whether the cause is controllable or uncontrollable by the individual.
6. Frijda’s Laws of Emotion
Overview: Nico Frijda proposed several “laws” that describe regularities in the relationship between appraisals and emotional responses.
Components:
Law of Situational Meaning: Emotions arise in response to meaning structures of situations.
Law of Concern: Emotions arise when events are relevant to one’s concerns.
Law of Apparent Reality: Emotions are elicited by events appraised as real.
Law of Change: Emotions are triggered by changes in circumstances.
Law of Habituation: Continuous exposure to a stimulus reduces its emotional impact.
Law of Comparative Feeling: Emotional intensity depends on comparisons with other events.
Law of Hedonic Asymmetry: Pleasure is more transient than pain.
Law of Conservation of Emotional Momentum: Emotions persist until the triggering conditions change.
7. Ellsworth’s Model of Appraisal Dimensions
Overview: Phoebe Ellsworth extended appraisal theory by emphasizing the importance of cultural and contextual factors in emotional appraisal.
Components:
Certainty: How certain one is about the event.
Attention: The extent to which the event captures attention.
Control: The degree of control one has over the event.
Pleasantness: Whether the event is perceived as pleasant or unpleasant.
Responsibility: Attribution of responsibility for the event.
Legitimacy: Whether the event is perceived as fair or unfair.
Practical Applications in Marketing
By understanding these emotional appraisal models, marketers can create content that:

Resonates with Core Concerns: Address the primary and secondary appraisals of the target audience.
Triggers Relevant Emotions: Design messages that align with specific appraisal dimensions to evoke desired emotional responses.
Enhances Perceived Control: Empower consumers by highlighting how products or services can help them manage or cope with challenges.
Builds Trust and Credibility: Ensure messages are consistent, predictable, and align with social norms to build trust.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/behavioural_principles", methods=["POST"])
def behavioural_principles():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
Using the following Behavioral Science principles, assess whether the marketing content does or does not apply each principle. Present the information in a table with columns: 'Applies the Principle (None, Some, A Lot)', 'Principle (Description)', 'Explanation', and 'How it could be implemented'. These are the principles to assess:

    1. Anchoring: The tendency to rely heavily on the first piece of information encountered (the "anchor") when making decisions.
        Example: Displaying a higher original price next to a discounted price to make the discount seem more substantial.
    2. Social Proof: People tend to follow the actions of others, assuming that those actions are correct.
        Example: Showing customer reviews and testimonials to build trust and encourage purchases.
    3. Scarcity: Items or opportunities become more desirable when they are perceived to be scarce or limited.
        Example: Using phrases like "limited time offer" or "only a few left in stock" to create urgency.
    4. Reciprocity: People feel obligated to return favors or kindnesses received from others.
        Example: Offering a free sample or trial to encourage future purchases.
    5. Loss Aversion: People prefer to avoid losses rather than acquire equivalent gains.
        Example: Emphasizing what customers stand to lose if they don't take action, such as missing out on a sale.
    6. Commitment and Consistency: Once people commit to something, they are more likely to follow through to maintain consistency.
        Example: Getting customers to make a small commitment first, like signing up for a newsletter, before asking for a larger commitment.
    7. Authority: People are more likely to trust and follow the advice of an authority figure.
        Example: Featuring endorsements from experts or industry leaders.
    8. Framing: The way information is presented can influence decision-making.
        Example: Highlighting the benefits of a product rather than the features, or framing a price as "only $1 a day" instead of "$30 a month".
    9. Endowment Effect: People value things more highly if they own them.
        Example: Allowing customers to try a product at home before making a purchase decision.
    10. Priming: Exposure to certain stimuli can influence subsequent behavior and decisions.
        Example: Using images and words that evoke positive emotions to enhance the appeal of a product.
    11. Decoy Effect: Adding a third option can make one of the original two options more attractive.
        Example: Introducing a higher-priced premium option to make the mid-tier option seem like better value.
    12. Default Effect: People tend to go with the default option presented to them.
        Example: Setting a popular product or service as the default selection on a website.
    13. Availability Heuristic: People judge the likelihood of events based on how easily examples come to mind.
        Example: Highlighting popular or recent customer success stories to create a perception of common positive outcomes.
    14. Cognitive Dissonance: The discomfort experienced when holding conflicting beliefs, leading to a change in attitude or behavior to reduce discomfort.
        Example: Reinforcing the positive aspects of a purchase to reduce buyer's remorse.
    15. Emotional Appeal: Emotions can significantly influence decision-making.
        Example: Using storytelling and emotional imagery to create a connection with the audience.
    16. Bandwagon Effect: People are more likely to do something if they see others doing it.
        Example: Showcasing the popularity of a product through sales numbers or social media mentions.
    17. Frequency Illusion (Baader-Meinhof Phenomenon): Once people notice something, they start seeing it everywhere.
        Example: Repeatedly exposing customers to a brand or product through various channels to increase recognition.
    18. In-group Favoritism: People prefer products or services associated with groups they identify with.
        Example: Creating marketing campaigns that resonate with specific demographics or communities.
    19. Hyperbolic Discounting: People prefer smaller, immediate rewards over larger, delayed rewards.
        Example: Offering instant discounts or rewards for immediate purchases.
    20. Paradox of Choice: Having too many options can lead to decision paralysis.
        Example: Simplifying choices by offering curated selections or recommended products.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/nlp_principles_analysis", methods=["POST"])
def nlp_principles_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
Using the following Neuro-Linguistic Programming (NLP) techniques, assess whether the marketing content does or does not apply each principle. present the information in a table with columns: Applies the principle (None, some, A Lot), Principle (Description), Explanation, how it could be implemented. These are the principles to assess:

Here are the top 20 Neuro-Linguistic Programming (NLP) techniques to assess the effectiveness of static marketing content, including examples:

Representational Systems:

Example: If your target audience prefers visual information, ensure the content includes vivid images and visually appealing graphics.
Anchoring:

Example: Use consistent colors and logos to create positive associations with your brand every time the audience sees them.
Meta-Modeling:

Example: Clarify ambiguous statements like "Our product is the best" by specifying "Our product is rated #1 for quality by Consumer Reports."
Milton Model:

Example: Use phrases like "You may find yourself feeling more relaxed when using our product" to embed suggestions subtly.
Chunking:

Example: Provide both high-level benefits (chunking up) and detailed features (chunking down) of your product to cater to different audience needs.
Pacing and Leading:

Example: Start with a relatable problem (pacing) like "Do you struggle with time management?" and lead to your solution: "Our planner can help you stay organized and efficient."
Swish Pattern:

Example: Replace negative images (e.g., cluttered desk) with positive images (e.g., clean, organized workspace) in your content.
Submodalities:

Example: Use bright, bold colors for calls to action to evoke excitement and urgency.
Perceptual Positions:

Example: Present content from the user's perspective ("You will benefit from..."), from others' perspectives ("Others will admire your..."), and from an observer's perspective ("Imagine the positive impact...").
Well-Formed Outcomes:

Example: Clearly state the desired outcome: "Increase your productivity by 20% with our planner in just one month."
Rapport Building:

Example: Use language that resonates with your audience’s values and experiences: "We understand how hectic life can be, and we’re here to help."
Calibration:

Example: Monitor engagement metrics like click-through rates and adjust content accordingly to better meet audience needs.
Reframing:

Example: Turn a negative situation into a positive opportunity: "Stuck in traffic? Use this time to listen to our educational podcasts and learn something new."
Logical Levels:

Example: Ensure your content addresses different levels, from environment ("Work anywhere") to identity ("Be a proactive leader").
Timeline Therapy:

Example: Highlight past successes, current benefits, and future potential: "Our product has helped thousands, it’s helping people right now, and it can help you too."
Meta Programs:

Example: Tailor content to different motivational patterns, such as "towards" goals ("Achieve your dreams with our help") or "away from" problems ("Avoid stress with our solution").
Strategy Elicitation:

Example: Show step-by-step how to use your product to achieve desired results, aligning with the audience's decision-making strategies.
Sensory Acuity:

Example: Use descriptive language that appeals to the senses: "Feel the soft texture, see the vibrant colors, and hear the clear sound."
Pattern Interrupts:

Example: Include unexpected elements like surprising statistics or bold images to capture attention and break habitual thought patterns.
Belief Change Techniques:

Example: Challenge limiting beliefs with testimonials or case studies that show successful outcomes, shifting beliefs towards the positive.
By utilizing these NLP techniques, you can create static marketing content that is more engaging, persuasive, and effective in achieving your marketing goals.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/text_analysis", methods=["POST"])
def text_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
As a UX design and marketing analysis consultant, you are tasked with reviewing the text content of a marketing asset (image or video, excluding the headline) for a client. Your goal is to provide a comprehensive analysis of the text's effectiveness and offer actionable recommendations for improvement, making sure that all responses are provided in English.
**Important:** Please provide all your analysis and recommendations in English, regardless of the language used in the original marketing asset.
**Part 1: Text Extraction and Contextualization**

* **Image Analysis:**
  1. **Text Extraction:** Thoroughly identify and extract ALL visible text within the image, including headlines, body copy, captions, calls to action, taglines, logos, and any other textual elements. Also, Translate non-English text to English.
  2. **Presentation:** Present the extracted text in a clear, bulleted list format, maintaining the original order and structure as much as possible.
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/Text_Analysis_2", methods=["POST"])
def Text_Analysis_2():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
If the content is non-english, translate the content to English. PLease evaluate the image against these principles:

1. Textual Analysis
Readability Analysis: Use tools like the Flesch-Kincaid readability tests to determine how easy the content is to read. This helps ensure that the language is appropriate for the target audience.
Lexical Diversity: Analyze the variety of words used in the content. High lexical diversity can indicate richness in language, which can be engaging, while lower diversity might be simpler and clearer.
2. Semantic Analysis
Keyword Analysis: Evaluate the frequency and placement of key terms related to the brand or product. Ensure that the most important keywords are prominently featured and well-integrated.
Topic Modeling: Use techniques like Latent Dirichlet Allocation (LDA) to identify the main topics covered in the content. This helps in understanding if the content aligns with the intended message and themes.
3. Sentiment Analysis
Polarity Assessment: Use natural language processing (NLP) tools to analyze the sentiment of the content, categorizing it as positive, negative, or neutral. This helps in ensuring the tone matches the intended emotional impact.
Emotion Detection: Beyond simple sentiment, more advanced NLP tools can detect specific emotions (joy, anger, sadness, etc.) conveyed by the content.
4. Structural Analysis
Narrative Structure: Examine the structure of the content to ensure it follows a logical flow. For instance, a typical narrative structure might include an introduction, problem statement, solution, and conclusion.
Visual Composition Analysis: For visual marketing content, analyze the layout, use of colors, fonts, and imagery. Ensure that these elements are aligned with branding guidelines and are aesthetically pleasing.
5. Linguistic Style Matching
Consistency with Brand Voice: Analyze if the content maintains consistency with the established brand voice and style guidelines. This involves checking for tone, style, and terminology.
Grammar and Syntax Analysis: Use grammar checking tools to ensure the content is free from grammatical errors and awkward phrasing.
6. Cohesion and Coherence Analysis
Cohesion Metrics: Measure how well different parts of the text link together. Tools like Coh-Metrix can provide insights into the coherence of the content.
Logical Flow: Evaluate the logical progression of ideas to ensure the content flows smoothly and makes logical sense from start to finish.
7. Visual and Multimodal Analysis
Image and Text Alignment: Analyze the relationship between text and images in the content. Ensure that images support and enhance the message conveyed by the text.
Aesthetic Quality: Evaluate the aesthetic elements of visual content, considering aspects like balance, symmetry, color harmony, and typography.
8. Compliance and Ethical Analysis
Regulatory Compliance: Ensure that the content complies with advertising regulations and industry standards.
Ethical Considerations: Analyze the content for any potential ethical issues, such as misleading claims, cultural insensitivity, or inappropriate content.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/Text_Analysis_2_table", methods=["POST"])
def Text_Analysis_2_table():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
If the content is non-english, translate the content to English. PLease evaluate the image against these principles in a table with a score for each element and sub element, from 1-5, in increments of 0.5. Please also include columns for analysis and  recommendations:

1. Textual Analysis
Readability Analysis: Use tools like the Flesch-Kincaid readability tests to determine how easy the content is to read. This helps ensure that the language is appropriate for the target audience.
Lexical Diversity: Analyze the variety of words used in the content. High lexical diversity can indicate richness in language, which can be engaging, while lower diversity might be simpler and clearer.
2. Semantic Analysis
Keyword Analysis: Evaluate the frequency and placement of key terms related to the brand or product. Ensure that the most important keywords are prominently featured and well-integrated.
Topic Modeling: Use techniques like Latent Dirichlet Allocation (LDA) to identify the main topics covered in the content. This helps in understanding if the content aligns with the intended message and themes.
3. Sentiment Analysis
Polarity Assessment: Use natural language processing (NLP) tools to analyze the sentiment of the content, categorizing it as positive, negative, or neutral. This helps in ensuring the tone matches the intended emotional impact.
Emotion Detection: Beyond simple sentiment, more advanced NLP tools can detect specific emotions (joy, anger, sadness, etc.) conveyed by the content.
4. Structural Analysis
Narrative Structure: Examine the structure of the content to ensure it follows a logical flow. For instance, a typical narrative structure might include an introduction, problem statement, solution, and conclusion.
Visual Composition Analysis: For visual marketing content, analyze the layout, use of colors, fonts, and imagery. Ensure that these elements are aligned with branding guidelines and are aesthetically pleasing.
5. Linguistic Style Matching
Consistency with Brand Voice: Analyze if the content maintains consistency with the established brand voice and style guidelines. This involves checking for tone, style, and terminology.
Grammar and Syntax Analysis: Use grammar checking tools to ensure the content is free from grammatical errors and awkward phrasing.
6. Cohesion and Coherence Analysis
Cohesion Metrics: Measure how well different parts of the text link together. Tools like Coh-Metrix can provide insights into the coherence of the content.
Logical Flow: Evaluate the logical progression of ideas to ensure the content flows smoothly and makes logical sense from start to finish.
7. Visual and Multimodal Analysis
Image and Text Alignment: Analyze the relationship between text and images in the content. Ensure that images support and enhance the message conveyed by the text.
Aesthetic Quality: Evaluate the aesthetic elements of visual content, considering aspects like balance, symmetry, color harmony, and typography.
8. Compliance and Ethical Analysis
Regulatory Compliance: Ensure that the content complies with advertising regulations and industry standards.
Ethical Considerations: Analyze the content for any potential ethical issues, such as misleading claims, cultural insensitivity, or inappropriate content.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/headline_analysis", methods=["POST"])
def headline_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/headline_detailed_analysis", methods=["POST"])
def headline_detailed_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/main_headline_detailed_analysis", methods=["POST"])
def main_headline_detailed_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/image_headline_detailed_analysis", methods=["POST"])
def image_headline_detailed_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/supporting_headline_detailed_analysis", methods=["POST"])
def supporting_headline_detailed_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/main_headline_analysis", methods=["POST"])
def main_headline_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/image_headline_analysis", methods=["POST"])
def image_headline_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/supporting_headline_analysis", methods=["POST"])
def supporting_headline_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = """
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/flash_analysis", methods=["POST"])
def flash_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/custom_prompt_analysis", methods=["POST"])
def custom_prompt_analysis():
    uploaded_file = request.files.get('uploaded_file')
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    custom_prompt = request.form.get('custom_prompt')
    
    if not custom_prompt:
        return jsonify({"error": "Custom prompt is required."}), 400

    try:
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            response = model.generate_content([custom_prompt, image])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                frames = extract_frames(tmp_path)
                if not frames:
                    return jsonify({"error": "No frames extracted from video"}), 400
                image = frames[0]  # Use the first frame for analysis

        if response.candidates and len(response.candidates[0].content.parts) > 0:
            return Response(response.candidates[0].content.parts[0].text.strip(), content_type="text/html")
        else:
            return jsonify({"error": "Unexpected response structure from the model."}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/meta_profile", methods=["POST"])
def meta_profile():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/linkedin_profile", methods=["POST"])
def linkedin_profile():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/x_profile", methods=["POST"])
def x_profile():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
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
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/Image_Analysis", methods=["POST"])
def image_analysis():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = f"""
    For each aspect listed below, provide a score from 1 to 5 in increments of 0.5 (1 being low, 5 being high) and an explanation for each aspect, along with suggestions for improvement. The results should be presented in a table format with the columns: Aspect, Score, Explanation, and Improvement. After the table, provide an explanation with suggestions for overall improvement. Here are the aspects to consider:

    Visual Appeal
    Impact: Attracts attention and conveys emotions.
    Analysis: Assess color scheme, composition, clarity, and aesthetic quality.
    Application: Ensure the image is clear, visually appealing, and professionally designed.

    Relevance
    Impact: Resonates with the target audience.
    Analysis: Determine if the image matches audience preferences, context, and brand alignment.
    Application: Align the image with the audience’s interests and brand values.

    Emotional Impact
    Impact: Evokes desired emotions.
    Analysis: Analyze the emotional resonance of the image.
    Application: Use storytelling and relatable scenarios to connect emotionally with the audience.

    Message Clarity
    Impact: Communicates the intended message effectively.
    Analysis: Ensure the main subject is clear and the image is not cluttered.
    Application: Focus on the key message and keep the design simple and straightforward.

    Engagement Potential
    Impact: Captures and retains audience attention.
    Analysis: Evaluate attention-grabbing aspects and interaction potential.
    Application: Use compelling visuals and narratives to encourage interaction.

    Brand Recognition
    Impact: Enhances brand recall and association.
    Analysis: Check for visible and well-integrated brand elements.
    Application: Use brand colors, logos, and consistent style to reinforce brand identity.

    Cultural Sensitivity
    Impact: Respects and represents cultural norms and diversity.
    Analysis: Assess inclusivity, cultural appropriateness, and global appeal.
    Application: Ensure the image is inclusive and culturally sensitive.

    Technical Quality
    Impact: Maintains high resolution and professional editing.
    Analysis: Evaluate resolution, lighting, and post-processing quality.
    Application: Use high-resolution images with proper lighting and professional editing.

    Color
    Impact: Influences mood, perception, and attention.
    Analysis: Analyze the psychological impact of the colors used.
    Application: Use colors purposefully to evoke desired emotions and enhance brand recognition.

    Typography
    Impact: Affects readability and engagement.
    Analysis: Assess font choice, size, placement, and readability.
    Application: Ensure typography complements the image and enhances readability.

    Symbolism
    Impact: Conveys complex ideas quickly.
    Analysis: Examine the use of symbols and icons.
    Application: Use universally recognized symbols that align with the ad’s message.

    Contrast
    Impact: Highlights important elements and improves visibility.
    Analysis: Check the contrast between different elements.
    Application: Use contrast to draw attention to key parts of the image.

    Layout Balance
    Impact: Ensures the image is visually balanced and pleasing to the eye.
    Analysis: Assess the distribution of elements within the image to ensure they are evenly balanced.
    Application: Arrange elements so that the visual weight is evenly distributed, avoiding clutter and ensuring harmony.

    Hierarchy
    Impact: Guides the viewer’s eye through the most important elements first.
    Analysis: Evaluate the visual hierarchy to ensure the most important elements stand out.
    Application: Use size, color, and placement to create a clear visual hierarchy, directing attention to key messages or elements.
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/Image_Analysis_2", methods=["POST"])
def image_analysis_2():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = f"""
    If the content is non-english, translate the content to English. Please evaluate the image against these principles:

    1. Emotional Appeal
    Does the image evoke a strong emotional response?
    What specific emotions are triggered (e.g., happiness, nostalgia, excitement, urgency)?
    How might these emotions influence the viewer's perception of the brand or product?
    How well does the image align with the intended emotional tone of the campaign?
    Does the emotional tone match the target audience's expectations and values?

    2. Eye Attraction
    Does the image grab attention immediately?
    Which elements (color, subject, composition) are most effective in drawing the viewer’s attention?
    Is there anything in the image that distracts from the main focal point?
    Is there a clear focal point in the image that naturally draws the viewer's eye?
    How effectively does the focal point communicate the key message or subject?

    3. Visual Appeal
    How aesthetically pleasing is the image overall?
    Are the elements of balance, symmetry, and composition well-executed?
    Does the image use any unique or creative visual techniques that enhance its appeal?
    Are the visual elements harmonious and balanced?
    Do any elements feel out of place or clash with the overall design?

    4. Text Overlay (Clarity, Emotional Connection, Readability)
    Is the text overlay easily readable?
    Is there sufficient contrast between the text and the background?
    Are font size, style, and color appropriate for readability?
    Does the text complement the image?
    Does it enhance the emotional connection with the audience?
    Is the messaging clear, concise, and impactful?
    Is the text aligned with the brand's identity?
    Does it maintain consistency with the brand’s tone and voice?

    5. Contrast and Clarity
    Is there adequate contrast between different elements of the image?
    How well do the foreground and background elements distinguish themselves?
    Does the contrast help highlight the key message or subject?
    Is the image clear and sharp?
    Are all important details easy to distinguish?
    Does the image suffer from any blurriness or pixelation?

    6. Visual Hierarchy
    Is there a clear visual hierarchy guiding the viewer’s eye?
    Are the most important elements (e.g., brand name, product, call to action) placed prominently?
    How effectively does the hierarchy direct attention from one element to the next?
    Are key elements ordered in terms of importance?
    Does the visual flow help reinforce the intended message?

    7. Negative Space
    Is negative space used effectively to balance the composition?
    Does the negative space help focus attention on the key elements?
    Is there enough negative space to avoid clutter without making the image feel empty?
    Does the use of negative space enhance the overall clarity of the message?
    How does it contribute to the image’s visual hierarchy and readability?

    8. Color Psychology
    Are the colors used in the image appropriate for the message and target audience?
    Do the colors evoke the intended emotional response (e.g., trust, excitement, calm)?
    Are any colors potentially off-putting or conflicting for the audience?
    How well do the colors align with the brand’s color palette?
    Are they consistent with the brand’s identity and overall messaging?
    Does the color scheme contribute to or detract from brand recognition?

    9. Depth and Texture
    Does the image have a sense of depth and texture?
    Are shadows, gradients, or layering techniques used effectively to create a three-dimensional feel?
    How does the depth or texture contribute to the realism and engagement of the image?
    Is the texture or depth distracting or enhancing?
    Does it add value to the visual appeal, or does it complicate the message?

    10. Brand Consistency
    Is the image consistent with the brand’s visual identity?
    Are color schemes, fonts, and overall style in line with the brand guidelines?
    Does the image reinforce the brand’s core values and messaging?
    Does the image maintain a coherent connection to previous branding efforts?
    Is there a risk of confusing the audience with a departure from established brand aesthetics?

    11. Psychological Triggers
    Does the image use any psychological triggers effectively?
    Are elements like scarcity, social proof, or authority present to encourage a desired action?
    How well do these triggers align with the target audience’s motivations and behaviors?
    Are the psychological triggers subtle or overt?
    Does the image risk appearing manipulative, or is the influence balanced and respectful?

    12. Emotional Connection
    How strong is the emotional connection between the image and the target audience?
    Does the image resonate with the audience’s values, desires, or pain points?
    Is the connection likely to inspire action or loyalty?
    Is the emotional connection authentic?
    Does it feel genuine, or is there a risk of the audience perceiving it as forced or inauthentic?

    13. Suitable Effect Techniques
    Are any special effects or filters used in the image?
    Do they enhance the overall message and visual appeal?
    Are the effects aligned with the brand’s identity and the image’s purpose?
    Do these effects support the key message and theme?
    Is there a risk of the effects distracting from or diluting the message?

    14. Key Message and Subject
    Is the key message of the image clear and easily understood at a glance?
    Is the message prominent, or does it get lost in other elements?
    How well does the image communicate its purpose or call to action?
    Is the subject of the image (product, service, idea) highlighted appropriately?
    Does the subject stand out as the main focus?
    Is there a clear connection between the subject and the intended message?
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500


@app.route("/Image_Analysis_2_table", methods=['GET', 'POST'])
def image_analysis_2_table():
    # Get the uploaded file from the request
    uploaded_file = request.files.get('uploaded_file')

    # Debugging: Check if the file was received and print details
    if uploaded_file:
        print(f"Received file: {uploaded_file.filename}")
    else:
        print("No file received.")

    # Check if the file is uploaded and has a valid extension
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type or no file uploaded"}), 400

    is_image = request.form.get('is_image', 'true').lower() == 'true'
    prompt = f"""
    If the content is non-english, translate the content to English. Please evaluate the image against these principles in a table with a score for each element, from 1-5, in increments of 0.5. Please also include columns for analysis and  recommendations:

    1. Emotional Appeal
    Does the image evoke a strong emotional response?
    What specific emotions are triggered (e.g., happiness, nostalgia, excitement, urgency)?
    How might these emotions influence the viewer's perception of the brand or product?
    How well does the image align with the intended emotional tone of the campaign?
    Does the emotional tone match the target audience's expectations and values?

    2. Eye Attraction
    Does the image grab attention immediately?
    Which elements (color, subject, composition) are most effective in drawing the viewer’s attention?
    Is there anything in the image that distracts from the main focal point?
    Is there a clear focal point in the image that naturally draws the viewer's eye?
    How effectively does the focal point communicate the key message or subject?

    3. Visual Appeal
    How aesthetically pleasing is the image overall?
    Are the elements of balance, symmetry, and composition well-executed?
    Does the image use any unique or creative visual techniques that enhance its appeal?
    Are the visual elements harmonious and balanced?
    Do any elements feel out of place or clash with the overall design?

    4. Text Overlay (Clarity, Emotional Connection, Readability)
    Is the text overlay easily readable?
    Is there sufficient contrast between the text and the background?
    Are font size, style, and color appropriate for readability?
    Does the text complement the image?
    Does it enhance the emotional connection with the audience?
    Is the messaging clear, concise, and impactful?
    Is the text aligned with the brand's identity?
    Does it maintain consistency with the brand’s tone and voice?

    5. Contrast and Clarity
    Is there adequate contrast between different elements of the image?
    How well do the foreground and background elements distinguish themselves?
    Does the contrast help highlight the key message or subject?
    Is the image clear and sharp?
    Are all important details easy to distinguish?
    Does the image suffer from any blurriness or pixelation?

    6. Visual Hierarchy
    Is there a clear visual hierarchy guiding the viewer’s eye?
    Are the most important elements (e.g., brand name, product, call to action) placed prominently?
    How effectively does the hierarchy direct attention from one element to the next?
    Are key elements ordered in terms of importance?
    Does the visual flow help reinforce the intended message?

    7. Negative Space
    Is negative space used effectively to balance the composition?
    Does the negative space help focus attention on the key elements?
    Is there enough negative space to avoid clutter without making the image feel empty?
    Does the use of negative space enhance the overall clarity of the message?
    How does it contribute to the image’s visual hierarchy and readability?

    8. Color Psychology
    Are the colors used in the image appropriate for the message and target audience?
    Do the colors evoke the intended emotional response (e.g., trust, excitement, calm)?
    Are any colors potentially off-putting or conflicting for the audience?
    How well do the colors align with the brand’s color palette?
    Are they consistent with the brand’s identity and overall messaging?
    Does the color scheme contribute to or detract from brand recognition?

    9. Depth and Texture
    Does the image have a sense of depth and texture?
    Are shadows, gradients, or layering techniques used effectively to create a three-dimensional feel?
    How does the depth or texture contribute to the realism and engagement of the image?
    Is the texture or depth distracting or enhancing?
    Does it add value to the visual appeal, or does it complicate the message?

    10. Brand Consistency
    Is the image consistent with the brand’s visual identity?
    Are color schemes, fonts, and overall style in line with the brand guidelines?
    Does the image reinforce the brand’s core values and messaging?
    Does the image maintain a coherent connection to previous branding efforts?
    Is there a risk of confusing the audience with a departure from established brand aesthetics?

    11. Psychological Triggers
    Does the image use any psychological triggers effectively?
    Are elements like scarcity, social proof, or authority present to encourage a desired action?
    How well do these triggers align with the target audience’s motivations and behaviors?
    Are the psychological triggers subtle or overt?
    Does the image risk appearing manipulative, or is the influence balanced and respectful?

    12. Emotional Connection
    How strong is the emotional connection between the image and the target audience?
    Does the image resonate with the audience’s values, desires, or pain points?
    Is the connection likely to inspire action or loyalty?
    Is the emotional connection authentic?
    Does it feel genuine, or is there a risk of the audience perceiving it as forced or inauthentic?

    13. Suitable Effect Techniques
    Are any special effects or filters used in the image?
    Do they enhance the overall message and visual appeal?
    Are the effects aligned with the brand’s identity and the image’s purpose?
    Do these effects support the key message and theme?
    Is there a risk of the effects distracting from or diluting the message?

    14. Key Message and Subject
    Is the key message of the image clear and easily understood at a glance?
    Is the message prominent, or does it get lost in other elements?
    How well does the image communicate its purpose or call to action?
    Is the subject of the image (product, service, idea) highlighted appropriately?
    Does the subject stand out as the main focus?
    Is there a clear connection between the subject and the intended message?
    """
    try:
        responses = []
        if is_image:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            responses = [model.generate_content([prompt, image]) for _ in range(3)]  # Send three requests
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            if frames is None or not frames:
                raise Exception("No frames were extracted from the video. Please check the video format.")
            
            responses = [model.generate_content([prompt, frames[0]]) for _ in range(3)]  # Send three requests

        # Merge responses
        merged_response = " ".join([resp.candidates[0].content.parts[0].text.strip() for resp in responses])
        return jsonify({"content": merged_response})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Failed to read or process the media: {e}"}), 500

@app.route("/", methods=["GET"])
def read_root():
    return {"message": "Welcome to the AI analysis Flask app!"}

@app.before_request
def enforce_https_in_production():
    if not request.is_secure:
        if request.headers.get('X-Forwarded-Proto', 'http') != 'https':
            url = request.url.replace("http://", "https://", 1)
            return redirect(url, code=301)
Talisman(app)        
if __name__ == "__main__":
    # Set up SSL context for HTTPS
    context = ('cert.pem', 'key.pem')  # Path to your SSL certificate and key
    http_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=80))
    https_thread = Thread(target=lambda: app.run(ssl_context=context, host='0.0.0.0', port=443))
    
    http_thread.start()
    https_thread.start()
    
    http_thread.join()
    https_thread.join()
