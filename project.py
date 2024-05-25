import os
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Check if GOOGLE_APPLICATION_CREDENTIALS environment variable is set
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if credentials_path is None:
    st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please check your .env file.")
else:
    # Set the Google application credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Configure the Generative AI API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("GENAI_API_KEY environment variable not set. Please check your .env file.")
    else:
        genai.configure(api_key=api_key)

        # Define generation configuration
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 5000,
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

        def object_detection_analysis(uploaded_file, model):
            prompt = (
                "You are an object detection specialist reviewing an image for a client. "
                "Your task is to analyze the provided image and detect objects within it. "
                "For each identified object, please provide the following details: \n\n"
                
                "1. **Object Type**: The type of object detected (e.g., person, car, dog, etc.).\n"
                "2. **Confidence Level**: Your confidence level in the detection, expressed as a percentage.\n"
                "3. **Bounding Box**: The coordinates of the bounding box around the detected object, formatted as (x_min, y_min, x_max, y_max).\n"
                "4. **Additional Notes**: Any relevant notes or observations about the detected object (e.g., partial occlusion, unusual appearance, etc.).\n\n"
                
                "Please respond with the details in the following format for each detected object:\n"
                "[Object Type, Confidence Level, Bounding Box, Additional Notes].\n\n"
                
                "After listing the detected objects, provide a brief explanation of the results, summarizing the overall detection accuracy and any notable observations."
            )

            image = Image.open(uploaded_file)
            response = model.generate_content([prompt, image])

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Detailed Object Detection Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None

        def sentiment_analysis_from_image(uploaded_file, model):
            prompt = (
                "Imagine you are an AI sentiment analysis specialist reviewing an image for a client. Your task is to analyze the provided image and determine the overall sentiment it conveys. For each identified sentiment, provide the following details: "
                "1. Sentiment: The primary emotion conveyed by the image (e.g., happy, sad, angry, surprised, neutral, etc.)."
                "2. Confidence Level: Your confidence level in the sentiment analysis, expressed as a percentage."
                "3. Facial Expressions: Describe the facial expressions of any people in the image and how they contribute to the sentiment."
                "4. Body Language: Describe the body language of any people in the image and how it influences the sentiment."
                "5. Color Usage: Analyze how the colors in the image contribute to the overall sentiment."
                "6. Objects and Setting: Identify any significant objects or settings in the image and explain their impact on the sentiment."
                "7. Contextual Notes: Any additional contextual observations that influence the sentiment (e.g., cultural context, event setting, etc.)."
                "8. Overall Impact: Assess the combined impact of all elements in the image on the conveyed sentiment."
                "\n\n"
                "Respond with the details in the following format for each identified sentiment: "
                "[Sentiment, Confidence Level, Facial Expressions, Body Language, Color Usage, Objects and Setting, Contextual Notes, Overall Impact]."
                "\n\n"
                "Finally, provide an overall summary of the image's sentiment, summarizing the key findings and their implications."
            )

            image = Image.open(uploaded_file)
            response = model.generate_content([prompt, image])

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Detailed Sentiment Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None

        def people_details_analysis(uploaded_file, model):
            prompt = (
                "Imagine you are an AI specialist analyzing an image to extract detailed information about the people present in it. For each person identified in the image, provide the following details: "
                "1. Number of People: The total number of people present in the image."
                "2. Gender: The gender of each person (male or female). Provide the total count for each gender."
                "3. Age: The approximate age of each person. Provide the total count for each age group."
                "4. Approximate Height: Provide the approximate height of each person."
                "5. Clothing Description: Describe the clothing and any notable accessories each person is wearing."
                "6. Facial Expressions: Describe the facial expressions of each person."
                "7. Body Language: Describe the body language and posture of each person."
                "8. Position in Image: Specify the position of each person in the image (e.g., center, left, right, background, foreground)."
                "9. Activity or Action: Describe any activity or action each person is engaged in."
                "10. Interaction: Note any interactions between the people in the image."
                "\n\n"
                "Respond with the details in the following format for each person: "
                "[Gender, Age, Approximate Height, Clothing Description, Facial Expressions, Body Language, Position in Image, Activity or Action, Interaction]."
                "\n\n"
                "Finally, provide an overall summary of the people in the image, summarizing the key observations and their implications, including the total number of males, females, and the count for each age group."
            )

            image = Image.open(uploaded_file)
            response = model.generate_content([prompt, image])

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Persons detail Analysis Results:")
                st.markdown(raw_response, unsafe_allow_html=True)  # Assuming the response is in HTML table format
            else:
                st.error("Unexpected response structure from the model.")
            return None

        def detailed_image_analysis(uploaded_file, model):
            prompt = (
                "Imagine you are an AI image analysis specialist reviewing an image for a client. Your task is to analyze the provided image and describe everything present in it with as much detail as possible. For each identified element, provide the following details: "
                "1. Objects: List and describe all objects present in the image."
                "2. People: List and describe all people in the image, including their gender, approximate age, height, clothing, facial expressions, body language, and position in the image."
                "3. Animals: List and describe any animals present in the image, including their species and any notable characteristics."
                "4. Environment: Describe the environment or setting of the image, including location type (e.g., indoor, outdoor), weather conditions (if applicable), and time of day."
                "5. Colors: Identify and describe the prominent colors used in the image."
                "6. Text: Identify and transcribe any text present in the image, noting its position and size."
                "7. Activities: Describe any activities or actions taking place in the image."
                "8. Interactions: Note any interactions between people, animals, or objects in the image."
                "9. Emotions: Describe any emotions conveyed by the people, animals, or overall scene."
                "10. Artistic Elements: Note any artistic elements such as composition, lighting, shadows, and focus."
                "11. Additional Details: Provide any additional relevant details that contribute to the overall understanding of the image."
                "\n\n"
                "Respond with the details in the following format: "
                "[Objects, People, Animals, Environment, Colors, Text, Activities, Interactions, Emotions, Artistic Elements, Additional Details]."
                "\n\n"
                "Finally, provide an overall summary of the image, summarizing the key observations and their implications."
            )

            image = Image.open(uploaded_file)
            response = model.generate_content([prompt, image])

            if response.candidates:
                raw_response = response.candidates[0].content.parts[0].text.strip()
                st.write("Detailed Image analysis Results:")
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
        st.title('Image Analysis AI Assistant')

        with st.sidebar:
            st.header("Options")
            object_analysis_button = st.button('Object Detection Analysis')
            sentiment_analysis_button = st.button('Sentiment Analysis')
            people_analysis_button = st.button('People Details Analysis')
            detailed_analysis_button = st.button('Detailed Image Analysis')
            flash_analysis_button = st.button('Flash Analysis')
            st.text_area("Enter your custom prompt for image analysis:", key="custom_prompt")
            custom_prompt_button = st.button('Custom Prompt Analysis')

        col1, col2 = st.columns(2)
        uploaded_file = col1.file_uploader("Upload your image here:")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = resize_image(image)
            col2.image(image, caption="Uploaded Image", use_column_width=True)

            if object_analysis_button:
                with st.spinner("Performing object detection analysis..."):
                    uploaded_file.seek(0)
                    object_analysis_result = object_detection_analysis(uploaded_file, model)
                    if object_analysis_result:
                        st.write("## Object Detection Analysis Results:")
                        st.markdown(object_analysis_result)

            if sentiment_analysis_button:
                with st.spinner("Performing sentiment analysis..."):
                    uploaded_file.seek(0)
                    sentiment_analysis_result = sentiment_analysis_from_image(uploaded_file, model)
                    if sentiment_analysis_result:
                        st.write("## Sentiment Analysis Results:")
                        st.markdown(sentiment_analysis_result)

            if people_analysis_button:
                with st.spinner("Performing people details analysis..."):
                    uploaded_file.seek(0)
                    people_details_result = people_details_analysis(uploaded_file, model)
                    if people_details_result:
                        st.write("## People Details Analysis Results:")
                        st.markdown(people_details_result)

            if detailed_analysis_button:
                with st.spinner("Performing detailed analysis..."):
                    uploaded_file.seek(0)
                    detailed_result = detailed_image_analysis(uploaded_file, model)
                    if detailed_result:
                        st.write("## Detailed Analysis Results:")
                        st.markdown(detailed_result)

            if flash_analysis_button:
                with st.spinner("Performing flash analysis..."):
                    uploaded_file.seek(0)
                    flash_result = flash_analysis(uploaded_file)
                    if flash_result:
                        st.write("## Flash Analysis Results:")
                        st.markdown(flash_result)

            if custom_prompt_button:
                with st.spinner("Performing custom prompt analysis..."):
                    uploaded_file.seek(0)
                    custom_prompt = st.session_state.custom_prompt
                    custom_prompt_result = custom_prompt_analysis(uploaded_file, custom_prompt, model)
                    if custom_prompt_result:
                        st.write("## Custom Prompt Analysis Results:")
                        st.markdown(custom_prompt_result)
