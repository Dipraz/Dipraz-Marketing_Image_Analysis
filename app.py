import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

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

    # Instantiate the Generative Model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # --- STREAMLIT APP STRUCTURE ---

    st.title('🧠 AI-Powered Marketing Content Creation')
    st.write("This app helps you generate and optimize marketing content with AI. Follow the steps below to input your campaign data and create targeted marketing strategies.")

    # Step 1: Collect Marketing Campaign Data
    st.header('Step 1: Collect Marketing Campaign Data')
    st.write("Input details about your company, product, brand guidelines, marketing objectives, and target audience.")

    company_info = st.text_area(
        'Company/Product Information:', 
        placeholder="Example: Introducing 'AquaMint', a new mint-flavored sparkling water targeting health-conscious consumers."
    )
    brand_guidelines = st.text_area(
        'Brand Guidelines:', 
        placeholder="Example: The product should convey freshness and vitality. Use shades of blue and green, emphasizing the product's natural and organic nature."
    )
    marketing_objectives = st.text_area(
        'Marketing Objectives:', 
        placeholder="Example: Increase brand awareness within 6 months and capture 5% market share in the sparkling water segment within a year."
    )
    target_audience = st.text_area(
        'Target Audience:', 
        placeholder="Example: Health-conscious individuals aged 20-35 who prefer low-calorie beverages and are active on social media."
    )

    if st.button('Analyze Data'):
        with st.spinner("Analyzing data..."):
            prompt = f"Analyze the following marketing data:\n\n" \
                     f"Company/Product Information: {company_info}\n\n" \
                     f"Brand Guidelines: {brand_guidelines}\n\n" \
                     f"Marketing Objectives: {marketing_objectives}\n\n" \
                     f"Target Audience: {target_audience}\n\n" \
                     f"Provide insights and key points relevant for content creation."

            response = model.generate_content(prompt)
            st.success("Data Analysis Complete")
            st.write(response.text if response else "No response generated.")

    st.markdown("---")  # Add a horizontal divider for better design

    # Step 2: Develop Customer Personas
    st.header('Step 2: Develop Customer Personas')
    st.write("Based on the insights from the data analysis, develop detailed customer personas to tailor your marketing efforts.")

    persona_data = st.text_area(
        'Enter Data for Persona Development (e.g., insights from Step 1, additional market research):',
        placeholder="Example: The primary persona is a young professional who frequents gyms and yoga classes, follows a healthy lifestyle, and prefers eco-friendly products."
    )

    if st.button('Generate Persona'):
        with st.spinner("Generating persona..."):
            prompt = f"Based on the following data, develop a detailed customer persona:\n\n{persona_data}"
            response = model.generate_content(prompt)
            st.success("Persona Generation Complete")
            st.write(response.text if response else "No response generated.")

    st.markdown("---")

    # Step 3: Generate Content Strategy
    st.header('Step 3: Generate Content Strategy')
    st.write("Create a content strategy that aligns with the developed personas and marketing objectives.")

    strategy_data = st.text_area(
        'Enter Persona and Campaign Data for Strategy:',
        placeholder="Example: Focus on digital marketing campaigns across social media, engaging influencers in health and wellness, promoting the natural ingredients and benefits of AquaMint."
    )

    if st.button('Generate Strategy'):
        with st.spinner("Generating strategy..."):
            prompt = f"Based on the following persona and campaign data, generate a detailed content strategy:\n\n{strategy_data}"
            response = model.generate_content(prompt)
            st.success("Strategy Generation Complete")
            st.write(response.text if response else "No response generated.")

    st.markdown("---")

    # Step 4: Brainstorm Content Ideas
    st.header('Step 4: Brainstorm Content Ideas')
    st.write("Brainstorm creative content ideas that align with the strategy and persona data.")

    content_ideas_data = st.text_area(
        'Enter Strategy and Persona Data for Content Ideas:',
        placeholder="Example: Content ideas could include social media challenges, influencer partnerships, and user-generated content campaigns that highlight the refreshing taste of AquaMint."
    )

    if st.button('Brainstorm Content Ideas'):
        with st.spinner("Brainstorming content ideas..."):
            prompt = f"Based on the following strategy and persona data, brainstorm content ideas:\n\n{content_ideas_data}"
            response = model.generate_content(prompt)
            st.success("Content Idea Brainstorming Complete")
            st.write(response.text if response else "No response generated.")

    st.markdown("---")

    # Step 5: Create Marketing Content and Images
    st.header('Step 5: Create Marketing Content and Images')
    st.write("Generate marketing content based on your strategy and campaign goals.")

    content_goal = st.text_area(
        'Enter Goal for Content Creation:',
        placeholder="Example: Create social media posts and ads that emphasize the refreshing and natural qualities of AquaMint, targeting health-conscious young adults."
    )

    if st.button('Create Content'):
        with st.spinner("Creating content..."):
            prompt = f"Create marketing content based on the following goal:\n\n{content_goal}"
            response = model.generate_content(prompt)
            st.success("Content Creation Complete")
            st.write(response.text if response else "No response generated.")

    st.markdown("---")

    # Step 6: Optimize and Refine Content
    st.header('Step 6: Optimize and Refine Content')
    st.write("Optimize existing marketing content to better align with your strategy and objectives.")

    content_to_optimize = st.text_area(
        'Enter Content to Optimize:',
        placeholder="Example: Review and enhance the initial social media posts to ensure they emphasize the product's health benefits and appeal to the target audience."
    )

    if st.button('Optimize Content'):
        with st.spinner("Optimizing content..."):
            prompt = f"Optimize the following content:\n\n{content_to_optimize}"
            response = model.generate_content(prompt)
            st.success("Content Optimization Complete")
            st.write(response.text if response else "No response generated.")

    st.markdown("---")

    # Step 7: Implement and Monitor Strategy
    st.header('Step 7: Implement and Monitor Strategy')
    st.write("Track the implementation of your marketing strategy and monitor its effectiveness.")

    monitoring_data = st.text_area(
        'Enter Data to Monitor Strategy Implementation:',
        placeholder="Example: Track engagement metrics on social media, monitor feedback, and analyze sales figures to gauge the effectiveness of the marketing strategy."
    )

    if st.button('Monitor Strategy'):
        with st.spinner("Monitoring strategy..."):
            prompt = f"Analyze the implementation of the following strategy and provide insights:\n\n{monitoring_data}"
            response = model.generate_content(prompt)
            st.success("Strategy Monitoring Complete")
            st.write(response.text if response else "No response generated.")
