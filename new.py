import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel(""gemini-2.5-flash"")

st.set_page_config(page_title="Multimodal Compliance AI", layout="wide")
st.title("📊 Multimodal Document & Compliance Analysis with Gemini 2.5 Flash")

# -----------------------------
# 🔍 Generic Analysis Section
# -----------------------------
with st.expander("🔎 General Media Analysis (Prompt + Any File)", expanded=True):
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="gen_images")
    uploaded_pdfs = st.file_uploader("Upload PDF Books", type=["pdf"], accept_multiple_files=True, key="gen_pdfs")
    uploaded_media = st.file_uploader("Upload Regulatory Media (e.g., TXT, CSV, etc.)", accept_multiple_files=True, key="gen_media")

    # ✅ Show image previews for general analysis
    if uploaded_images:
        st.markdown("🖼️ **Preview of Uploaded Images:**")
        for img in uploaded_images:
            st.image(img, caption=img.name, use_container_width=True)

    prompt = st.text_area("Enter your custom prompt for analysis:", height=200, key="gen_prompt")

    if st.button("Run General Analysis", key="gen_button"):
        if not (uploaded_images or uploaded_pdfs or uploaded_media):
            st.warning("Please upload at least one file.")
        elif not prompt:
            st.warning("Please enter a prompt for analysis.")
        else:
            contents = [prompt]

            for image_file in uploaded_images:
                img_bytes = image_file.getvalue()
                contents.append({"mime_type": image_file.type, "data": img_bytes})

            for pdf_file in uploaded_pdfs:
                pdf_bytes = pdf_file.getvalue()
                contents.append({"mime_type": "application/pdf", "data": pdf_bytes})

            for media_file in uploaded_media:
                contents.append({"mime_type": media_file.type, "data": media_file.getvalue()})

            try:
                response = model.generate_content(contents)
                st.subheader("🧠 Analysis Result:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# -----------------------------
# ✅ Compliance Checker Section
# -----------------------------
st.markdown("---")
st.subheader("🛡️ Compliance Checker: Compare Media Against Regulations")

rulebooks = st.file_uploader("📚 Upload Rulebooks (PDF)", type=["pdf"], accept_multiple_files=True, key="rules")
media_files = st.file_uploader("🖼️ Upload Media Files (Images, PDFs, PPTX, etc.)", accept_multiple_files=True, key="media")

# ✅ Show image previews in the compliance section
if media_files:
    image_extensions = ["jpg", "jpeg", "png"]
    st.markdown("🖼️ **Preview of Uploaded Media Images:**")
    for media in media_files:
        if any(media.name.lower().endswith(ext) for ext in image_extensions):
            st.image(media, caption=media.name, use_container_width=True)

compliance_prompt = st.text_area(
    "🔧 Optional: Custom compliance prompt (e.g., 'Highlight all ad claims that may violate health disclaimers')",
    height=150,
    key="compliance_prompt"
)

if st.button("Analyze for Compliance"):
    if not (rulebooks and media_files):
        st.warning("Upload both rulebooks and media files.")
    else:
        base_context = ["Analyze these materials for rule compliance. Detect violations and suggest improvements."]
        if compliance_prompt:
            base_context.append(compliance_prompt)

        for rule_pdf in rulebooks:
            base_context.append({"mime_type": "application/pdf", "data": rule_pdf.getvalue()})

        for media in media_files:
            base_context.append({"mime_type": media.type, "data": media.getvalue()})

        try:
            response = model.generate_content(base_context)
            st.subheader("📋 Compliance Report:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Error during compliance check: {e}")

# -----------------------------
# ❓ Ask Rulebook-Only Questions
# -----------------------------
st.markdown("---")
st.subheader("💬 Ask a Question Based on Rulebooks")

query = st.text_input(
    "Enter your question about the rules (e.g., 'Are health-related claims allowed in product ads?')",
    key="rule_query"
)

if st.button("Ask Rulebook", key="query_button"):
    if not rulebooks:
        st.warning("Please upload rulebooks first.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        context = [query]
        for rule_pdf in rulebooks:
            context.append({"mime_type": "application/pdf", "data": rule_pdf.getvalue()})

        try:
            response = model.generate_content(context)
            st.subheader("📘 Answer from Rulebook:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Error while querying rulebook: {e}")


