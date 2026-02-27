import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f8fafc, #eef2f7);
}
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 15px;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
.safe-box {
    background-color: #e6f4ea;
    color: #1b5e20;
}
.bully-box {
    background-color: #fdecea;
    color: #b71c1c;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛡️ Cyberbullying Detection System")
st.write("AI-powered system to analyze text and detect harmful online content.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- USER INPUT ----------------
text = st.text_area("Enter Text to Analyze", height=150)

if st.button("Analyze Text"):

    if text.strip() == "":
        st.warning("Please enter valid text.")
    else:
        with st.spinner("Analyzing with Transformer model..."):
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            probabilities = probs.detach().numpy()[0]

            negative_prob = float(probabilities[0])  # considered bullying
            positive_prob = float(probabilities[1])  # considered safe

            bullying_percent = round(negative_prob * 100, 2)
            safe_percent = round(positive_prob * 100, 2)

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-box bully-box">
                🚨 Bullying Probability<br>
                {bullying_percent}%
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box safe-box">
                ✅ Safe Probability<br>
                {safe_percent}%
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if bullying_percent > safe_percent:
            st.error("⚠️ The text is likely to contain harmful or negative content.")
        else:
            st.success("✅ The text appears safe and non-harmful.")

        confidence = max(bullying_percent, safe_percent)
        st.info(f"Model Confidence: {confidence}%")
