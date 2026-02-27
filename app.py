import streamlit as st
import pandas as pd
import re
import nltk
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️", layout="centered")

# Custom CSS for animations
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("cyberbullying_dataset.csv")

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['clean_text'] = data['text'].apply(preprocess)

# ---------------------------
# TF-IDF + Logistic Regression
# ---------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

# ---------------------------
# UI
# ---------------------------
st.title("🛡️ Cyberbullying Detection App")
st.write("Detect whether a sentence contains bullying content.")

user_input = st.text_area("✍️ Enter a sentence")

if st.button("🔍 Analyze Text"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

        clean_input = preprocess(user_input)
        vector_input = vectorizer.transform([clean_input])

        prediction = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)[0]

        bullying_index = list(model.classes_).index("bullying")
        safe_index = list(model.classes_).index("not_bullying")

        bullying_prob = probabilities[bullying_index]
        safe_prob = probabilities[safe_index]

        bullying_percent = round(bullying_prob * 100, 2)
        safe_percent = round(safe_prob * 100, 2)

        st.subheader("📊 Prediction Result")

        # Animated Progress Bars
        bullying_bar = st.progress(0)
        safe_bar = st.progress(0)

        for i in range(int(bullying_percent) + 1):
            bullying_bar.progress(i)
            time.sleep(0.01)

        for i in range(int(safe_percent) + 1):
            safe_bar.progress(i)
            time.sleep(0.01)

        st.write(f"🚨 **Bullying Probability:** {bullying_percent}%")
        st.write(f"✅ **Safe Probability:** {safe_percent}%")

        # Final Result
        if prediction == "bullying":
            st.error("⚠️ This text is classified as Bullying.")
        else:
            st.success("🎉 This text is classified as Safe.")

        st.balloons()