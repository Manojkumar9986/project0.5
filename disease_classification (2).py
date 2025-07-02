import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import base64

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set hospital-themed background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main-box {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background (image must be in the same folder)
set_background("hospital.jpg")

# Load model and CSV
@st.cache_resource
def load_model():
    with open("disease_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("wordcloud.csv")

model = load_model()
text_df = load_data()

# Setup NLP tools
stop_words = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Label mapping
label_dict = {
    0: "Depression",
    1: "Diabetes, Type 2",
    2: "High Blood Pressure"
}

# App title
st.markdown("<h1 style='text-align:center;'>🩺 Disease Predictor & Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<div class='main-box'>", unsafe_allow_html=True)

# Input review
user_input = st.text_area("✍️ Enter a patient's review:", height=150)

# Action buttons
col1, col2, col3 = st.columns(3)

# Prediction
with col1:
    if st.button("🔍 Predict Condition"):
        cleaned_input = clean_text(user_input)
        if cleaned_input.strip():
            prediction = model.predict([cleaned_input])[0]
            st.success(f"🧠 Predicted Condition: **{label_dict[prediction]}**")
        else:
            st.warning("Please enter valid descriptive text.")

# Sentiment Analysis
with col2:
    if st.button("💬 Analyze Sentiment"):
        if user_input.strip():
            polarity = TextBlob(user_input).sentiment.polarity
            sentiment = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"
            st.info(f"**Sentiment:** {sentiment}")
        else:
            st.warning("Please enter a review to analyze.")

# WordCloud from Input
with col3:
    if st.button("☁️ WordCloud (Input)"):
        cleaned = clean_text(user_input)
        if cleaned.strip():
            wc = WordCloud(width=800, height=400, background_color="black", colormap="Pastel1").generate(cleaned)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No meaningful words to visualize.")

# WordCloud from Dataset
if st.button("📊 WordCloud (Dataset)"):
    all_text = " ".join(text_df["full_text"].astype(str))
    wc_all = WordCloud(width=1000, height=500, background_color="black", colormap="Pastel1").generate(all_text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc_all, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

