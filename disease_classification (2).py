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

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set background image using local file
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("hospital.jpg")  # Make sure this image is in the same folder

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Load model and data
text_df = pd.read_csv("wordcloud.csv")
with open('disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

label_dict = {
    0: 'Depression',
    1: 'Diabetes, Type 2',
    2: 'High Blood Pressure'
}

# App layout
st.markdown("<h1 style='text-align: center;'>🧠 Disease Detection & Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Input box
user_input = st.text_area("Enter a medical review (e.g. patient feedback):", height=150)

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Predict Condition"):
        cleaned_input = clean_text(user_input)
        if cleaned_input.strip():
            pred = model.predict([cleaned_input])
            label = label_dict[pred[0]]
            st.success(f"Predicted Condition: **{label}**")
        else:
            st.warning("Input is too short or not meaningful.")

with col2:
    if st.button("Analyze Sentiment"):
        polarity = TextBlob(user_input).sentiment.polarity
        if polarity > 0:
            sentiment = "Positive 😊"
        elif polarity < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"
        st.info(f"Sentiment: **{sentiment}**")

with col3:
    if st.button("Generate Input WordCloud"):
        cleaned = clean_text(user_input)
        if cleaned.strip():
            wc_user = WordCloud(width=800, height=400, background_color='black', colormap='Pastel1').generate(cleaned)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc_user, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No meaningful words to visualize.")

# Global wordcloud from dataset
if st.button("Generate WordCloud from Dataset"):
    text = " ".join(text_df['full_text'].astype(str))
    wordcloud = WordCloud(width=1000, height=600, background_color='black', colormap='Pastel1').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

