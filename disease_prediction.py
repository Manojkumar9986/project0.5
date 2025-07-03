import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle, cloudpickle
import streamlit as st
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('wordnet')
# Define the cleaning function exactly as before
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z:]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

tab0, tab1 = st.tabs(["WordCloud","Predict Disease"])

text_df = pd.read_csv("worldcloud.csv")
tfidf_path = 'tfidf_vector.pkl'
with open(tfidf_path, 'rb') as file:
    tfidf_vect = pickle.load(file)

model_path = 'disease_model.pkl'
with open(model_path, 'rb') as file1:
    model = cloudpickle.load(file1) # for custom function in pipeline

label_dict = {
  0: 'Depression',
  1: 'Diabetes, Type 2',
  2: 'High Blood Pressure'
}
with tab0:
    st.title('WordCloud')
    if st.button("Generate Word Cloud"):
        text = " ".join(text_df['full_text'].astype(str))
        # Generate word cloud
        wordcloud = WordCloud(width=1000, height=600, background_color='black', colormap='Pastel1').generate(text)
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud)
        ax.axis('off')
        # Display in Streamlit
        st.pyplot(fig)

with tab1:
    st.title('🧠 Disease Prediction With Review')
    st.markdown("""
            > ⚠️ **Note:** This model is trained only on three conditions: **Depression**, **Type 2 Diabetes**, 
            and **High Blood Pressure**.
            >
            > If your review describes a different condition or treatment (e.g., birth control, acne, 
            or general wellness), the model may still predict one of the known conditions if similar terms 
            are present.
            >
            > This is a limitation of the model, and unrelated reviews may be misclassified.
            """)

    user_input = st.text_area(
        "Enter your detailed review:",
        height=100
    )

    if st.button("Predict"):
        if not user_input.strip():
            st.error("Please enter a valid review.")
        else:
            user_vec = tfidf_vect.transform([user_input]) 
            if user_vec.nnz == 0:
                st.error("Invalid Input")
            else:
                scores = model.decision_function([user_input])
                max_score = max(scores[0])
                threshold = 0.5  # 🔧 adjust based on testing
            
                if max_score < threshold:
                    st.warning("The model is not confident. This review may not match any known condition.")
                else:
                    label = model.predict([user_input])[0]
                    st.success(f"**Predicted Condition:** {label_dict[label]}")


    else:
        st.info("Please enter a review to predict the condition.")
        
    if st.button("Analyze Sentiment"):
        blob = TextBlob(user_input)
        # Get sentiment polarity
        polarity = blob.sentiment.polarity
        # Interpretation
        if polarity > 0:
            sentiment = "Positive 😊"
        elif polarity < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        st.write(f"**Sentiment:** {sentiment}")
        
    if st.button('Generate WordCloud'):
        cleaned = clean_text(user_input)
        if cleaned.strip():  # Check that text is not empty after cleaning
            st.markdown('**WordCloud for User Input**')
            wordcloud_user = WordCloud(width=800, height=400, background_color='black',
                                       colormap='Pastel1').generate(cleaned)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.imshow(wordcloud_user)
            ax1.axis('off')
            st.pyplot(fig1)
        else:
            st.warning("No meaningful words left after cleaning. Try entering more descriptive text.")
