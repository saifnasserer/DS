import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("final_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"(http\S+|www\S+)", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def preprocess(text):
    words = clean_text(text).split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("🚨 Emergency Tweet Sentiment Classifier")
st.write("Enter a tweet and get sentiment prediction")

tweet = st.text_area("Type tweet here")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter text")
    else:
        processed = preprocess(tweet)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ Positive Tweet")
        else:
            st.error("🚨 Negative / Critical Tweet")
