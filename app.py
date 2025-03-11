# Import necessary libraries
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Model and vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Text preprocessing function
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Enter a movie review to predict whether it's positive or negative!")

# User input
review = st.text_area("Enter your movie review:", "I loved this movie! The acting was amazing and the plot kept me engaged.")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.error("Please enter a review before predicting!")
    else:
        # Preprocess the input
        processed_review = preprocess_text(review)
        
        # Transform the input using the TF-IDF vectorizer
        review_tfidf = vectorizer.transform([processed_review])
        
        try:
            # Predict sentiment
            prediction = model.predict(review_tfidf)[0]
            probability = model.predict_proba(review_tfidf)[0]
            
            # Display results
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probability[prediction] * 100
            
            st.markdown(f"**Predicted Sentiment:** {sentiment}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Add a visual indicator
            if sentiment == "Positive":
                st.success("🎉 This review is positive!")
            else:
                st.error("👎 This review is negative!")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Python. Check out the code on [GitHub](https://github.com/yourusername/sentiment-analysis)!")