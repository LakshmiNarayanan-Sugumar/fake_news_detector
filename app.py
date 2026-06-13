import streamlit as st
import joblib
import re

# loading the saved models and vectorizer
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return lr_model, rf_model, vectorizer

lr_model, rf_model, vectorizer = load_models()

# cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'https?://\S+', "", text)
    text = re.sub(r'[^a-zA-Z\s]', "", text)
    text = re.sub(r'\s+', " ", text)
    return text

# UI
st.title("🔍 Fake News Detector")
st.write("Paste a news article below and select a model to predict whether it is real or fake.")

# model selection
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])

# text input
article = st.text_area("Paste your news article here", height=250)

# predict button
if st.button("Predict"):
    if article.strip() == "":
        st.warning("⚠️ Please paste an article first.")
    else:
        # clean and vectorize
        cleaned = clean_text(article)
        vectorized = vectorizer.transform([cleaned])

        # select model
        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(vectorized)[0]
            confidence = lr_model.predict_proba(vectorized)[0]
        else:
            prediction = rf_model.predict(vectorized)[0]
            confidence = rf_model.predict_proba(vectorized)[0]

        # show result
        if prediction == 1:
            st.success("✅ REAL News")
        else:
            st.error("🚨 FAKE News")

        # show confidence
        st.write(f"Confidence — Real: {confidence[1]:.2%} | Fake: {confidence[0]:.2%}")