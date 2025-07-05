import streamlit as st
import numpy as np
import joblib

# Load the saved model, vectorizer, and label encoder
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="📝 Sentiment Analysis App", layout="centered")
st.title("📝 Sentiment Analysis App")
st.markdown("Enter a sentence below. The model will predict the **sentiment** and show **confidence**.")

user_input = st.text_area("Enter your text:", "")

if user_input.strip():
    user_input_tfidf = vectorizer.transform([user_input])
    prediction = best_model.predict(user_input_tfidf)
    predicted_num = prediction[0]
    predicted_label = "Positive" if predicted_num == 1 else "Negative"

    if hasattr(best_model, "predict_proba"):
        probas = best_model.predict_proba(user_input_tfidf)[0]
        confidence = np.max(probas) * 100
    else:
        score = best_model.decision_function(user_input_tfidf)
        confidence = (1 / (1 + np.exp(-np.abs(score))))[0] * 100

    st.subheader(f"Predicted Sentiment for input text: **{predicted_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.progress(confidence / 100)
