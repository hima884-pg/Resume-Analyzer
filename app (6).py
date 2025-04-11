import streamlit as st
import joblib
model = joblib.load("ResumeAnalyzer/resume_classifier_model.pkl")
vectorizer = joblib.load("ResumeAnalyzer/tfidf_vectorizer.pkl")
st.title("📄 Resume Classifier")
st.markdown("Predict the job category from resume text using a trained ML model.")
resume_text = st.text_area("Paste your resume text here:", height=300)
if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text!")
    else:
        cleaned_text = resume_text.lower()
        vec = vectorizer.transform([cleaned_text])
        prediction = model.predict(vec)
        st.success(f"**Predicted Category:** {prediction[0]}")