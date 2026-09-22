import streamlit as st
import numpy as np
import joblib

# Load Model and Scaler
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Student Pass/Fail Prediction", layout="centered")

st.title("🎓 Student Pass/Fail Prediction System")
st.write("Predict student academic success using Machine Learning based on historic performance attributes.")

# Input Form
st.header("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours (per week)", min_value=0.0, max_value=100.0, value=15.0)
    attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=80.0)
    internal_marks = st.number_input("Internal Marks (out of 50)", min_value=0.0, max_value=50.0, value=35.0)
    assignment_marks = st.number_input("Assignment Marks (out of 50)", min_value=0.0, max_value=50.0, value=40.0)

with col2:
    prev_marks = st.number_input("Previous Exam Marks (out of 100)", min_value=0.0, max_value=100.0, value=65.0)
    practice_score = st.number_input("Practice Test Scores (out of 100)", min_value=0.0, max_value=100.0, value=70.0)
    backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)

if st.button("Predict Result"):
    # Input Processing
    input_data = np.array([[study_hours, attendance, internal_marks, assignment_marks, prev_marks, practice_score, backlogs]])
    scaled_input = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    st.subheader("🎯 Prediction Output")
    if prediction == 1:
        st.success(f"**Result: PASS** 🎉 (Confidence: {probabilities[1]*100:.2f}%)")
    else:
        st.error(f"**Result: FAIL** ⚠️ (Confidence: {probabilities[0]*100:.2f}%)")
        
    st.write(f"**Pass Probability:** {probabilities[1]*100:.2f}% | **Fail Probability:** {probabilities[0]*100:.2f}%")
    
    # Model Interpretation
    st.markdown("---")
    st.subheader("💡 Result Interpretation & Key Insights")
    st.write("""
    - **Impact Factors:** Higher study hours, attendance, and internal marks heavily contribute to a passing outcome.
    - **Risk Factors:** A high number of backlogs significantly decreases the probability of passing.
    - **Confusion Matrix Context:** The model balances precision and recall to minimize misclassifying at-risk students.
    """)