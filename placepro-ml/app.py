# To run the app, use the command:
# streamlit run app.py
# To train the model, you would typically have a separate script like 'train_model.py'

import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("placement_model.pkl", "rb"))

st.title("🎓 Placement Prediction System")

ssc = st.slider("SSC Percentage", 40, 100, 75)
hsc = st.slider("HSC Percentage", 40, 100, 75)
deg = st.slider("Degree Percentage", 40, 100, 75)
work_exp = st.selectbox("Work Experience", ["Yes", "No"])
test_score = st.slider("Aptitude Test Score", 30, 100, 70)
skills = st.selectbox("Primary Skill", ["Python", "Java", "C++", "SQL", "None"])
internships = st.slider("Internships Done", 0, 5, 1)
comm_skill = st.slider("Communication Skill (1-10)", 1, 10, 7)
extra_curr = st.selectbox("Extra Curriculars", ["Yes", "No"])

we = 1 if work_exp == "Yes" else 0
ec = 1 if extra_curr == "Yes" else 0
skill_map = {"Python": 3, "Java": 2, "C++": 0, "SQL": 4, "None": 1}
skill_val = skill_map.get(skills, 1)

input_data = np.array([[ssc, hsc, deg, we, test_score, skill_val,
                        internships, comm_skill, ec]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Placed 🎉" if prediction == 1 else "Not Placed ❌"
    st.success(f"Prediction: {result}")

# To run this app, you need to have the model saved as 'placement_model.pkl'.
# You can train the model using a script like 'train_model.py' and save it using
# pickle. Make sure to have the necessary libraries installed:
# pip install streamlit pandas numpy scikit-learn
# To run the app, use the command:
# streamlit run app.py
# To train the model, you would typically have a separate script like 'train_model.py'
# that handles the training process and saves the model as 'placement_model.pkl'.
# Note: Ensure that the model is trained with the same features as used in this app.
# The model should be trained on a dataset that includes these features and the target variable.
# The dataset should be preprocessed similarly to how the input data is structured here.
# This app is a simple demonstration of how to use a machine learning model for predictions.
# Make sure to handle exceptions and edge cases in a production environment.
