import streamlit as st
import pickle
import numpy as np

# Load the trained model (no scaler)
with open('hd.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app layout
st.title('Heart Disease Prediction App')
st.image('heart.jpg', width=700, caption='Early Detection Saves Lives ❤️')

# Sidebar instructions
st.sidebar.header('How to Use')
st.sidebar.markdown("""
1. Enter the patient’s medical information below.
2. Click **Predict** to find out the risk.
3. The app uses a trained ML model to help you make informed health decisions.
""")

# User input form
st.subheader('Enter Patient Details')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=45)
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)

with col2:
    restecg = st.selectbox('Resting ECG Results (restecg)', [0, 1, 2])
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])

# Prediction
if st.button('Predict'):
    input_data = np.array([[age, cp, restecg, thalach, exang]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error('⚠️ The patient is at **risk of heart disease**.')
    else:
        st.success('✅ The patient is **not at risk** of heart disease.')
