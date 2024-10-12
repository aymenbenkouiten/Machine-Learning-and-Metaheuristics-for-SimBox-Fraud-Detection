import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd 
from sklearn.preprocessing import PowerTransformer

st.set_page_config(page_title="Prediction", page_icon="🔎", layout='centered')

features = [
    "AVG_CALL_FREQ", "AVG_DAILY_CALLS", "AVG_DAILY_CALLS_DUR", "CALLTYPE_0_RATIO",
    "CALLTYPE_1_RATIO", "CALLTYPE_6_RATIO", "CALLTYPE_7_RATIO", "INTERNATIONAL_RATIO",
    "MAX_CALL_FREQ", "MAX_DAILY_CALLS", "MAX_DAILY_CALLS_DUR", "MIN_CALL_FREQ",
    "MIN_DAILY_CALLS", "MIN_DAILY_CALLS_DUR", "OFF_NET_RATIO", "ON_NET_RATIO",
    "STD_CALL_FREQ"
]

st.title("Fraud Detection Prediction")

# Load models from the models folder
model_folder = "models"
model_files = [f for f in os.listdir(model_folder) if f.endswith(".joblib")]

if not model_files:
    st.error("No model files found in the 'models' folder.")
else:
    selected_model_file = st.selectbox("Select a model", model_files)

    # Load the selected model
    model_path = os.path.join(model_folder, selected_model_file)
    model = joblib.load(model_path)

    st.header("Input Features")

    input_data = []
    for feature in features:
        value = st.number_input(feature, value=0.0)
        input_data.append(value)

    input_data = np.array(input_data).reshape(1, -1)

    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    transformed = pt.fit_transform(input_data)
    transformed_df = pd.DataFrame(transformed, columns=features)


    if st.button("Predict"):
        prediction = model.predict(transformed_df)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        st.subheader(f"Prediction: {result}")